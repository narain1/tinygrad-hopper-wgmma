"""Warp-specialized bf16 GEMM for Hopper (sm_90a), written as tinygrad UOps.

Layers, bottom to top:
  1. primitives  one function per PTX feature, each returns one effect UOp. An op that must run after ``x`` takes
                 ``buf.after(x)[...]`` as its address operand (tinygrad has no statement order, only data dependencies).
  2. helpers     ``Ring`` (an N-stage shared-memory queue made of FULL/EMPTY mbarrier pairs), ``region`` (an ``if`` for a
                 warp role), tile orders (grouped / 1D GROUP_M / Hilbert).
  3. builder     ``build_gemm_ws(N, WSCfg(...))`` -> ``run(A, B, C)`` -> ``launch()``.

The one trick that makes warp specialization expressible: tinygrad has no if/else, but a RANGE may have a symbolic trip
count. ``range(producer ? n : 0)`` is a loop only the producer runs, so each role gets its own loop body.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

SMEM_LIMIT = 227 * 1024   # opt-in dynamic shared memory per block on H100/H200


# =============================================================================================================
# 1. primitives
# =============================================================================================================
def _custom(fmt: str, *srcs: UOp, gate: UOp | None = None) -> UOp:
  """A CUSTOM effect: C source with {i} = rendered srcs[i] (literal braces doubled). Always a braced block, so two
  customs in one scope can declare the same locals. gate: an optional bool UOp, renders as ``if(gate) {...}``."""
  fmt, srcs = ("if({%d}) {{ " % len(srcs) + fmt + " }}", srcs + (gate,)) if gate is not None else ("{{ " + fmt + " }}", srcs)
  return UOp(Ops.CUSTOM, tuple(srcs), arg=(fmt, dtypes.void))


def _sa(i: int) -> str:
  return "(unsigned)__cvta_generic_to_shared({%d})" % i   # shared-window address of src i


# ---- mbarrier ----
def mbar_init(bar: UOp, count: int, gate: UOp | None) -> UOp:
  return _custom('asm volatile("mbarrier.init.shared::cta.b64 [%%0], %d;" :: "r"(%s) : "memory");' % (count, _sa(0)), bar, gate=gate)


def fence_mbar_init(dep: UOp, gate: UOp | None) -> UOp:
  return _custom('asm volatile("fence.mbarrier_init.release.cluster; fence.proxy.async.shared::cta;" ::: "memory"); (void){0};',
                 dep, gate=gate)


def mbar_expect_tx(bar: UOp, nbytes: int, gate: UOp | None) -> UOp:
  """arrive once and add ``nbytes`` to the transaction count: the phase completes when that many TMA bytes landed"""
  return _custom('asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%%0], %d;" :: "r"(%s) : "memory");' % (nbytes, _sa(0)),
                 bar, gate=gate)


def mbar_arrive(bar: UOp, gate: UOp | None) -> UOp:
  return _custom('asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%%0];" :: "r"(%s) : "memory");' % _sa(0), bar, gate=gate)


def mbar_arrive_cluster(bar: UOp, cta: UOp, gate: UOp | None) -> UOp:
  """arrive on the barrier at the same SMEM offset in CTA ``cta`` of this cluster (mapa = map into that CTA's window)"""
  return _custom('asm volatile("{{ .reg .b32 ra; mapa.shared::cluster.u32 ra, %%0, %%1; '
                 'mbarrier.arrive.shared::cluster.b64 _, [ra]; }}" :: "r"(%s), "r"((unsigned)({1})) : "memory");' % _sa(0),
                 bar, cta, gate=gate)


_label = [0]


def mbar_wait(bar: UOp, phase: UOp, gate: UOp | None) -> UOp:
  """spin until the barrier's phase with parity ``phase`` has completed. Each wait gets its own PTX label."""
  _label[0] += 1
  label = f"TGW{_label[0]}"
  return _custom('asm volatile("{{ .reg .pred P; %s: mbarrier.try_wait.parity.shared::cta.b64 P, [%%0], %%1; @!P bra %s; }}" '
                 ':: "r"(%s), "r"((unsigned)({1})) : "memory");' % (label, label, _sa(0)), bar, phase, gate=gate)


# ---- TMA ----
def tma_load(dst: UOp, tmap: UOp, bar: UOp, row: UOp, kcol: UOp, gate: UOp | None) -> UOp:
  """3D tensor map (64 inner elements, rows, K/64), 128B swizzle: copy the box at (0, row, kcol/64) into dst and
  complete its bytes on bar"""
  return _custom('asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes'
                 ' [%%0], [%%1, {{0, %%3, %%4}}], [%%2];" :: "r"(%s), "l"((unsigned long)({1})), "r"(%s), "r"((int)({3})),'
                 ' "r"((int)({4})/64) : "memory");' % (_sa(0), _sa(2)), dst, tmap, bar, row, kcol, gate=gate)


def tma_load_multicast(dst: UOp, tmap: UOp, bar: UOp, row: UOp, kcol: UOp, mask: int, gate: UOp | None) -> UOp:
  """one L2 read, written to the same SMEM offset (and completed on the same barrier) in every CTA of ``mask``"""
  return _custom('asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster'
                 ' [%%0], [%%1, {{0, %%3, %%4}}], [%%2], %%5;" :: "r"(%s), "l"((unsigned long)({1})), "r"(%s), "r"((int)({3})),'
                 ' "r"((int)({4})/64), "h"((unsigned short)%d) : "memory");' % (_sa(0), _sa(2), mask), dst, tmap, bar, row, kcol, gate=gate)


def epilogue_tma_store(sC: UOp, acc: UOp, tmapC: UOp, row: UOp, col: UOp, ci: UOp, tid: UOp, BN: int) -> UOp:
  """registers -> bf16 -> stmatrix into a 128B-swizzled SMEM tile -> one TMA bulk store per consumer warpgroup.
  Named barrier ci+1 syncs only this warpgroup. The store stays async: the next tile only waits for the TMA engine to
  finish *reading* sC (wait_group.read) before overwriting it."""
  return _custom(
    'float* d={1}; unsigned base=(unsigned)__cvta_generic_to_shared({0}); int tid=(int)({6}), lane=tid%%32, warp=tid/32, bar=(int)({5})+1;'
    ' if(tid==0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");'
    ' asm volatile("bar.sync %%0, 128;" :: "r"(bar) : "memory");'
    ' int mi=lane/8, r=warp*16+(mi&1)*8+(lane&7);'
    ' _Pragma("unroll") for(int w=0; w<%d; w++){{ int c=w*16+(mi>>1)*8;'
    ' unsigned addr=base+(c/64)*8192+r*128+((((c%%64)/8)^(r&7))<<4);'
    ' __nv_bfloat162 v0=__floats2bfloat162_rn(d[w*8+0],d[w*8+1]), v1=__floats2bfloat162_rn(d[w*8+2],d[w*8+3]),'
    ' v2=__floats2bfloat162_rn(d[w*8+4],d[w*8+5]), v3=__floats2bfloat162_rn(d[w*8+6],d[w*8+7]);'
    ' asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%%0], {{%%1, %%2, %%3, %%4}};" :: "r"(addr),'
    ' "r"(*(unsigned*)&v0), "r"(*(unsigned*)&v1), "r"(*(unsigned*)&v2), "r"(*(unsigned*)&v3) : "memory"); }}'
    ' asm volatile("fence.proxy.async.shared::cta;" ::: "memory");'
    ' asm volatile("bar.sync %%0, 128;" :: "r"(bar) : "memory");'
    ' if(tid==0){{ asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%%0, {{0, %%2, %%3}}], [%%1];"'
    ' :: "l"((unsigned long)({2})), "r"(base), "r"((int)({3})), "r"((int)({4})/64) : "memory");'
    ' asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }}' % (BN // 16),
    sC, acc, tmapC, row, col, ci, tid)


def tma_store_drain(dep: UOp, keep: UOp, gate: UOp | None) -> UOp:
  """wait for every outstanding bulk store of this thread (SMEM must outlive the last store's reads).
  keep: the output buffer, referenced so it stays a kernel argument when it is only written through a tensor map."""
  return _custom('asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); (void){0}; (void){1};', dep, keep, gate=gate)


# ---- misc ----
def cluster_sync(dep: UOp) -> UOp:
  """every thread of every CTA in the cluster; release/acquire publishes barrier inits and DSMEM writes cluster-wide"""
  return _custom('asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); (void){0};', dep)


def wgmma_wait(n: int, dep: UOp) -> UOp:
  """wgmma.wait_group n: block until at most n committed wgmma groups of this warpgroup are still running"""
  return _custom('asm volatile("wgmma.wait_group.sync.aligned %d;" ::: "memory"); (void){0};' % n, dep)


def setmaxnreg(n: int, inc: bool, dep: UOp, gate: UOp | None) -> UOp:
  """EXPERIMENTAL. ptxas only honors this when everything after it is dominated by it (use inside a region(), not under
  a gate). ``build_gemm_ws`` gates it, so ``WSCfg.regs`` has no measurable effect (see the README)."""
  return _custom('asm volatile("setmaxnreg.%s.sync.aligned.u32 %d;"); (void){0};' % ("inc" if inc else "dec", n), dep, gate=gate)


def store_acc_bf16(C: UOp, acc: UOp, row0: UOp, col0: UOp, ld: int, BN: int, gate: UOp | None) -> UOp:
  """register epilogue: this thread's m64nBN accumulator fragment -> bf16x2 stores into row-major C.
  row0 = first row of the thread's (row, row+8) pair, col0 = first column (2*(tid%4) already added)."""
  return _custom('float* d={1}; __nv_bfloat162* c=(__nv_bfloat162*)({0}); long r=(long)({2}), cc=(long)({3});'
                 ' _Pragma("unroll") for(int w=0; w<%d; w++){{ long o=(r*%d+cc+w*16)/2;'
                 ' c[o]=__floats2bfloat162_rn(d[w*8+0],d[w*8+1]); c[o+%d]=__floats2bfloat162_rn(d[w*8+2],d[w*8+3]);'
                 ' c[o+4]=__floats2bfloat162_rn(d[w*8+4],d[w*8+5]); c[o+%d+4]=__floats2bfloat162_rn(d[w*8+6],d[w*8+7]); }}'
                 % (BN // 16, ld, 4 * ld, 4 * ld), C, acc, row0, col0, gate=gate)


# =============================================================================================================
# 2. helpers
# =============================================================================================================
def region(cond: UOp, axis: int) -> UOp:
  """``if (cond) { ... }`` for a warp role: a loop that runs once when cond holds, zero times otherwise. Close it with
  ``.end(r)``. Unlike a gate, the body is a real basic block (no per-instruction predicates)."""
  return UOp.range(cond.where(1, 0), axis, AxisType.LOOP)


class Ring:
  """A ``stages``-deep SMEM queue between one producer thread and ``consumers`` consumer warpgroups.
  Stage s has a FULL barrier (count 1: the producer's expect_tx; TMA bytes complete it) and an EMPTY barrier
  (count consumers * cluster: one arrival per consumer warpgroup of every CTA that receives the stage).
  Iteration j uses stage j % stages with phase parity (j // stages) % 2, on both sides.

  Producer:  wait(empty, j) -> expect_tx(full) -> TMA loads into the stage
  Consumer:  wait(full, j)  -> compute         -> release(stage)
  ``prime`` makes every stage start out empty (one release per consumer), so the producer's first waits pass."""
  BYTES_PER_STAGE = 16                                    # two 8-byte mbarriers

  def __init__(self, smem: UOp, bar_off: int, stages: int, consumers: int, cluster: int = 1):
    self.smem, self.bar_off, self.stages, self.consumers, self.cluster = smem, bar_off, stages, consumers, cluster

  def full(self, s, dep: UOp) -> UOp: return self.smem.after(dep)[self.bar_off + s * 4]
  def empty(self, s, dep: UOp) -> UOp: return self.smem.after(dep)[self.bar_off + (self.stages + s) * 4]
  def slot(self, j: UOp) -> tuple[UOp, UOp]: return j % self.stages, (j // self.stages) % 2

  def inits(self, gate: UOp) -> list[UOp]:
    return [mbar_init(self.full(s, self.smem), 1, gate) for s in range(self.stages)] + \
           [mbar_init(self.empty(s, self.smem), self.consumers * self.cluster, gate) for s in range(self.stages)]

  def release(self, s, dep: UOp, tid: UOp, gate: UOp | None = None) -> UOp:
    """one arrival per consumer warpgroup on stage s. With a cluster the stage was multicast into every CTA, so
    consumer thread r (< cluster) arrives on CTA r's empty barrier."""
    if self.cluster == 1: return mbar_arrive(self.empty(s, dep), tid.eq(0) if gate is None else gate & tid.eq(0))
    lead = tid < self.cluster
    return mbar_arrive_cluster(self.empty(s, dep), tid, lead if gate is None else gate & lead)

  def prime(self, dep: UOp, tid: UOp, is_cons: UOp) -> UOp:
    # gate on is_cons: the producer's thread also has tid == 0, and an extra arrival on the count-`consumers` barrier
    # would let the producer overwrite a stage one phase early
    return UOp.group(*[self.release(s, dep, tid, is_cons) for s in range(self.stages)])


def init_barriers(smem: UOp, rings: list[Ring], t: UOp, cluster: int = 1) -> UOp:
  """thread 0 initializes every ring's barriers, fence, block barrier (+ cluster barrier: peers' barriers must exist
  before anyone arrives on them remotely). Returns the token everything else orders after."""
  tok = UOp.group(*[i for r in rings for i in r.inits(t.eq(0))])
  tok = UOp.barrier(fence_mbar_init(smem.after(tok)[0], t.eq(0)))
  return cluster_sync(smem.after(tok)[0]) if cluster > 1 else tok


def grouped_tile_order(CMT: int, NT: int, GM: int, GN: int):
  """linear tile id -> (tile row, tile col), L2 friendly. If GM x GN super-tiles divide the grid: row-major over
  super-tiles, row-major inside one (fast.cu Schedule<1>). Else Triton-style GROUP_M: GM tile rows swept across all
  NT columns, column-major inside the group, last group ragged."""
  if CMT % GM == 0 and NT % GN == 0:
    def coords(tile: UOp):
      st, pos = tile // (GM * GN), tile % (GM * GN)
      return (st // (NT // GN)) * GM + pos // GN, (st % (NT // GN)) * GN + pos % GN
    return coords, True

  def coords1d(tile: UOp):
    first = (tile // (GM * NT)) * GM
    rows = (first > CMT - GM).where(CMT - first, GM)
    r = tile % (GM * NT)
    return first + r % rows, r // rows
  return coords1d, False


def hilbert_schedule(M: int, N: int, cores: int) -> list[int]:
  """fast.cu createHilbert: walk an M x N tile grid along a Hilbert curve and deal the tiles round-robin to ``cores``,
  so at step i the cores work on curve positions [i*cores, (i+1)*cores): a compact patch of A rows and B cols (L2).
  Returns a flat [cores][M*N/cores] table of (m << 16) | n."""
  def d2xy(n, d):
    x = y = 0; s = 1; t = d
    while s < n:
      rx = 1 & (t // 2); ry = 1 & (t ^ rx)
      if ry == 0:
        if rx == 1: x, y = s - 1 - x, s - 1 - y
        x, y = y, x
      x += s * rx; y += s * ry; t //= 4; s *= 2
    return x, y
  dim = 1
  while dim < max(M, N): dim *= 2
  pos: list[list[int]] = [[] for _ in range(cores)]
  core = 0
  for i in range(dim * dim):
    x, y = d2xy(dim, i)
    if x < M and y < N: pos[core].append((x << 16) | y); core = (core + 1) % cores
  if any(len(p) != len(pos[0]) for p in pos):
    raise ValueError(f"hilbert needs an equal number of tiles per core ({M}x{N} tiles over {cores} cores)")
  return [v for p in pos for v in p]


# =============================================================================================================
# 3. warp-specialized GEMM
# =============================================================================================================
@dataclass(frozen=True)
class WSCfg:
  BN: int = 256                # wgmma N, one instruction per 64 K
  BK: int = 64                 # one 128B swizzle atom along K (the only supported value)
  stages: int = 3              # ring depth
  consumers: int = 2           # consumer warpgroups, 64 rows each: BM = 64 * consumers
  num_sm: int = 128            # persistent CTAs. 512 tiles / 128 = exactly 4 per CTA at N=4096 (a tuning choice, not a device query)
  group_m: int = 16            # L2 tile order, super-tiles of group_m x group_n tiles
  group_n: int = 8
  tma_store: bool = False      # stmatrix + async TMA store epilogue
  regs: tuple[int, int] | None = None   # EXPERIMENTAL setmaxnreg (producer, consumer); no measurable effect, see README
  cluster: int = 1             # 2: a CTA pair shares one multicast B tile
  hilbert: bool = False        # Hilbert tile order from a host-built table
  inflight: bool = False       # EXPERIMENT: keep one wgmma group in flight (wait_group 1), release the previous stage late

  def __post_init__(self):
    if self.BK != 64: raise ValueError("BK must be 64 (one 128-byte swizzle atom)")
    if self.BN <= 0 or self.BN % 16 or self.BN > 256: raise ValueError("BN must be a multiple of 16 in (0, 256]")
    if self.tma_store and self.BN % 64: raise ValueError("tma_store needs BN to be a multiple of 64")
    if self.stages < 2: raise ValueError("stages must be at least 2")
    if self.consumers not in (1, 2): raise ValueError("consumers must be 1 or 2")
    if self.cluster not in (1, 2): raise ValueError("cluster must be 1 or 2")
    if self.num_sm < self.cluster: raise ValueError("num_sm must be at least the cluster size")
    if self.group_m < 1 or self.group_n < 1: raise ValueError("group_m and group_n must be positive")
    if self.regs is not None and (len(self.regs) != 2 or any(r % 8 or not 24 <= r <= 256 for r in self.regs)):
      raise ValueError("regs must be two multiples of 8 in [24, 256]")
    if self.smem_bytes > SMEM_LIMIT: raise ValueError(f"needs {self.smem_bytes} bytes of shared memory, the limit is {SMEM_LIMIT}")

  @property
  def BM(self): return 64 * self.consumers
  @property
  def threads(self): return 128 * (self.consumers + 1)
  @property
  def nacc(self): return self.BN // 2                     # fp32 accumulators per thread for m64nBN
  @property
  def a_elems(self): return self.BM * self.BK
  @property
  def b_elems(self): return self.BN * self.BK
  @property
  def c_elems(self): return self.BM * self.BN if self.tma_store else 0
  @property
  def smem_bytes(self): return ((self.a_elems + self.b_elems) * self.stages + self.c_elems) * 2 + Ring.BYTES_PER_STAGE * self.stages
  @property
  def name(self): return (f"ws_c{self.consumers}_s{self.stages}_n{self.BN}_sm{self.num_sm}_g{self.group_m}x{self.group_n}"
                          f"{'_tmast' if self.tma_store else ''}{'_hil' if self.hilbert else ''}{'_w1' if self.inflight else ''}"
                          f"{f'_r{self.regs[0]}x{self.regs[1]}' if self.regs else ''}{f'_cl{self.cluster}' if self.cluster > 1 else ''}")


def build_gemm_ws(N: int, cfg: WSCfg, M: int | None = None, K: int | None = None):
  """C[M,N] = A[M,K] @ B[K,N] (square by default), bf16 in, fp32 accumulate, bf16 out.
  M % BM, N % BN and K % BK must be 0. Returns ``run(A, B, C) -> launch()``; ``launch()`` returns the realized C.
  ``run.kernel_name`` is the generated CUDA kernel's name (what Nsight Compute reports)."""
  from .runtime import install_runtime_hooks, set_dynamic_smem_bytes
  from .tensor_map import make_tensor_map

  M, K = N if M is None else M, N if K is None else K
  if min(M, N, K) <= 0: raise ValueError("M, N and K must be positive")
  BM, BN, BK, QS, CL = cfg.BM, cfg.BN, cfg.BK, cfg.stages, cfg.cluster
  if M % BM or N % BN or K % BK: raise ValueError(f"shape {M}x{N}x{K} not a multiple of tile {BM}x{BN}x{BK}")
  MT, NT, NK = M // BM, N // BN, K // BK
  if MT % CL: raise ValueError(f"{MT} row tiles not divisible by cluster {CL}")
  # schedule over cluster tiles: a cluster owns CL vertically adjacent tiles (same column -> same B tile, multicast)
  CMT = MT // CL
  TILES = CMT * NT
  NCL = min(cfg.num_sm // CL, TILES)                   # clusters in the grid
  GRID, TPC = NCL * CL, TILES // NCL
  if cfg.hilbert and TILES % NCL: raise ValueError(f"hilbert needs equal tiles per cluster ({TILES} tiles, {NCL} clusters)")
  tile_coords, grouped_2d = grouped_tile_order(CMT, NT, min(max(cfg.group_m // CL, 1), CMT), min(cfg.group_n, NT))
  # SMEM: [A stages][B stages][C staging][barriers], u16 offsets; C staging is 1024B aligned (128B swizzle)
  B_OFF = cfg.a_elems * QS
  C_OFF = B_OFF + cfg.b_elems * QS
  BAR_OFF = C_OFF + cfg.c_elems
  TX = (cfg.a_elems + cfg.b_elems) * 2                 # bytes per stage (each CTA receives all of A and B)
  shp = f"{N}" if M == N == K else f"{M}x{N}x{K}"
  kernel_name = f"hopper_{shp}_{'' if grouped_2d else 'g1d_'}{cfg.name}"
  install_runtime_hooks()
  set_dynamic_smem_bytes(kernel_name, cfg.smem_bytes)

  def kern(C: UOp, tA: UOp, tB: UOp, *extra: UOp) -> UOp:
    tC = extra[0] if cfg.tma_store else None
    sched = extra[-1] if cfg.hilbert else None
    g, t = UOp.special(GRID, "gidx0"), UOp.special(cfg.threads, "lidx0")
    wg, tid = t // 128, t % 128
    prod0, is_cons, ci = t.eq(0), wg >= 1, wg - 1
    cid, rank = g // CL, g % CL                        # 1D clusters: consecutive blockIdx.x, rank == %cluster_ctarank
    ntile = (TILES - 1 - cid) // NCL + 1               # cluster tiles cid, cid+NCL, ... < TILES

    def tile_rc(slot: UOp) -> tuple[UOp, UOp]:         # slot-th tile of this cluster -> (block row of this CTA, block col)
      if cfg.hilbert:
        e = sched.flatten()[cid * TPC + slot]
        cm, bn = e // 65536, e % 65536
      else: cm, bn = tile_coords(cid + slot * NCL)
      return cm * CL + rank, bn
    smem = UOp.placeholder((cfg.smem_bytes // 2,), dtypes.uint16, slot=0, addrspace=AddrSpace.LOCAL)
    acc = UOp.placeholder((cfg.nacc,), dtypes.float, slot=1, addrspace=AddrSpace.REG)
    ring = Ring(smem, BAR_OFF, QS, cfg.consumers, CL)
    sA = lambda s, dep: smem.after(dep)[s * cfg.a_elems]
    sB = lambda s, dep: smem.after(dep)[B_OFF + s * cfg.b_elems]
    tAf, tBf = tA.flatten()[0], tB.flatten()[0]

    tok = init_barriers(smem, [ring], t, CL)
    if cfg.regs is not None:
      tok = UOp.group(setmaxnreg(cfg.regs[0], False, smem.after(tok)[0], wg.eq(0)),
                      setmaxnreg(cfg.regs[1], True, smem.after(tok)[0], is_cons))
    tok = ring.prime(tok, tid, is_cons)
    # define the accumulator before the loops: otherwise ptxas serializes every wgmma behind a wait (C7515), ~2x slower
    z = UOp.range(cfg.nacc, 3)
    tok = acc.after(tok)[z].store(UOp.const(0.0, dtypes.float)).end(z)

    # ---- producer: thread 0, one flat loop over (tile, k), runs ahead across tile boundaries ----
    jp = UOp.range(prod0.where(ntile * NK, 0), 0, AxisType.LOOP)
    s, ph = ring.slot(jp)
    bm, bn = tile_rc(jp // NK); kk = jp % NK
    e = mbar_expect_tx(ring.full(s, mbar_wait(ring.empty(s, UOp.group(tok, jp)), ph, None)), TX, None)
    a = tma_load(sA(s, e), tAf, ring.full(s, e), bm * BM, kk * BK, None)
    if CL == 1: prod = tma_load(sB(s, a), tBf, ring.full(s, a), bn * BN, kk * BK, None).end(jp)
    else: prod = tma_load_multicast(sB(s, a), tBf, ring.full(s, a), bn * BN, kk * BK, (1 << CL) - 1, rank.eq(0)).end(jp)

    # ---- consumers: tile loop, K loop inside, epilogue after the K loop ----
    ti = UOp.range(is_cons.where(ntile, 0), 1, AxisType.LOOP)
    k = UOp.range(NK, 2, AxisType.LOOP)
    s, ph = ring.slot(ti * NK + k)
    wf = mbar_wait(ring.full(s, UOp.group(prod, ti, k)), ph, None)
    mm = UOp.wgmma(smem.after(wf)[s * cfg.a_elems + ci * (64 * BK)], sB(s, wf), acc.after(prod, ti, k, wf)[0], (64, BN, BK),
                   dtypes.bfloat16, scale_d=k.ne(0).cast(dtypes.int), wait=-1 if cfg.inflight else 0)
    if cfg.inflight:
      # wgmma k may still run; wgmma k-1 is done after wait_group 1, so its stage can be refilled now
      kl = ring.release((s + QS - 1) % QS, wgmma_wait(1, smem.after(mm)[0]), tid, k.ne(0)).end(k)
      kl = ring.release((ti * NK + NK - 1) % QS, wgmma_wait(0, smem.after(kl)[0]), tid)   # drain, free the tile's last stage
    else: kl = ring.release(s, mm, tid).end(k)
    bm, bn = tile_rc(ti)
    if cfg.tma_store:
      st = epilogue_tma_store(smem.after(kl)[C_OFF + ci * (64 * BN)], acc.after(kl)[0], tC.flatten()[0], bm * BM + ci * 64,
                              bn * BN, ci, tid, BN)
      body = tma_store_drain(smem.after(st.end(ti))[0], C.flatten()[0], is_cons & tid.eq(0))
    else:
      lane, warp = tid % 32, tid // 32
      st = store_acc_bf16(C.flatten()[0], acc.after(kl)[0], bm * BM + ci * 64 + warp * 16 + lane // 4,
                          bn * BN + 2 * (tid % 4), N, BN, None)
      body = st.end(ti)
    if CL > 1: body = cluster_sync(smem.after(body)[0])   # no CTA exits while a peer can still arrive / multicast into it
    return body.sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

  def run(A: Tensor, B: Tensor, Cout: Tensor):
    if A.shape != (M, K) or B.shape != (K, N) or Cout.shape != (M, N): raise ValueError("A, B and C shapes do not match the GEMM")
    if not (A.dtype == B.dtype == Cout.dtype == dtypes.bfloat16): raise ValueError("A, B and C must be bfloat16")
    BT = B.T.contiguous().realize()       # K-major B. Keep it alive: the tensor map holds its raw device pointer
    tA = make_tensor_map(A, BM, BK, M, K).realize()
    tB = make_tensor_map(BT, BN, BK, N, K).realize()
    extra = []
    if cfg.tma_store: extra.append(make_tensor_map(Cout, 64, BN, M, N).realize())
    if cfg.hilbert: extra.append(Tensor(hilbert_schedule(CMT, NT, NCL), dtype=dtypes.int32).realize())

    def launch(_keep=(A, BT)): return Tensor.custom_kernel(Cout, tA, tB, *extra, fxn=kern)[0].realize()
    return launch
  run.out_dtype = dtypes.bfloat16
  run.kernel_name = kernel_name
  return run
