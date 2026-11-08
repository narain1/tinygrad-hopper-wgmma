"""A small warp-specialized bf16 GEMM for Hopper WGMMA experiments."""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import UOp, dtypes
from tinygrad.uop.ops import Ops

def _custom(fmt: str, *srcs: UOp, gate: UOp | None = None) -> UOp:
  """One CUSTOM effect. Braces in PTX are doubled because tinygrad formats source operands into ``fmt``."""
  fmt, srcs = ("if({%d}) {{ " % len(srcs) + fmt + " }}", srcs + (gate,)) if gate is not None else ("{{ " + fmt + " }}", srcs)
  return UOp(Ops.CUSTOM, tuple(srcs), arg=(fmt, dtypes.void))


def _shared(index: int) -> str:
  return "(unsigned)__cvta_generic_to_shared({%d})" % index


def mbar_init(barrier: UOp, count: int, gate: UOp | None) -> UOp:
  return _custom('asm volatile("mbarrier.init.shared::cta.b64 [%%0], %d;" :: "r"(%s) : "memory");' % (count, _shared(0)), barrier, gate=gate)


def fence_mbar_init(dep: UOp, gate: UOp | None) -> UOp:
  return _custom('asm volatile("fence.mbarrier_init.release.cluster; fence.proxy.async.shared::cta;" ::: "memory"); (void){0};', dep, gate=gate)


_label = [0]
def mbar_wait(barrier: UOp, phase: UOp, gate: UOp | None) -> UOp:
  _label[0] += 1
  label = f"TGW{_label[0]}"
  return _custom('asm volatile("{{ .reg .pred P; %s: mbarrier.try_wait.parity.shared::cta.b64 P, [%%0], %%1; @!P bra %s; }}" '
                 ':: "r"(%s), "r"((unsigned)({1})) : "memory");' % (label, label, _shared(0)), barrier, phase, gate=gate)


def mbar_expect_tx(barrier: UOp, nbytes: int, gate: UOp | None) -> UOp:
  return _custom('asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%%0], %d;" :: "r"(%s) : "memory");' % (nbytes, _shared(0)), barrier, gate=gate)


def mbar_arrive(barrier: UOp, gate: UOp | None) -> UOp:
  return _custom('asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%%0];" :: "r"(%s) : "memory");' % _shared(0), barrier, gate=gate)


def tma_load(dst: UOp, tmap: UOp, barrier: UOp, row: UOp, col: UOp, gate: UOp | None) -> UOp:
  return _custom('asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes'
                 ' [%%0], [%%1, {{0, %%3, %%4}}], [%%2];" :: "r"(%s), "l"((unsigned long)({1})), "r"(%s), '
                 '"r"((int)({3})), "r"((int)({4})/64) : "memory");' % (_shared(0), _shared(2)), dst, tmap, barrier, row, col, gate=gate)


def store_bf16(output: UOp, accumulator: UOp, row: UOp, col: UOp, ld: int, bn: int) -> UOp:
  return _custom('float* d={1}; __nv_bfloat162* c=(__nv_bfloat162*)({0}); long r=(long)({2}), cc=(long)({3});'
                 ' _Pragma("unroll") for(int w=0; w<%d; w++){{ long o=(r*%d+cc+w*16)/2;'
                 ' c[o]=__floats2bfloat162_rn(d[w*8+0],d[w*8+1]); c[o+%d]=__floats2bfloat162_rn(d[w*8+2],d[w*8+3]);'
                 ' c[o+4]=__floats2bfloat162_rn(d[w*8+4],d[w*8+5]); c[o+%d+4]=__floats2bfloat162_rn(d[w*8+6],d[w*8+7]); }}'
                 % (bn // 16, ld, 4 * ld, 4 * ld), output, accumulator, row, col)


class Ring:
  """Staged FULL/EMPTY mbarrier queue shared by one TMA producer and consumers."""
  BYTES_PER_STAGE = 16

  def __init__(self, smem, barrier_offset, stages, consumers):
    self.smem, self.barrier_offset, self.stages, self.consumers = smem, barrier_offset, stages, consumers

  def full(self, stage, dep): return self.smem.after(dep)[self.barrier_offset + stage * 4]
  def empty(self, stage, dep): return self.smem.after(dep)[self.barrier_offset + (self.stages + stage) * 4]
  def slot(self, iteration): return iteration % self.stages, (iteration // self.stages) % 2
  def initialize(self, thread):
    effects = [mbar_init(self.full(s, self.smem), 1, thread.eq(0)) for s in range(self.stages)]
    effects += [mbar_init(self.empty(s, self.smem), self.consumers, thread.eq(0)) for s in range(self.stages)]
    return UOp.group(*effects)
  def release(self, stage, dep, tid, consumers): return mbar_arrive(self.empty(stage, dep), consumers & tid.eq(0))


@dataclass(frozen=True)
class WSCfg:
  BN: int = 256
  BK: int = 64
  stages: int = 3
  consumers: int = 2
  num_sm: int = 128
  group_m: int = 16
  group_n: int = 8
  out_bf16: bool = True

  def __post_init__(self):
    if self.BN <= 0 or self.BN % 16: raise ValueError("BN must be a positive multiple of 16")
    if self.BK <= 0 or self.BK % 64: raise ValueError("BK must be a positive multiple of 64")
    if self.stages < 2: raise ValueError("stages must be at least 2")
    if self.consumers < 1: raise ValueError("consumers must be positive")
    if self.num_sm < 1: raise ValueError("num_sm must be positive")
    if self.group_m < 1 or self.group_n < 1: raise ValueError("group sizes must be positive")
    if not self.out_bf16: raise ValueError("this compact experiment supports bf16 output only")

  @property
  def BM(self): return 64 * self.consumers
  @property
  def threads(self): return 128 * (self.consumers + 1)
  @property
  def smem_bytes(self): return ((self.BM * self.BK + self.BN * self.BK) * self.stages) * 2 + Ring.BYTES_PER_STAGE * self.stages
  @property
  def name(self): return f"ws_c{self.consumers}_s{self.stages}_n{self.BN}_sm{self.num_sm}"


def build_gemm_ws(N: int, cfg: WSCfg, M: int | None = None, K: int | None = None):
  """Build C[M,N] = A[M,K] @ B[K,N]. Inputs and output are bf16 CUDA tensors."""
  if N <= 0: raise ValueError("N must be positive")
  M, K = N if M is None else M, N if K is None else K
  if M <= 0 or K <= 0: raise ValueError("M and K must be positive")
  if M % cfg.BM or N % cfg.BN or K % cfg.BK:
    raise ValueError(f"shape {M}x{N}x{K} must be divisible by tile {cfg.BM}x{cfg.BN}x{cfg.BK}")
  from tinygrad import Tensor, UOp, dtypes
  from tinygrad.dtype import AddrSpace
  from tinygrad.uop.ops import AxisType, KernelInfo
  from .runtime import install_runtime_hooks, set_dynamic_smem_bytes
  from .tensor_map import make_tensor_map

  kernel_name = f"hopper_{M}x{N}x{K}_{cfg.name}"
  install_runtime_hooks()
  set_dynamic_smem_bytes(kernel_name, cfg.smem_bytes)
  mt, nt, nk = M // cfg.BM, N // cfg.BN, K // cfg.BK
  tiles, grid = mt * nt, min(cfg.num_sm, mt * nt)

  def kernel(C, map_a, map_b):
    group, thread = UOp.special(grid, "gidx0"), UOp.special(cfg.threads, "lidx0")
    warp_group, tid = thread // 128, thread % 128
    producer, consumer, consumer_index = thread.eq(0), warp_group >= 1, warp_group - 1
    smem = UOp.placeholder((cfg.smem_bytes // 2,), dtypes.uint16, slot=0, addrspace=AddrSpace.LOCAL)
    accumulators = UOp.placeholder((cfg.BN // 2,), dtypes.float, slot=1, addrspace=AddrSpace.REG)
    ring = Ring(smem, (cfg.BM * cfg.BK + cfg.BN * cfg.BK) * cfg.stages, cfg.stages, cfg.consumers)
    initialized = ring.initialize(thread)
    init = UOp.barrier(fence_mbar_init(smem.after(initialized)[0], thread.eq(0)))
    init = UOp.group(*[ring.release(s, init, tid, consumer) for s in range(cfg.stages)])
    zero = UOp.range(cfg.BN // 2, 3); init = accumulators.after(init)[zero].store(UOp.const(0.0, dtypes.float)).end(zero)
    per_group = (tiles - 1 - group) // grid + 1
    producer_loop = UOp.range(producer.where(per_group * nk, 0), 0, AxisType.LOOP)
    stage, phase = ring.slot(producer_loop); tile, kk = producer_loop // nk, producer_loop % nk
    row, col = (group + tile * grid) // nt, (group + tile * grid) % nt
    ready = mbar_expect_tx(ring.full(stage, mbar_wait(ring.empty(stage, UOp.group(init, producer_loop)), phase, None)), (cfg.BM * cfg.BK + cfg.BN * cfg.BK) * 2, None)
    a = tma_load(smem.after(ready)[stage * cfg.BM * cfg.BK], map_a.flatten()[0], ring.full(stage, ready), row * cfg.BM, kk * cfg.BK, None)
    produced = tma_load(smem.after(a)[cfg.BM * cfg.BK * cfg.stages + stage * cfg.BN * cfg.BK], map_b.flatten()[0], ring.full(stage, a), col * cfg.BN, kk * cfg.BK, None).end(producer_loop)
    tile_loop = UOp.range(consumer.where(per_group, 0), 1, AxisType.LOOP); k_loop = UOp.range(nk, 2, AxisType.LOOP)
    stage, phase = ring.slot(tile_loop * nk + k_loop)
    ready = mbar_wait(ring.full(stage, UOp.group(produced, tile_loop, k_loop)), phase, None)
    mma = UOp.wgmma(smem.after(ready)[stage * cfg.BM * cfg.BK + consumer_index * 64 * cfg.BK], smem.after(ready)[cfg.BM * cfg.BK * cfg.stages + stage * cfg.BN * cfg.BK], accumulators.after(produced, tile_loop, k_loop, ready)[0], (64, cfg.BN, cfg.BK), dtypes.bfloat16, scale_d=k_loop.ne(0).cast(dtypes.int))
    released = ring.release(stage, mma, tid, consumer).end(k_loop)
    row, col = (group + tile_loop * grid) // nt, (group + tile_loop * grid) % nt
    lane, warp = tid % 32, tid // 32
    stored = store_bf16(C.flatten()[0], accumulators.after(released)[0], row * cfg.BM + consumer_index * 64 + warp * 16 + lane // 4, col * cfg.BN + 2 * (tid % 4), N, cfg.BN)
    return stored.end(tile_loop).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

  def run(A: Tensor, B: Tensor, C: Tensor):
    if A.shape != (M, K) or B.shape != (K, N) or C.shape != (M, N): raise ValueError("A, B, and C shapes do not match GEMM")
    if A.dtype != dtypes.bfloat16 or B.dtype != dtypes.bfloat16 or C.dtype != dtypes.bfloat16: raise ValueError("A, B, and C must be bfloat16")
    transposed_b = B.T.contiguous().realize()
    map_a, map_b = make_tensor_map(A, cfg.BM, cfg.BK, M, K).realize(), make_tensor_map(transposed_b, cfg.BN, cfg.BK, N, K).realize()
    def launch(_keep=(A, transposed_b, map_a, map_b)): return Tensor.custom_kernel(C, map_a, map_b, fxn=kernel)[0].realize()
    return launch
  run.out_dtype = dtypes.bfloat16
  return run
