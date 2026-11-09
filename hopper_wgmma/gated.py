"""The first two ladder steps: a TMA pipeline, then warp specialization with role-GATED effects in one uniform loop.

Every instruction is predicated on the thread's role and a single K loop carries both the producer's refill and the
consumers' wgmma. It works but is ~10% slower than the role-split loops in ``gemm.build_gemm_ws``, because the consumers'
loop still carries all of the producer's predicated instructions. Kept to reproduce the first rows of the ladder."""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

from .gemm import (SMEM_LIMIT, fence_mbar_init, mbar_arrive, mbar_expect_tx, mbar_init, mbar_wait, store_acc_bf16,
                   tma_load)


@dataclass(frozen=True)
class GatedCfg:
  """consumers=0: one warpgroup does both roles (plain TMA pipeline). consumers=1|2: one producer warpgroup plus that
  many wgmma warpgroups (warp specialization, gated)."""
  BN: int = 256
  BK: int = 64
  stages: int = 3
  consumers: int = 2

  def __post_init__(self):
    if self.BK != 64: raise ValueError("BK must be 64 (one 128-byte swizzle atom)")
    if self.BN <= 0 or self.BN % 16 or self.BN > 256: raise ValueError("BN must be a multiple of 16 in (0, 256]")
    if self.stages < 2: raise ValueError("stages must be at least 2")
    if self.consumers not in (0, 1, 2): raise ValueError("consumers must be 0, 1 or 2")
    if self.smem_bytes > SMEM_LIMIT: raise ValueError(f"needs {self.smem_bytes} bytes of shared memory, the limit is {SMEM_LIMIT}")

  @property
  def ncons(self): return max(self.consumers, 1)
  @property
  def BM(self): return 64 * self.ncons
  @property
  def threads(self): return 128 * (self.ncons + (1 if self.consumers else 0))
  @property
  def nacc(self): return self.BN // 2
  @property
  def a_elems(self): return self.BM * self.BK
  @property
  def b_elems(self): return self.BN * self.BK
  @property
  def smem_bytes(self): return (self.a_elems + self.b_elems) * 2 * self.stages + 16 * self.stages


def build_gated_gemm(N: int, cfg: GatedCfg):
  """C[N,N] (bf16) = A[N,N] @ B[N,N], one CTA per output tile. Same contract as ``build_gemm_ws``."""
  from .runtime import install_runtime_hooks, set_dynamic_smem_bytes
  from .tensor_map import make_tensor_map

  BM, BN, BK, QS = cfg.BM, cfg.BN, cfg.BK, cfg.stages
  if N <= 0 or N % BM or N % BN or N % BK: raise ValueError(f"N={N} must be a multiple of the tile {BM}x{BN}x{BK}")
  MT, NT, NK = N // BM, N // BN, N // BK
  if NK < QS: raise ValueError(f"K has {NK} blocks, fewer than {QS} stages")
  TILES = MT * NT
  B_OFF = cfg.a_elems * QS
  BAR_OFF = (cfg.a_elems + cfg.b_elems) * QS
  TX = (cfg.a_elems + cfg.b_elems) * 2
  kernel_name = f"hopper_gated_{N}_c{cfg.consumers}_s{QS}_n{BN}"
  install_runtime_hooks()
  set_dynamic_smem_bytes(kernel_name, cfg.smem_bytes)

  def kern(C: UOp, tA: UOp, tB: UOp) -> UOp:
    g = UOp.special(TILES, "gidx0"); t = UOp.special(cfg.threads, "lidx0")
    wg, tid = t // 128, t % 128
    cons0 = 1 if cfg.consumers else 0
    is_cons = (wg >= cons0) if cons0 else wg.eq(0)
    prod0 = (wg.eq(0) & tid.eq(0))
    ci = wg - cons0
    smem = UOp.placeholder((cfg.smem_bytes // 2,), dtypes.uint16, slot=0, addrspace=AddrSpace.LOCAL)
    acc = UOp.placeholder((cfg.nacc,), dtypes.float, slot=1, addrspace=AddrSpace.REG)
    tAf, tBf = tA.flatten()[0], tB.flatten()[0]
    full = lambda s, dep: smem.after(dep)[BAR_OFF + s * 4]
    empty = lambda s, dep: smem.after(dep)[BAR_OFF + (QS + s) * 4]
    sA = lambda s, dep: smem.after(dep)[s * cfg.a_elems]
    sB = lambda s, dep: smem.after(dep)[B_OFF + s * cfg.b_elems]
    bm, bn = g // NT, g % NT

    tok = UOp.group(*[mbar_init(full(s, smem), 1, t.eq(0)) for s in range(QS)],
                    *[mbar_init(empty(s, smem), cfg.ncons, t.eq(0)) for s in range(QS)])
    tok = UOp.barrier(fence_mbar_init(smem.after(tok)[0], t.eq(0)))
    # define the accumulator before the loop, otherwise ptxas serializes every mma_async with a wait (C7515)
    z = UOp.range(cfg.nacc, 1)
    tok = acc.after(tok)[z].store(UOp.const(0.0, dtypes.float)).end(z)

    def load_iter(s, j: UOp, dep: UOp, gate: UOp) -> UOp:
      kk = j % NK
      e = mbar_expect_tx(full(s, dep), TX, gate)
      a = tma_load(sA(s, e), tAf, full(s, e), bm * BM, kk * BK, gate)
      return tma_load(sB(s, a), tBf, full(s, a), bn * BN, kk * BK, gate)

    tok = UOp.group(*[load_iter(s, UOp.const(s, dtypes.int), smem.after(tok)[0], prod0) for s in range(QS)])

    j = UOp.range(NK, 0, AxisType.LOOP)
    s, ph = j % QS, (j // QS) % 2
    wf = mbar_wait(full(s, UOp.group(tok, j)), ph, is_cons)
    mm = UOp.wgmma(smem.after(wf)[s * cfg.a_elems + ci * (64 * BK)], sB(s, wf), acc.after(tok, j, wf)[0], (64, BN, BK),
                   dtypes.bfloat16, scale_d=j.ne(0).cast(dtypes.int), gate=is_cons)
    rel = mbar_arrive(empty(s, mm), is_cons & tid.eq(0))
    lane, warp = tid % 32, tid // 32
    st = store_acc_bf16(C.flatten()[0], acc.after(mm)[0], bm * BM + ci * 64 + warp * 16 + lane // 4,
                        bn * BN + 2 * (tid % 4), N, BN, is_cons & j.eq(NK - 1))
    refill = prod0 & (j + QS < NK)
    we = mbar_wait(empty(s, rel), ph, refill)
    ld = load_iter(s, j + QS, smem.after(we)[0], refill)
    return UOp.group(st, rel, ld).end(j).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

  def run(A: Tensor, B: Tensor, Cout: Tensor):
    if A.shape != (N, N) or B.shape != (N, N) or Cout.shape != (N, N): raise ValueError("A, B and C must be N x N")
    if not (A.dtype == B.dtype == Cout.dtype == dtypes.bfloat16): raise ValueError("A, B and C must be bfloat16")
    BT = B.T.contiguous().realize()   # keep alive: the tensor maps hold raw device pointers
    tA = make_tensor_map(A, BM, BK, N, N).realize()
    tB = make_tensor_map(BT, BN, BK, N, N).realize()

    def launch(_keep=(A, BT)): return Tensor.custom_kernel(Cout, tA, tB, fxn=kern)[0].realize()
    return launch
  run.out_dtype = dtypes.bfloat16
  run.kernel_name = kernel_name
  return run
