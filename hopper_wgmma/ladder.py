"""The optimization ladder as named configs, in the order the blog adds them."""
from __future__ import annotations

from .gated import GatedCfg, build_gated_gemm
from .gemm import WSCfg, build_gemm_ws

LADDER = {
  "tma_ring": GatedCfg(BN=128, stages=4, consumers=0),            # TMA + 4-stage pipeline, one warpgroup
  "gated_ws": GatedCfg(consumers=2),                              # warp specialization, role-gated instructions
  "role_split": WSCfg(),                                          # role-split loops + persistent CTAs + 16x8 tile order
  "role_split_regs": WSCfg(regs=(24, 240)),                       # role_split + setmaxnreg (experimental, expected no effect)
  "tma_store": WSCfg(tma_store=True),                             # + stmatrix/TMA-store epilogue
  "hilbert": WSCfg(tma_store=True, hilbert=True),                 # + Hilbert tile order
  "cluster": WSCfg(tma_store=True, cluster=2),                    # clusters of 2 with multicast B (instead of Hilbert)
  "cluster_hilbert": WSCfg(tma_store=True, hilbert=True, cluster=2),
}


def build(N: int, cfg, M: int | None = None, K: int | None = None):
  """Build the right GEMM for ``cfg``. Gated configs are square only."""
  if isinstance(cfg, GatedCfg):
    if (M, K) not in ((None, None), (N, N)): raise ValueError("the gated builder is square only")
    return build_gated_gemm(N, cfg)
  return build_gemm_ws(N, cfg, M, K)

# Extra tile shapes tried per problem shape by the sweep (BN=192 lets e.g. 3072 divide evenly over the CTAs).
SWEEP = {
  "tma_store": LADDER["tma_store"],
  "hilbert": LADDER["hilbert"],
  "tma_store_n192": WSCfg(BN=192, tma_store=True),
  "hilbert_n192": WSCfg(BN=192, tma_store=True, hilbert=True),
}

# One-off comparisons quoted in the blog post.
EXTRA = {
  "tma_store_132": WSCfg(tma_store=True, num_sm=132),   # all SMs instead of 128: 512 tiles / 132 CTAs leaves a partial wave
}
