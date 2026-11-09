"""The hand-written kernel from the blog post: one thread per element, out = a * 2. Runs it and checks the result."""
import _path  # noqa: F401
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import KernelInfo


def times_two(out: UOp, a: UOp) -> UOp:
  i = UOp.special(1024, "gidx0") * 256 + UOp.special(256, "lidx0")   # blockIdx.x * 256 + threadIdx.x
  return out.flatten()[i].store(a.flatten()[i] * 2).sink(arg=KernelInfo(name="times_two", opts_to_apply=()))


a = Tensor.rand(1024 * 256)
out = Tensor.custom_kernel(Tensor.empty(1024 * 256), a, fxn=times_two)[0]   # outputs go first; every argument is returned
assert (out.numpy() == a.numpy() * 2).all()
print("times_two OK")
