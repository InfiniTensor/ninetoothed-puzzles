from collections.abc import Callable
from dataclasses import dataclass, field

ntl = None


class PuzzleDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is unavailable."""


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    observations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Puzzle:
    slug: str
    title: str
    chapter: str
    concepts: tuple[str, ...]
    prompt: str
    hint: str
    checker: Callable[[], CheckResult] = field(repr=False)
    requires_cuda: bool = False

    def run(self) -> CheckResult:
        return self.checker()


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise PuzzleDependencyError("install numpy to run this level") from exc
    return np


def _nt():
    global ntl

    try:
        import ninetoothed
        import ninetoothed.language as ntl_module
        from ninetoothed import Symbol, Tensor
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise PuzzleDependencyError(
            "install ninetoothed, or set PYTHONPATH to a NineToothed checkout"
        ) from exc
    ntl = ntl_module
    return ninetoothed, ntl_module, Symbol, Tensor


def _torch_cuda():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise PuzzleDependencyError("install torch to run CUDA levels") from exc

    if not torch.cuda.is_available():
        raise PuzzleDependencyError("CUDA is not available")
    return torch


def _check_symbolic_map() -> CheckResult:
    np = _np()
    _, _, _, Tensor = _nt()

    x = Tensor(1)
    observed = x.eval({x: Tensor(shape=(5,))})
    expected = np.arange(5)

    if not np.array_equal(observed, expected):
        raise AssertionError(f"expected {expected}, got {observed}")

    return CheckResult(
        True,
        ("Tensor(1) with shape (5,) evaluates to source indices [0, 1, 2, 3, 4].",),
    )


def _check_tile_door() -> CheckResult:
    np = _np()
    _, _, Symbol, Tensor = _nt()

    block_size = Symbol("block_size", constexpr=True)
    x = Tensor(1)
    tiled = x.tile((block_size,))
    observed = tiled.eval({x: Tensor(shape=(10,)), block_size: 4})
    expected = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, -1, -1]])

    if not np.array_equal(observed, expected):
        raise AssertionError(f"expected {expected}, got {observed}")

    return CheckResult(
        True,
        ("A length-10 vector tiled by 4 becomes 3 program blocks with padding.",),
    )


def _check_broadcast_bridge() -> CheckResult:
    np = _np()
    _, _, _, Tensor = _nt()

    x = Tensor(1)
    expanded = x.unsqueeze(0).expand((3, -1))
    observed = expanded.eval({x: Tensor(shape=(4,))})
    expected = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]
    )

    if not np.array_equal(observed, expected):
        raise AssertionError(f"expected {expected}, got {observed}")

    return CheckResult(
        True,
        ("unsqueeze creates a new axis; expand repeats the symbolic index view.",),
    )


def _check_slice_key() -> CheckResult:
    np = _np()
    _, _, _, Tensor = _nt()

    x = Tensor(shape=(3, 4, 2))
    observed = x[:, 1, :].eval()
    expected = np.arange(24).reshape(3, 4, 2)[:, 1, :]

    if not np.array_equal(observed, expected):
        raise AssertionError(f"expected {expected}, got {observed}")

    return CheckResult(True, ("NineToothed slicing follows familiar tensor indexing.",))


def _check_program_grid() -> CheckResult:
    np = _np()
    _, _, Symbol, Tensor = _nt()

    block_size = Symbol("block_size", constexpr=True)
    x = Tensor(1)
    y = Tensor(1)
    z = Tensor(1)

    arranged = (
        x.tile((block_size,)),
        y.tile((block_size,)),
        z.tile((block_size,)),
    )
    subs = {
        x: Tensor(shape=(8,)),
        y: Tensor(shape=(8,)),
        z: Tensor(shape=(8,)),
        block_size: 4,
    }
    evaluated = tuple(tensor.eval(subs) for tensor in arranged)
    expected_first_program = np.array([0, 1, 2, 3])
    expected_second_program = np.array([4, 5, 6, 7])

    if evaluated[0].shape != (2, 4):
        raise AssertionError(f"expected outer program grid (2, 4), got {evaluated[0].shape}")
    if not np.array_equal(evaluated[0][0], expected_first_program):
        raise AssertionError("program 0 did not receive the first block")
    if not np.array_equal(evaluated[0][1], expected_second_program):
        raise AssertionError("program 1 did not receive the second block")

    return CheckResult(
        True,
        (
            "The arranged vector has 2 outer elements, so the kernel launches 2 program instances.",
            "Each program instance sees a local block of 4 values.",
        ),
    )


def _make_matmul_arrangement():
    _, _, Symbol, Tensor = _nt()

    block_size_m = Symbol("block_size_m", constexpr=True)
    block_size_n = Symbol("block_size_n", constexpr=True)
    block_size_k = Symbol("block_size_k", constexpr=True)

    def arrangement(lhs, rhs, output):
        output_tiled = output.tile((block_size_m, block_size_n))

        lhs_tiled = (
            lhs.tile((block_size_m, block_size_k))
            .tile((1, -1))
            .expand((-1, output_tiled.shape[1]))
        )
        lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)

        rhs_tiled = (
            rhs.tile((block_size_k, block_size_n))
            .tile((-1, 1))
            .expand((output_tiled.shape[0], -1))
        )
        rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

        return lhs_tiled, rhs_tiled, output_tiled

    return arrangement, (block_size_m, block_size_n, block_size_k), Tensor


def _check_matmul_map() -> CheckResult:
    arrangement, symbols, Tensor = _make_matmul_arrangement()
    block_size_m, block_size_n, block_size_k = symbols

    lhs = Tensor(2)
    rhs = Tensor(2)
    output = Tensor(2)
    lhs_arranged, rhs_arranged, output_arranged = arrangement(lhs, rhs, output)

    subs = {
        lhs: Tensor(shape=(4, 6)),
        rhs: Tensor(shape=(6, 8)),
        output: Tensor(shape=(4, 8)),
        block_size_m: 2,
        block_size_n: 4,
        block_size_k: 3,
    }
    lhs_concrete = lhs_arranged.subs(subs)
    rhs_concrete = rhs_arranged.subs(subs)
    output_concrete = output_arranged.subs(subs)

    if lhs_concrete.shape != output_concrete.shape:
        raise AssertionError("lhs outer shape must match output outer shape")
    if rhs_concrete.shape != output_concrete.shape:
        raise AssertionError("rhs outer shape must match output outer shape")
    if output_concrete.shape != (2, 2):
        raise AssertionError(f"expected a 2x2 output block grid, got {output_concrete.shape}")
    if lhs_concrete.dtype.shape[0] != rhs_concrete.dtype.shape[0]:
        raise AssertionError("lhs and rhs must expose the same K-block count")

    return CheckResult(
        True,
        (
            "The output matrix maps to a 2x2 program grid.",
            "Each output block receives one row of lhs blocks and one column of rhs blocks.",
        ),
    )


def _check_vector_add_kernel() -> CheckResult:
    torch = _torch_cuda()
    ninetoothed, _, Symbol, Tensor = _nt()

    block_size = Symbol("block_size", constexpr=True)

    @ninetoothed.jit
    def add_kernel(
        lhs: Tensor(1).tile((block_size,)),
        rhs: Tensor(1).tile((block_size,)),
        output: Tensor(1).tile((block_size,)),
    ):
        output = lhs + rhs  # noqa: F841

    lhs = torch.randn(257, device="cuda")
    rhs = torch.randn(257, device="cuda")
    output = torch.empty_like(lhs)

    add_kernel(lhs, rhs, output, block_size=64)
    expected = lhs + rhs

    if not torch.allclose(output, expected):
        raise AssertionError("vector add kernel did not match torch")

    return CheckResult(True, ("A JIT kernel can tile three vectors and write one blockwise result.",))


def _check_row_sum_kernel() -> CheckResult:
    torch = _torch_cuda()
    ninetoothed, _, Symbol, Tensor = _nt()

    block_size = Symbol("block_size", constexpr=True)

    def arrangement(x, y, block_size=block_size):
        x_arranged = x.tile((1, block_size))
        x_arranged = x_arranged.tile((1, -1))
        y_arranged = y.tile((1, 1))
        return x_arranged, y_arranged

    def application(x, y):
        acc = ntl.zeros(y.shape, dtype=y.dtype)
        for i in range(x.shape[1]):
            acc += ntl.sum(x[0, i], axis=-1)
        y = acc  # noqa: F841

    kernel = ninetoothed.make(arrangement, application, (Tensor(2, other=0), Tensor(2)))

    x = torch.randn((9, 31), device="cuda")
    y = torch.empty((9, 1), device="cuda")
    kernel(x, y, block_size=8)
    expected = torch.sum(x, dim=-1, keepdim=True)

    if not torch.allclose(y, expected, atol=1e-5):
        raise AssertionError("row-sum kernel did not match torch")

    return CheckResult(
        True,
        ("The application loop iterates over K-blocks inside each row program.",),
    )


def _check_softmax_kernel() -> CheckResult:
    torch = _torch_cuda()
    ninetoothed, _, Symbol, Tensor = _nt()

    block_size = Symbol("block_size", constexpr=True)

    @ninetoothed.jit
    def softmax_kernel(
        input_row: Tensor(2, other=float("-inf")).tile((1, block_size)),
        output_row: Tensor(2).tile((1, block_size)),
    ):
        row_minus_max = input_row - ntl.max(input_row)
        numerator = ntl.exp(row_minus_max)
        denominator = ntl.sum(numerator)
        output_row = numerator / denominator  # noqa: F841

    x = torch.randn((7, 23), device="cuda")
    output = torch.empty_like(x)
    softmax_kernel(x, output, block_size=x.shape[-1])
    expected = torch.softmax(x, dim=-1)

    if not torch.allclose(output, expected, atol=1e-5):
        raise AssertionError("softmax kernel did not match torch")

    return CheckResult(
        True,
        ("Reductions from ninetoothed.language can build a numerically stable row softmax.",),
    )


PUZZLES: tuple[Puzzle, ...] = (
    Puzzle(
        slug="symbolic-map",
        title="Map the symbolic tensor",
        chapter="Orientation",
        concepts=("Tensor", "eval", "subs"),
        prompt=(
            "Create a one-dimensional symbolic Tensor, substitute shape (5,), "
            "and predict the source-index array produced by eval()."
        ),
        hint="A concrete one-dimensional tensor evaluates to its source indices.",
        checker=_check_symbolic_map,
    ),
    Puzzle(
        slug="tile-door",
        title="Open the tile door",
        chapter="Meta-operations",
        concepts=("tile", "nested tensors", "padding"),
        prompt=(
            "Tile a length-10 vector with block size 4. The last block should "
            "show how NineToothed represents out-of-bounds elements."
        ),
        hint="The default eval view uses -1 for padded positions.",
        checker=_check_tile_door,
    ),
    Puzzle(
        slug="broadcast-bridge",
        title="Build the broadcast bridge",
        chapter="Meta-operations",
        concepts=("unsqueeze", "expand"),
        prompt=(
            "Turn a length-4 vector into three symbolic rows by adding a new "
            "axis and expanding it."
        ),
        hint="Use unsqueeze(0) before expand((3, -1)).",
        checker=_check_broadcast_bridge,
    ),
    Puzzle(
        slug="slice-key",
        title="Find the slice key",
        chapter="Meta-operations",
        concepts=("getitem", "slicing", "eval"),
        prompt=(
            "Slice a concrete symbolic tensor with shape (3, 4, 2) and compare "
            "the evaluated indices with NumPy."
        ),
        hint="Regular tensor indexing is part of the arrangement language.",
        checker=_check_slice_key,
    ),
    Puzzle(
        slug="program-grid",
        title="Read the program grid",
        chapter="Arrange and apply",
        concepts=("arrangement", "program instances", "tile"),
        prompt=(
            "Arrange three vectors with the same tile shape and identify which "
            "local block each program instance receives."
        ),
        hint="Only the outermost arranged tensor shape launches programs.",
        checker=_check_program_grid,
    ),
    Puzzle(
        slug="matmul-map",
        title="Scout the matmul map",
        chapter="Arrange and apply",
        concepts=("tile", "expand", "squeeze", "matmul"),
        prompt=(
            "Build the non-CUDA arrangement for matrix multiplication and prove "
            "that lhs, rhs, and output share the same outer block grid."
        ),
        hint="Rows of lhs and columns of rhs must expand to the output grid.",
        checker=_check_matmul_map,
    ),
    Puzzle(
        slug="vector-add-kernel",
        title="Cast vector add",
        chapter="CUDA kernels",
        concepts=("jit", "Tensor annotations", "blockwise write"),
        prompt="Write the first CUDA kernel: z = x + y for tiled vectors.",
        hint="Annotate each parameter as Tensor(1).tile((block_size,)).",
        checker=_check_vector_add_kernel,
        requires_cuda=True,
    ),
    Puzzle(
        slug="row-sum-kernel",
        title="Harvest row sums",
        chapter="CUDA kernels",
        concepts=("make", "application loop", "ntl.sum"),
        prompt=(
            "Tile each row into smaller blocks, loop over those blocks in the "
            "application function, and sum a row into one output value."
        ),
        hint="Use Tensor(2, other=0) so padded row elements do not change the sum.",
        checker=_check_row_sum_kernel,
        requires_cuda=True,
    ),
    Puzzle(
        slug="softmax-kernel",
        title="Stabilize softmax",
        chapter="CUDA kernels",
        concepts=("ntl.max", "ntl.exp", "ntl.sum", "softmax"),
        prompt="Implement row-wise softmax using reductions from ninetoothed.language.",
        hint="Subtract the row max before exponentiating.",
        checker=_check_softmax_kernel,
        requires_cuda=True,
    ),
)

PUZZLES_BY_SLUG = {puzzle.slug: puzzle for puzzle in PUZZLES}
