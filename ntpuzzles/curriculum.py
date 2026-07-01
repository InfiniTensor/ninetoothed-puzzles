from __future__ import annotations

from dataclasses import dataclass, field


Text = dict[str, str]


@dataclass(frozen=True)
class LessonBlock:
    title: Text
    body: Text
    code: str = ""


@dataclass(frozen=True)
class Quest:
    slug: str
    module: str
    kind: str
    title: Text
    subtitle: Text
    concepts: tuple[str, ...]
    lesson: tuple[LessonBlock, ...]
    question: Text
    expected: object = None
    accepted: tuple[object, ...] = ()
    answer_note: Text = field(default_factory=dict)
    hint: Text = field(default_factory=dict)
    success: Text = field(default_factory=dict)
    requires_cuda: bool = False
    starter_name: str = ""
    starter: str = ""
    validator: str = ""
    required_fragments: tuple[str, ...] = ()


@dataclass(frozen=True)
class Module:
    slug: str
    title: Text
    summary: Text


def text(value: Text, lang: str) -> str:
    return value.get(lang) or value.get("en") or next(iter(value.values()))


def _vector_add_starter() -> str:
    return """\
import ninetoothed
from ninetoothed import Symbol, Tensor

block_size = Symbol("block_size", constexpr=True)


@ninetoothed.jit
def kernel(
    lhs: Tensor(1).tile((block_size,)),
    rhs: Tensor(1).tile((block_size,)),
    output: Tensor(1).tile((block_size,)),
):
    # TODO: write lhs + rhs into output.
    output = lhs
"""


def _row_sum_starter() -> str:
    return """\
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

block_size = Symbol("block_size", constexpr=True)


def arrangement(x, y, block_size=block_size):
    x_arranged = x.tile((1, block_size))
    x_arranged = x_arranged.tile((1, -1))
    y_arranged = y.tile((1, 1))
    return x_arranged, y_arranged


def application(x, y):
    acc = ntl.zeros(y.shape, dtype=y.dtype)
    for i in range(x.shape[1]):
        # TODO: add the sum of the i-th local block.
        acc += 0
    y = acc


def build_kernel():
    return ninetoothed.make(arrangement, application, (Tensor(2, other=0), Tensor(2)))
"""


def _softmax_starter() -> str:
    return """\
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

block_size = Symbol("block_size", constexpr=True)


@ninetoothed.jit
def kernel(
    input_row: Tensor(2, other=float("-inf")).tile((1, block_size)),
    output_row: Tensor(2).tile((1, block_size)),
):
    # TODO: implement stable row-wise softmax.
    row_minus_max = input_row
    numerator = row_minus_max
    denominator = 1
    output_row = numerator / denominator
"""


MODULES: tuple[Module, ...] = (
    Module(
        "orientation",
        {"zh": "第一章：符号张量入门", "en": "Chapter 1: Symbolic Tensors"},
        {
            "zh": "先学会 NineToothed 里的张量为什么不是数据，而是可求值的索引地图。",
            "en": "Learn why a NineToothed tensor is an evaluable index map, not data storage.",
        },
    ),
    Module(
        "meta",
        {"zh": "第二章：元操作地图", "en": "Chapter 2: Meta-Operation Maps"},
        {
            "zh": "用 tile、expand、slice、permute 把张量排布成适合程序实例处理的形状。",
            "en": "Use tile, expand, slicing, and permute to arrange tensors for program instances.",
        },
    ),
    Module(
        "arrange",
        {"zh": "第三章：排布与程序网格", "en": "Chapter 3: Arrangement and Program Grids"},
        {
            "zh": "理解最外层形状如何决定 kernel 会启动多少个程序实例。",
            "en": "Understand how the outermost shape decides the program launch grid.",
        },
    ),
    Module(
        "apply",
        {"zh": "第四章：应用函数", "en": "Chapter 4: Application Functions"},
        {
            "zh": "看清应用函数拿到的是局部块，并开始使用 ninetoothed.language 做块内计算。",
            "en": "See that application functions receive local blocks and compute with ninetoothed.language.",
        },
    ),
    Module(
        "tools",
        {"zh": "第五章：语言操作与工程工具", "en": "Chapter 5: Language Ops and Tools"},
        {
            "zh": "在进入更大 kernel 前，认识 ntl 常用操作、调试、可视化、JIT/make/AOT 和自动调优入口。",
            "en": "Before larger kernels, learn common ntl operations, debugging, visualization, JIT/make/AOT, and auto-tuning hooks.",
        },
    ),
    Module(
        "kernels",
        {"zh": "第六章：真正写 kernel", "en": "Chapter 6: Real Kernels"},
        {
            "zh": "编辑真实 Python 文件，通过 CUDA 判题器验证你的 NineToothed kernel。",
            "en": "Edit real Python files and validate your NineToothed kernels on CUDA.",
        },
    ),
)


QUESTS: tuple[Quest, ...] = (
    Quest(
        slug="tensor-rank",
        module="orientation",
        kind="answer",
        title={"zh": "张量不是数组", "en": "A Tensor Is Not an Array"},
        subtitle={"zh": "从维度数开始，而不是从数据开始。", "en": "Start with rank, not stored values."},
        concepts=("Tensor", "ndim"),
        lesson=(
            LessonBlock(
                {"zh": "核心概念", "en": "Core idea"},
                {
                    "zh": "在 PyTorch 里，tensor 通常装着真实数据；在 NineToothed 里，Tensor 更像一张符号索引地图。`Tensor(2)` 表示一个二维符号张量，它还不知道具体形状。",
                    "en": "In PyTorch a tensor usually stores values. In NineToothed, Tensor is closer to a symbolic index map. `Tensor(2)` means a rank-2 symbolic tensor whose concrete shape is still unknown.",
                },
                "from ninetoothed import Tensor\nx = Tensor(2)\nprint(x.shape)",
            ),
        ),
        question={"zh": "`Tensor(2)` 的维度数是多少？", "en": "What is the rank of `Tensor(2)`?"},
        expected=2,
        answer_note={"zh": "请输入一个数字。", "en": "Enter one number."},
        hint={"zh": "括号里的 2 不是形状，而是 ndim。", "en": "The 2 is ndim, not a concrete shape."},
        success={"zh": "对。你已经区分了维度数和具体形状。", "en": "Correct. You separated rank from concrete shape."},
    ),
    Quest(
        slug="eval-vector",
        module="orientation",
        kind="answer",
        title={"zh": "让索引地图显形", "en": "Reveal the Index Map"},
        subtitle={"zh": "用 eval 看见符号张量对应的源索引。", "en": "Use eval to see source indices."},
        concepts=("Tensor.eval", "subs"),
        lesson=(
            LessonBlock(
                {"zh": "为什么要 eval", "en": "Why eval exists"},
                {
                    "zh": "符号张量不能直接当数据打印。给它一个具体形状后，`eval` 会返回每个位置对应的源索引，这对理解排布非常关键。",
                    "en": "A symbolic tensor is not data. Once you provide a concrete shape, `eval` returns the source index at each position, which is crucial for understanding arrangement.",
                },
                "x = Tensor(1)\nobserved = x.eval({x: Tensor(shape=(5,))})",
            ),
        ),
        question={"zh": "`observed` 是什么？", "en": "What is `observed`?"},
        expected=[0, 1, 2, 3, 4],
        answer_note={"zh": "用 Python 字面量，例如 `[1, 2]`。", "en": "Use Python literal syntax, for example `[1, 2]`."},
        hint={"zh": "长度为 5 的一维张量，源索引从 0 到 4。", "en": "A length-5 vector has source indices 0 through 4."},
        success={"zh": "很好。你已经能读一维索引地图了。", "en": "Good. You can read a 1D index map."},
    ),
    Quest(
        slug="subs-shape",
        module="orientation",
        kind="answer",
        title={"zh": "subs 仍然返回符号张量", "en": "subs Still Returns a Tensor"},
        subtitle={"zh": "代入形状，但不把张量变成 ndarray。", "en": "Substitute shape without turning the tensor into ndarray."},
        concepts=("Tensor.subs", "shape"),
        lesson=(
            LessonBlock(
                {"zh": "eval 和 subs 的差别", "en": "eval versus subs"},
                {
                    "zh": "`eval` 直接给出可观察的索引数组；`subs` 则返回一个已经代入具体符号值的 NineToothed Tensor，后面还可以继续做可视化或元操作。",
                    "en": "`eval` returns an observable index array. `subs` returns a NineToothed Tensor with symbols replaced, so it can still be visualized or arranged further.",
                },
                "x = Tensor(2)\ny = x.subs({x: Tensor(shape=(2, 3))})\nprint(y.shape)",
            ),
        ),
        question={"zh": "`y.shape` 是什么？", "en": "What is `y.shape`?"},
        expected=[2, 3],
        accepted=([2, 3], (2, 3)),
        answer_note={"zh": "可以输入 `[2, 3]`。", "en": "You can enter `[2, 3]`."},
        hint={"zh": "这次只代入形状，没有改变维度顺序。", "en": "Only the shape is substituted; no dimension order changed."},
        success={"zh": "对。subs 是继续学习排布时很有用的中间状态。", "en": "Correct. subs gives a useful intermediate state for studying arrangement."},
    ),
    Quest(
        slug="tile-even",
        module="meta",
        kind="answer",
        title={"zh": "整齐分块", "en": "Clean Tiling"},
        subtitle={"zh": "先看没有越界的 tile。", "en": "Start with tiling that has no boundary padding."},
        concepts=("tile", "nested tensor"),
        lesson=(
            LessonBlock(
                {"zh": "tile 创建嵌套张量", "en": "tile creates a nested tensor"},
                {
                    "zh": "`tile((2,))` 会把一维张量切成外层块，每个外层元素都是长度为 2 的内层张量。",
                    "en": "`tile((2,))` splits a vector into outer blocks; each outer element is an inner tensor of length 2.",
                },
                "x = Tensor(1)\nobserved = x.tile((2,)).eval({x: Tensor(shape=(6,))})",
            ),
        ),
        question={"zh": "`observed` 是什么？", "en": "What is `observed`?"},
        expected=[[0, 1], [2, 3], [4, 5]],
        answer_note={"zh": "输入嵌套列表。", "en": "Enter a nested list."},
        hint={"zh": "长度 6，每块 2 个元素，所以外层有 3 块。", "en": "Length 6 with block size 2 gives 3 outer blocks."},
        success={"zh": "对。现在你看到了 tile 的两层结构。", "en": "Correct. You can see the two-level structure from tile."},
    ),
    Quest(
        slug="tile-padding",
        module="meta",
        kind="answer",
        title={"zh": "边界补位", "en": "Boundary Padding"},
        subtitle={"zh": "NineToothed 如何表示越界位置。", "en": "How NineToothed represents out-of-bounds positions."},
        concepts=("tile", "padding", "other"),
        lesson=(
            LessonBlock(
                {"zh": "默认可视化里的 -1", "en": "The default -1 in eval"},
                {
                    "zh": "当最后一块不满时，`eval` 里默认用 `-1` 表示越界位置。真正写 kernel 时可以用 `Tensor(..., other=0)` 等方式指定越界读取值。",
                    "en": "When the last block is partial, `eval` shows out-of-bounds positions as `-1` by default. In real kernels, `Tensor(..., other=0)` can define the value loaded out of bounds.",
                },
                "x = Tensor(1)\nobserved = x.tile((2,)).eval({x: Tensor(shape=(5,))})",
            ),
        ),
        question={"zh": "`observed` 是什么？", "en": "What is `observed`?"},
        expected=[[0, 1], [2, 3], [4, -1]],
        answer_note={"zh": "注意最后一块。", "en": "Watch the final block."},
        hint={"zh": "最后只剩一个真实元素。", "en": "Only one real element remains in the final block."},
        success={"zh": "对。边界处理是写 kernel 前必须形成的直觉。", "en": "Correct. Boundary intuition matters before writing kernels."},
    ),
    Quest(
        slug="tile-2d-shape",
        module="meta",
        kind="answer",
        title={"zh": "二维分块的形状", "en": "Shape of 2D Tiling"},
        subtitle={"zh": "区分外层块网格和内层块形状。", "en": "Separate the outer block grid from the inner block shape."},
        concepts=("tile", "shape", "dtype.shape"),
        lesson=(
            LessonBlock(
                {"zh": "外层和内层", "en": "Outer and inner levels"},
                {
                    "zh": "一个 `(4, 6)` 矩阵按 `(2, 3)` 分块后，外层是块网格 `(2, 2)`，内层每块是 `(2, 3)`。",
                    "en": "A `(4, 6)` matrix tiled by `(2, 3)` has an outer block grid `(2, 2)` and each inner block has shape `(2, 3)`.",
                },
                "x = Tensor(shape=(4, 6))\ny = x.tile((2, 3))\nprint(y.shape, y.dtype.shape)",
            ),
        ),
        question={"zh": "请回答 `[外层行块数, 外层列块数, 内层行数, 内层列数]`。", "en": "Answer `[outer_rows, outer_cols, inner_rows, inner_cols]`."},
        expected=[2, 2, 2, 3],
        answer_note={"zh": "例如 `[1, 2, 3, 4]`。", "en": "For example, `[1, 2, 3, 4]`."},
        hint={"zh": "4/2=2，6/3=2。", "en": "4/2=2 and 6/3=2."},
        success={"zh": "对。你已经开始能读嵌套张量的形状了。", "en": "Correct. You can read nested tensor shapes."},
    ),
    Quest(
        slug="flatten-inner",
        module="meta",
        kind="answer",
        title={"zh": "压平内层块", "en": "Flatten the Inner Block"},
        subtitle={"zh": "只改变内层 dtype，不改变外层程序网格。", "en": "Change the inner dtype shape without changing the outer grid."},
        concepts=("dtype", "flatten"),
        lesson=(
            LessonBlock(
                {"zh": "为什么改 dtype", "en": "Why assign dtype"},
                {
                    "zh": "嵌套张量的 `dtype` 可以本身也是 Tensor。给 `x_tiled.dtype` 赋值，等于改变每个外层块内部的局部视图。",
                    "en": "The `dtype` of a nested tensor can itself be a Tensor. Assigning `x_tiled.dtype` changes the local view inside each outer block.",
                },
                "x = Tensor(shape=(4, 4))\ny = x.tile((2, 2))\ny.dtype = y.dtype.flatten()",
            ),
        ),
        question={"zh": "此时 `y` 的整体 eval 形状是什么？", "en": "What is the full eval shape of `y` now?"},
        expected=[2, 2, 4],
        accepted=([2, 2, 4], (2, 2, 4)),
        answer_note={"zh": "外层 `(2,2)` 不变，内层 `(2,2)` 被压平成 `4`。", "en": "Outer `(2,2)` stays; inner `(2,2)` flattens to `4`."},
        hint={"zh": "只压平每个块内部。", "en": "Only flatten inside each block."},
        success={"zh": "对。局部视图和程序网格是两件事。", "en": "Correct. Local view and launch grid are separate."},
    ),
    Quest(
        slug="unsqueeze-expand",
        module="meta",
        kind="answer",
        title={"zh": "广播桥", "en": "Broadcast Bridge"},
        subtitle={"zh": "用 unsqueeze + expand 对齐形状。", "en": "Align shapes with unsqueeze + expand."},
        concepts=("unsqueeze", "expand"),
        lesson=(
            LessonBlock(
                {"zh": "先加轴，再扩展", "en": "Add an axis, then expand"},
                {
                    "zh": "`unsqueeze(0)` 把一维视图变成一行；`expand((3, -1))` 把这一行扩成三行，`-1` 表示保持原维度大小。",
                    "en": "`unsqueeze(0)` turns a vector view into one row; `expand((3, -1))` repeats it into three rows, with `-1` preserving the original dimension.",
                },
                "x = Tensor(1)\nobserved = x.unsqueeze(0).expand((3, -1)).eval({x: Tensor(shape=(4,))})",
            ),
        ),
        question={"zh": "`observed` 是什么？", "en": "What is `observed`?"},
        expected=[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
        answer_note={"zh": "输入三行嵌套列表。", "en": "Enter a three-row nested list."},
        hint={"zh": "三行都引用同一个一维索引地图。", "en": "All three rows reference the same vector index map."},
        success={"zh": "对。后面 matmul 对齐块网格会大量用到这个直觉。", "en": "Correct. This intuition is used heavily when aligning matmul block grids."},
    ),
    Quest(
        slug="squeeze-axis",
        module="meta",
        kind="answer",
        title={"zh": "去掉长度为 1 的轴", "en": "Remove a Size-1 Axis"},
        subtitle={"zh": "squeeze 让局部索引更顺手。", "en": "squeeze makes local indexing cleaner."},
        concepts=("squeeze", "shape", "eval"),
        lesson=(
            LessonBlock(
                {"zh": "为什么 squeeze", "en": "Why squeeze"},
                {
                    "zh": "排布过程中经常会临时出现长度为 1 的轴。`squeeze(0)` 可以去掉第 0 维，让后续 application 里少写一层索引。",
                    "en": "Arrangement often creates temporary size-1 axes. `squeeze(0)` removes axis 0 so the application needs one less index.",
                },
                "x = Tensor(shape=(1, 4))\ny = x.squeeze(0)\n# y.shape == (4,)",
            ),
        ),
        question={"zh": "补全 solve()，返回 `x.squeeze(0)` 后的形状。", "en": "Complete solve() and return the shape after `x.squeeze(0)`."},
        expected=[4],
        accepted=([4], (4,)),
        answer_note={"zh": "请让 solve() 返回 `[4]`，但要用 `.squeeze(0)` 算出来。", "en": "Return `[4]`, computed with `.squeeze(0)`."},
        hint={"zh": "用 `list(y.shape)`。", "en": "Use `list(y.shape)`."},
        success={"zh": "对。squeeze 能清理排布后的多余轴。", "en": "Correct. squeeze cleans up extra arrangement axes."},
        required_fragments=(".squeeze",),
    ),
    Quest(
        slug="ravel-observe",
        module="meta",
        kind="answer",
        title={"zh": "观察 ravel", "en": "Observe ravel"},
        subtitle={"zh": "先用 eval 观察真实行为，不靠猜。", "en": "Use eval to observe behavior instead of guessing."},
        concepts=("ravel", "eval"),
        lesson=(
            LessonBlock(
                {"zh": "API 学习要看行为", "en": "API learning needs behavior checks"},
                {
                    "zh": "`ravel()` 是 Tensor API 的一部分。学习元操作时，不要只记名字，要用小张量 eval 出来观察它对索引地图的影响。",
                    "en": "`ravel()` is part of the Tensor API. When learning meta-operations, do not just memorize the name; evaluate a small tensor to inspect its index map.",
                },
                "x = Tensor(shape=(2, 3))\ny = x.ravel()\nobserved = y.eval()",
            ),
        ),
        question={"zh": "补全 solve()，返回 `Tensor(shape=(2,3)).ravel().eval().tolist()`。", "en": "Complete solve() and return `Tensor(shape=(2,3)).ravel().eval().tolist()`."},
        expected=[[0, 1, 2], [3, 4, 5]],
        answer_note={"zh": "重点是用 `.ravel()` 和 `.eval()` 观察。", "en": "The point is to observe with `.ravel()` and `.eval()`."},
        hint={"zh": "当前版本中这个小例子的 eval 仍是二维索引地图。", "en": "In this version, this tiny example still evaluates as a 2D index map."},
        success={"zh": "对。你用实验确认了 ravel 的当前行为。", "en": "Correct. You verified ravel behavior experimentally."},
        required_fragments=(".ravel", ".eval"),
    ),
    Quest(
        slug="pad-border",
        module="meta",
        kind="answer",
        title={"zh": "pad 的边界地图", "en": "The Boundary Map of pad"},
        subtitle={"zh": "padding 不是魔法，它也能 eval 出索引关系。", "en": "Padding is not magic; eval can reveal its index relation."},
        concepts=("pad", "eval", "padding"),
        lesson=(
            LessonBlock(
                {"zh": "二维 padding", "en": "2D padding"},
                {
                    "zh": "`x.pad(((1, 0), (0, 1)))` 会在上方补一行、右侧补一列。`eval` 里补出来的位置显示为 `-1`。",
                    "en": "`x.pad(((1, 0), (0, 1)))` pads one row on top and one column on the right. Padded positions show as `-1` in eval.",
                },
                "x = Tensor(shape=(2, 2))\nobserved = x.pad(((1, 0), (0, 1))).eval()",
            ),
        ),
        question={"zh": "补全 solve()，返回上述 observed 的 `.tolist()`。", "en": "Complete solve() and return `observed.tolist()`."},
        expected=[[-1, -1, -1], [0, 1, -1], [2, 3, -1]],
        answer_note={"zh": "要在代码里调用 `.pad(...)`，不要手写常量。", "en": "Call `.pad(...)` in code instead of hard-coding the constant."},
        hint={"zh": "原始 2x2 索引地图是 `[[0,1],[2,3]]`。", "en": "The original 2x2 index map is `[[0,1],[2,3]]`."},
        success={"zh": "对。pad 的越界区域也已经可视化了。", "en": "Correct. The out-of-bounds padding area is now visible."},
        required_fragments=(".pad", ".eval"),
    ),
    Quest(
        slug="slice-index",
        module="meta",
        kind="answer",
        title={"zh": "切片钥匙", "en": "The Slice Key"},
        subtitle={"zh": "切片也是排布语言的一部分。", "en": "Slicing is part of the arrangement language."},
        concepts=("getitem", "slicing"),
        lesson=(
            LessonBlock(
                {"zh": "像 NumPy 一样读", "en": "Read it like NumPy"},
                {
                    "zh": "NineToothed 的切片可以先按 NumPy 的索引地图理解：`Tensor(shape=(3,4,2))` 的 eval 就像 `np.arange(24).reshape(3,4,2)`。",
                    "en": "NineToothed slicing can first be understood like a NumPy index map: `Tensor(shape=(3,4,2)).eval()` resembles `np.arange(24).reshape(3,4,2)`.",
                },
                "x = Tensor(shape=(3, 4, 2))\nobserved = x[:, 1, :].eval()",
            ),
        ),
        question={"zh": "`observed` 是什么？", "en": "What is `observed`?"},
        expected=[[2, 3], [10, 11], [18, 19]],
        answer_note={"zh": "取第二个中间维度。", "en": "Take index 1 along the middle dimension."},
        hint={"zh": "可以想象 `np.arange(24).reshape(3,4,2)[:,1,:]`。", "en": "Imagine `np.arange(24).reshape(3,4,2)[:,1,:]`."},
        success={"zh": "对。熟悉切片后，复杂排布会少很多恐惧感。", "en": "Correct. Slicing makes complex arrangements less intimidating."},
    ),
    Quest(
        slug="permute-map",
        module="meta",
        kind="answer",
        title={"zh": "转置索引地图", "en": "Permute the Index Map"},
        subtitle={"zh": "permute 改变的是索引视图。", "en": "permute changes the index view."},
        concepts=("permute", "eval"),
        lesson=(
            LessonBlock(
                {"zh": "先用小矩阵观察", "en": "Observe with a tiny matrix"},
                {
                    "zh": "对于 `Tensor(shape=(2,3))`，原始 eval 是 `[[0,1,2],[3,4,5]]`。`permute((1,0))` 后就像转置这个索引地图。",
                    "en": "For `Tensor(shape=(2,3))`, the original eval is `[[0,1,2],[3,4,5]]`. `permute((1,0))` transposes that index map.",
                },
                "x = Tensor(shape=(2, 3))\nobserved = x.permute((1, 0)).eval()",
            ),
        ),
        question={"zh": "`observed` 是什么？", "en": "What is `observed`?"},
        expected=[[0, 3], [1, 4], [2, 5]],
        answer_note={"zh": "输入转置后的索引地图。", "en": "Enter the transposed index map."},
        hint={"zh": "第一列 `[0,3]` 变成第一行。", "en": "The first column `[0,3]` becomes the first row."},
        success={"zh": "对。permute 是从排布走向卷积、attention 前的重要工具。", "en": "Correct. permute becomes important for convolution and attention arrangements."},
    ),
    Quest(
        slug="block-size-symbol",
        module="arrange",
        kind="answer",
        title={"zh": "块大小为什么是符号", "en": "Why Block Size Is Symbolic"},
        subtitle={"zh": "constexpr 是编译时信息。", "en": "constexpr is compile-time information."},
        concepts=("Symbol", "constexpr", "block_size"),
        lesson=(
            LessonBlock(
                {"zh": "调试时先用 constexpr", "en": "Use constexpr while debugging"},
                {
                    "zh": "NineToothed 的最内层张量形状需要在编译时确定。刚学习或调试时，用 `Symbol(..., constexpr=True)` 明确传入块大小，比一上来自动调优更可控。",
                    "en": "NineToothed needs innermost tensor shapes at compile time. While learning or debugging, `Symbol(..., constexpr=True)` with an explicit block size is more predictable than auto-tuning.",
                },
                "block_size = Symbol('block_size', constexpr=True)\nx.tile((block_size,))",
            ),
        ),
        question={"zh": "如果向量长度是 10、block_size 是 4，外层块数量是多少？", "en": "If vector length is 10 and block_size is 4, how many outer blocks are there?"},
        expected=3,
        answer_note={"zh": "向上取整。", "en": "Use ceiling division."},
        hint={"zh": "10 个元素需要 4、4、2 三块。", "en": "10 elements need blocks of 4, 4, and 2."},
        success={"zh": "对。程序网格通常就来自这个外层数量。", "en": "Correct. The program grid often comes from this outer count."},
    ),
    Quest(
        slug="block-size-helper",
        module="arrange",
        kind="answer",
        title={"zh": "自动调优入口 block_size()", "en": "The block_size() Auto-Tuning Hook"},
        subtitle={"zh": "从 constexpr 过渡到可调配置。", "en": "Move from constexpr to tunable configuration."},
        concepts=("block_size", "auto-tuning", "Symbol"),
        lesson=(
            LessonBlock(
                {"zh": "什么时候用 block_size()", "en": "When to use block_size()"},
                {
                    "zh": "开发早期常用 `Symbol(..., constexpr=True)` 固定块大小；性能调优阶段可以改用 `ninetoothed.block_size()`，让 NineToothed 生成可调配置。",
                    "en": "Early development often uses `Symbol(..., constexpr=True)` for fixed block size. During tuning, `ninetoothed.block_size()` lets NineToothed generate tunable configurations.",
                },
                "from ninetoothed import block_size\nBLOCK_SIZE = block_size()",
            ),
        ),
        question={"zh": "补全 solve()，创建 `BLOCK_SIZE = block_size()`，并返回字符串 `'block_size'`。", "en": "Complete solve(): create `BLOCK_SIZE = block_size()` and return `'block_size'`."},
        expected="block_size",
        answer_note={"zh": "判题会检查你确实调用了 `block_size()`。", "en": "The judge checks that you really call `block_size()`."},
        hint={"zh": "文件开头已经导入了 `block_size`。", "en": "`block_size` is already imported at the top of the file."},
        success={"zh": "对。你知道什么时候从固定块大小走向自动调优。", "en": "Correct. You know where auto-tuning enters the flow."},
        required_fragments=("block_size()",),
    ),
    Quest(
        slug="visualize-ready",
        module="arrange",
        kind="answer",
        title={"zh": "可视化前先具体化", "en": "Substitute Before Visualizing"},
        subtitle={"zh": "visualize 需要一个具体化后的张量。", "en": "visualize needs a concrete tensor."},
        concepts=("Tensor.subs", "visualization", "tile"),
        lesson=(
            LessonBlock(
                {"zh": "为什么先 subs", "en": "Why subs first"},
                {
                    "zh": "排布后的符号张量如果还带着未知形状，直接看图很困难。先用 `subs` 代入一个小形状，就可以把外层和内层结构看清楚。",
                    "en": "If an arranged tensor still has unknown shapes, it is hard to inspect visually. Substitute a small concrete shape first to reveal outer and inner structure.",
                },
                "x = Tensor(2)\ny = x.tile((2, 2))\nconcrete = y.subs({x: Tensor(shape=(4, 4))})",
            ),
        ),
        question={"zh": "补全 solve()，返回 `[list(concrete.shape), list(concrete.dtype.shape)]`。", "en": "Complete solve() and return `[list(concrete.shape), list(concrete.dtype.shape)]`."},
        expected=[[2, 2], [2, 2]],
        answer_note={"zh": "要使用 `.tile` 和 `.subs`。", "en": "Use `.tile` and `.subs`."},
        hint={"zh": "4x4 按 2x2 分块，外层和内层都是 2x2。", "en": "A 4x4 tensor tiled by 2x2 has both outer and inner shapes of 2x2."},
        success={"zh": "对。可视化之前先构造小而具体的案例。", "en": "Correct. Build a tiny concrete case before visualizing."},
        required_fragments=(".tile", ".subs"),
    ),
    Quest(
        slug="arrange-three-vectors",
        module="arrange",
        kind="answer",
        title={"zh": "三个向量同排布", "en": "Arrange Three Vectors Together"},
        subtitle={"zh": "参数之间必须建立同一个外层关系。", "en": "Parameters need a shared outer relationship."},
        concepts=("arrangement", "tile"),
        lesson=(
            LessonBlock(
                {"zh": "排布函数做什么", "en": "What arrangement does"},
                {
                    "zh": "`arrangement(x,y,z)` 不计算 `z=x+y`；它只决定每个程序实例会拿到 x、y、z 的哪一块。",
                    "en": "`arrangement(x,y,z)` does not compute `z=x+y`; it only decides which local block of x, y, and z each program instance receives.",
                },
                "def arrangement(x, y, z, block_size=block_size):\n    return x.tile((block_size,)), y.tile((block_size,)), z.tile((block_size,))",
            ),
        ),
        question={"zh": "长度 8、block_size 4 时，外层程序数量是多少？", "en": "For length 8 and block_size 4, how many program instances are launched?"},
        expected=2,
        answer_note={"zh": "每个外层块对应一个程序实例。", "en": "Each outer block maps to one program instance."},
        hint={"zh": "8 被分成两个长度为 4 的块。", "en": "8 splits into two blocks of length 4."},
        success={"zh": "对。排布不是计算，但它决定并行结构。", "en": "Correct. Arrangement is not computation, but it defines parallel structure."},
    ),
    Quest(
        slug="program-local-block",
        module="arrange",
        kind="answer",
        title={"zh": "程序实例看到什么", "en": "What a Program Instance Sees"},
        subtitle={"zh": "应用函数处理的是局部块。", "en": "The application function receives local blocks."},
        concepts=("program instance", "local block"),
        lesson=(
            LessonBlock(
                {"zh": "外层索引选择局部块", "en": "Outer index selects a local block"},
                {
                    "zh": "如果 `x.tile((4,))` 后 eval 是 `[[0,1,2,3],[4,5,6,7]]`，第 0 个程序实例拿第一块，第 1 个程序实例拿第二块。",
                    "en": "If `x.tile((4,)).eval()` is `[[0,1,2,3],[4,5,6,7]]`, program 0 receives the first block and program 1 receives the second.",
                },
            ),
        ),
        question={"zh": "第 1 个程序实例拿到的 x 局部块是什么？", "en": "What local x block does program instance 1 receive?"},
        expected=[4, 5, 6, 7],
        answer_note={"zh": "注意程序实例从 0 开始编号。", "en": "Program instances are zero-indexed."},
        hint={"zh": "第 1 个实例对应第二个外层块。", "en": "Program 1 maps to the second outer block."},
        success={"zh": "对。现在 application 的参数含义就清楚了。", "en": "Correct. Now the meaning of application parameters is clearer."},
    ),
    Quest(
        slug="outer-shape-rule",
        module="arrange",
        kind="answer",
        title={"zh": "外层形状必须对齐", "en": "Outer Shapes Must Align"},
        subtitle={"zh": "多个参数的最外层网格要一致。", "en": "Multiple parameters must share the same outer grid."},
        concepts=("expand", "outer shape"),
        lesson=(
            LessonBlock(
                {"zh": "为什么 matmul 需要 expand", "en": "Why matmul needs expand"},
                {
                    "zh": "矩阵乘法里，输出块网格是 `(M块, N块)`。lhs 只有行方向，rhs 只有列方向，所以要用 `expand` 把它们对齐到输出块网格。",
                    "en": "In matmul, the output block grid is `(M_blocks, N_blocks)`. lhs naturally follows rows and rhs follows columns, so `expand` aligns both to the output grid.",
                },
            ),
        ),
        question={"zh": "让 lhs/rhs 对齐 output 外层网格的关键元操作是什么？", "en": "Which meta-operation aligns lhs/rhs to the output outer grid?"},
        expected="expand",
        accepted=("expand", "Expand", "EXPAND"),
        answer_note={"zh": "输入一个函数名。", "en": "Enter one function name."},
        hint={"zh": "它会扩展外层视图而不复制真实数据。", "en": "It expands the outer view without copying real data."},
        success={"zh": "对。expand 是建立参数关系的关键。", "en": "Correct. expand is key to relating parameters."},
    ),
    Quest(
        slug="matmul-output-grid",
        module="arrange",
        kind="answer",
        title={"zh": "矩阵乘法输出网格", "en": "Matmul Output Grid"},
        subtitle={"zh": "先只看输出块，不急着写 dot。", "en": "Study output blocks before writing dot."},
        concepts=("matmul", "tile", "output grid"),
        lesson=(
            LessonBlock(
                {"zh": "输出决定程序数量", "en": "Output decides program count"},
                {
                    "zh": "若 output shape 是 `(4,8)`，按 `(2,4)` 分块，输出块网格就是 `(2,2)`。每个输出块对应一个程序实例。",
                    "en": "If output shape is `(4,8)` and block shape is `(2,4)`, the output block grid is `(2,2)`. Each output block maps to one program instance.",
                },
            ),
        ),
        question={"zh": "请回答输出块网格形状。", "en": "What is the output block grid shape?"},
        expected=[2, 2],
        accepted=([2, 2], (2, 2)),
        answer_note={"zh": "输入 `[行块数, 列块数]`。", "en": "Enter `[row_blocks, col_blocks]`."},
        hint={"zh": "4/2=2，8/4=2。", "en": "4/2=2 and 8/4=2."},
        success={"zh": "对。matmul 的并行网格先定住了。", "en": "Correct. The matmul parallel grid is now fixed."},
    ),
    Quest(
        slug="matmul-k-blocks",
        module="arrange",
        kind="answer",
        title={"zh": "K 方向迭代块数", "en": "K-Block Iteration Count"},
        subtitle={"zh": "每个输出块内部还要沿 K 累加。", "en": "Each output block still accumulates along K."},
        concepts=("matmul", "K blocks", "loop"),
        lesson=(
            LessonBlock(
                {"zh": "为什么 application 里有 for k", "en": "Why application has `for k`"},
                {
                    "zh": "如果 K=6、BLOCK_SIZE_K=3，那么每个输出块需要做 2 次小矩阵乘法并累加。这个 2 不是外层程序数量，而是每个程序内部的循环次数。",
                    "en": "If K=6 and BLOCK_SIZE_K=3, each output block does 2 small matrix multiplies and accumulates them. This 2 is an inner loop count, not the outer program count.",
                },
            ),
        ),
        question={"zh": "K=6、BLOCK_SIZE_K=3 时，每个程序内部循环几次？", "en": "For K=6 and BLOCK_SIZE_K=3, how many inner loop iterations are needed?"},
        expected=2,
        answer_note={"zh": "输入一个数字。", "en": "Enter one number."},
        hint={"zh": "6 被分成两个长度为 3 的 K 块。", "en": "6 splits into two K-blocks of length 3."},
        success={"zh": "对。外层并行和内层迭代终于分开了。", "en": "Correct. Outer parallelism and inner iteration are now separated."},
    ),
    Quest(
        slug="application-block",
        module="apply",
        kind="answer",
        title={"zh": "应用函数不是拿全量张量", "en": "Application Does Not Receive the Whole Tensor"},
        subtitle={"zh": "它拿的是当前程序实例的局部块。", "en": "It receives the current program instance's local block."},
        concepts=("application", "local tensor"),
        lesson=(
            LessonBlock(
                {"zh": "vector add 的 application", "en": "Vector-add application"},
                {
                    "zh": "在向量加法里，如果排布使用 `tile((4,))`，那么 application 里的 `x`、`y`、`z` 都是长度为 4 的局部块，而不是完整向量。",
                    "en": "In vector add, if arrangement uses `tile((4,))`, then `x`, `y`, and `z` inside application are local blocks of length 4, not full vectors.",
                },
                "def application(x, y, z):\n    z = x + y",
            ),
        ),
        question={"zh": "block_size=4 时，application 里的 `x.shape` 是多少？", "en": "With block_size=4, what is `x.shape` inside application?"},
        expected=[4],
        accepted=([4], (4,)),
        answer_note={"zh": "输入 `[4]` 这种形状。", "en": "Enter a shape like `[4]`."},
        hint={"zh": "它只看到局部块。", "en": "It sees only a local block."},
        success={"zh": "对。写应用函数时一定站在单个程序实例里思考。", "en": "Correct. Think from one program instance when writing application."},
    ),
    Quest(
        slug="padding-other-zero",
        module="apply",
        kind="answer",
        title={"zh": "越界读取值", "en": "Out-of-Bounds Load Value"},
        subtitle={"zh": "求和时 padding 应该不改变结果。", "en": "For sums, padding should not change the result."},
        concepts=("Tensor(other=...)", "padding", "sum"),
        lesson=(
            LessonBlock(
                {"zh": "other 的作用", "en": "What other does"},
                {
                    "zh": "最后一块可能越界。求和时越界位置应该贡献 0，所以可用 `Tensor(2, other=0)`。如果求最大值，可能要用 `float('-inf')`。",
                    "en": "The final block may go out of bounds. For sum, out-of-bounds positions should contribute 0, so use `Tensor(2, other=0)`. For max, `float('-inf')` may be better.",
                },
            ),
        ),
        question={"zh": "做行求和时，`other` 最适合设成多少？", "en": "For row sum, what should `other` usually be?"},
        expected=0,
        answer_note={"zh": "输入一个数字。", "en": "Enter one number."},
        hint={"zh": "加法单位元。", "en": "The additive identity."},
        success={"zh": "对。边界值要跟计算语义匹配。", "en": "Correct. Boundary values must match computation semantics."},
    ),
    Quest(
        slug="reduction-sum",
        module="apply",
        kind="answer",
        title={"zh": "块内归约", "en": "Reduction Inside a Block"},
        subtitle={"zh": "使用 ninetoothed.language 做局部计算。", "en": "Use ninetoothed.language for local computation."},
        concepts=("ninetoothed.language", "ntl.sum"),
        lesson=(
            LessonBlock(
                {"zh": "ntl 类似 triton.language", "en": "ntl resembles triton.language"},
                {
                    "zh": "应用函数里可使用 `ninetoothed.language as ntl` 提供的操作。例如局部块求和常用 `ntl.sum(block, axis=-1)`。",
                    "en": "Inside application, use operations from `ninetoothed.language as ntl`. For example, local block reduction often uses `ntl.sum(block, axis=-1)`.",
                },
                "acc += ntl.sum(x[0, i], axis=-1)",
            ),
        ),
        question={"zh": "补全 solve()，把局部块归约绑定到 `ntl.sum`。", "en": "Complete solve() by binding the local-block reducer to `ntl.sum`."},
        expected="ntl.sum",
        accepted=("ntl.sum", "sum", "ntl.sum()"),
        answer_note={"zh": "代码里要引用 `ntl.sum`，不是只记住一个名字。", "en": "Reference `ntl.sum` in code, not just as a memorized name."},
        hint={"zh": "它在 ninetoothed.language 里，后面 row-sum 会真正调用它。", "en": "It lives in ninetoothed.language and will be called for real in row-sum."},
        success={"zh": "对。你已经准备好补全 row-sum kernel 了。", "en": "Correct. You are ready to complete the row-sum kernel."},
    ),
    Quest(
        slug="ntl-zeros-accumulator",
        module="tools",
        kind="answer",
        title={"zh": "初始化累加器", "en": "Initialize an Accumulator"},
        subtitle={"zh": "很多 kernel 都从一个局部 accumulator 开始。", "en": "Many kernels begin with a local accumulator."},
        concepts=("ninetoothed.language", "ntl.zeros"),
        lesson=(
            LessonBlock(
                {"zh": "zeros 的位置", "en": "Where zeros fits"},
                {
                    "zh": "在 matmul、row-sum 这类应用函数里，常先用 `ntl.zeros(output.shape, dtype=ntl.float32)` 创建局部累加器，再逐步累加。",
                    "en": "In matmul or row-sum application functions, you often create a local accumulator with `ntl.zeros(output.shape, dtype=ntl.float32)` before accumulating.",
                },
                "accumulator = ntl.zeros(output.shape, dtype=ntl.float32)",
            ),
        ),
        question={"zh": "补全 application() 里的全零累加器初始化。", "en": "Complete the zero-accumulator initialization inside application()."},
        expected="ntl.zeros",
        answer_note={"zh": "填入类似 `ntl.zeros(output.shape, dtype=ntl.float32)` 的表达式。", "en": "Fill an expression like `ntl.zeros(output.shape, dtype=ntl.float32)`."},
        hint={"zh": "累加器形状应当跟局部 output 一致。", "en": "The accumulator shape should match the local output."},
        success={"zh": "对。局部累加器是很多应用函数的起点。", "en": "Correct. Local accumulators start many application functions."},
        required_fragments=("ntl.zeros",),
    ),
    Quest(
        slug="ntl-full-sentinel",
        module="tools",
        kind="answer",
        title={"zh": "创建哨兵值", "en": "Create a Sentinel Value"},
        subtitle={"zh": "最大值归约常需要 `-inf` 初值。", "en": "Max reductions often need `-inf` initialization."},
        concepts=("ninetoothed.language", "ntl.full"),
        lesson=(
            LessonBlock(
                {"zh": "full 的用途", "en": "What full is for"},
                {
                    "zh": "`ntl.full(shape, value, dtype=...)` 可以构造指定值的局部张量。比如在线性 attention 或 softmax 里，最大值初始量常用 `float('-inf')`。",
                    "en": "`ntl.full(shape, value, dtype=...)` creates a local tensor filled with a chosen value. In softmax-like code, max accumulators often start at `float('-inf')`.",
                },
                "m_i = ntl.full((q.shape[-2],), float('-inf'), dtype=ntl.float32)",
            ),
        ),
        question={"zh": "补全 application() 里的哨兵张量初始化。", "en": "Complete the sentinel-tensor initialization inside application()."},
        expected="ntl.full",
        answer_note={"zh": "填入类似 `ntl.full(..., float('-inf'), dtype=...)` 的表达式。", "en": "Fill an expression like `ntl.full(..., float('-inf'), dtype=...)`."},
        hint={"zh": "最大值归约的初值通常不是 0，而是 `-inf`。", "en": "A max-reduction sentinel is usually `-inf`, not 0."},
        success={"zh": "对。full 是构造局部常量张量的常用入口。", "en": "Correct. full is the common entry for local constant tensors."},
        required_fragments=("ntl.full",),
    ),
    Quest(
        slug="ntl-dot-contract",
        module="tools",
        kind="answer",
        title={"zh": "小块矩阵乘", "en": "Small Block Matmul"},
        subtitle={"zh": "matmul 的 application 核心是 `ntl.dot`。", "en": "The core of matmul application is `ntl.dot`."},
        concepts=("ninetoothed.language", "ntl.dot", "matmul"),
        lesson=(
            LessonBlock(
                {"zh": "dot 在哪里出现", "en": "Where dot appears"},
                {
                    "zh": "排布已经把 lhs 的一行 K 块和 rhs 的一列 K 块对齐后，application 里循环 K 块，并用 `ntl.dot(lhs[k], rhs[k])` 做局部矩阵乘。",
                    "en": "After arrangement aligns a row of lhs K-blocks with a column of rhs K-blocks, application loops over K and uses `ntl.dot(lhs[k], rhs[k])` for local matmul.",
                },
                "accumulator += ntl.dot(lhs[k], rhs[k])",
            ),
        ),
        question={"zh": "补全 application() 里的局部小矩阵乘表达式。", "en": "Complete the local block-matmul expression inside application()."},
        expected="ntl.dot",
        answer_note={"zh": "这里要真的调用 `ntl.dot(lhs, rhs)`。", "en": "Actually call `ntl.dot(lhs, rhs)` here."},
        hint={"zh": "排布已经把 lhs/rhs 的局部 K 块对齐；application 只负责局部乘加。", "en": "Arrangement has aligned lhs/rhs local K-blocks; application only does local multiply-accumulate."},
        success={"zh": "对。matmul kernel 的 application 核心已经很明确了。", "en": "Correct. The core of a matmul application is clear now."},
        required_fragments=("ntl.dot",),
    ),
    Quest(
        slug="ntl-where-mask",
        module="tools",
        kind="answer",
        title={"zh": "条件选择和 mask", "en": "Conditional Selection and Masks"},
        subtitle={"zh": "`where` 是边界和 dropout 的常见积木。", "en": "`where` is a common building block for boundaries and dropout."},
        concepts=("ninetoothed.language", "ntl.where"),
        lesson=(
            LessonBlock(
                {"zh": "where 的直觉", "en": "where intuition"},
                {
                    "zh": "`ntl.where(condition, a, b)` 根据条件在两个局部张量之间选择。dropout、causal mask、越界 mask 都会用到它。",
                    "en": "`ntl.where(condition, a, b)` selects between two local tensors. Dropout, causal masks, and boundary masks all use it.",
                },
                "output = ntl.where(mask, value, 0)",
            ),
        ),
        question={"zh": "补全 application() 里的 mask 条件选择表达式。", "en": "Complete the masked conditional-selection expression inside application()."},
        expected="ntl.where",
        answer_note={"zh": "这里要写出 `ntl.where(mask, value, 0)` 这一类局部表达式。", "en": "Write a local expression such as `ntl.where(mask, value, 0)`."},
        hint={"zh": "mask、保留值、替代值三者一起传入。", "en": "Pass the mask, kept value, and replacement value together."},
        success={"zh": "对。后面做 mask 会自然很多。", "en": "Correct. Later masking code will feel more natural."},
        required_fragments=("ntl.where",),
    ),
    Quest(
        slug="tensor-offsets",
        module="tools",
        kind="answer",
        title={"zh": "局部 offsets", "en": "Local Offsets"},
        subtitle={"zh": "随机数和 mask 常需要知道局部元素位置。", "en": "Randomness and masks often need local element positions."},
        concepts=("offsets", "ninetoothed.language"),
        lesson=(
            LessonBlock(
                {"zh": "offsets() 的用途", "en": "What offsets() is for"},
                {
                    "zh": "应用函数里，局部张量可以调用 `offsets()` 取得局部元素相对源张量的位置。dropout 的随机 mask 就常用 `ntl.rand(seed, input.offsets())`。",
                    "en": "Inside application, a local tensor can call `offsets()` to get element positions relative to the source tensor. Dropout masks often use `ntl.rand(seed, input.offsets())`.",
                },
                "mask = ntl.rand(seed, input.offsets()) > p",
            ),
        ),
        question={"zh": "补全 solve()，返回取得局部元素位置的方法名。", "en": "Complete solve() and return the method name for local element positions."},
        expected="offsets",
        answer_note={"zh": "要在代码里出现 `.offsets()`。", "en": "Use `.offsets()` in the code."},
        hint={"zh": "它是局部 Tensor 的方法。", "en": "It is a method on local Tensor objects."},
        success={"zh": "对。你已经见到随机 mask 所需的位置信息入口。", "en": "Correct. You have seen the position hook needed for random masks."},
        required_fragments=(".offsets()",),
    ),
    Quest(
        slug="make-contract",
        module="tools",
        kind="answer",
        title={"zh": "make 的三个组成部分", "en": "The Three Parts of make"},
        subtitle={"zh": "排布、应用、参数张量模板。", "en": "Arrangement, application, and tensor templates."},
        concepts=("make", "arrangement", "application"),
        lesson=(
            LessonBlock(
                {"zh": "make 的契约", "en": "The make contract"},
                {
                    "zh": "`ninetoothed.make(arrangement, application, tensors)` 把排布函数、应用函数和参数张量模板组合成可调用 kernel。",
                    "en": "`ninetoothed.make(arrangement, application, tensors)` combines arrangement, application, and tensor templates into a callable kernel.",
                },
                "kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)))",
            ),
        ),
        question={"zh": "补全 `builder = ...`，把 arrangement、application 和模板张量接成 kernel builder。", "en": "Complete `builder = ...` to connect arrangement, application, and tensor templates into a kernel builder."},
        expected="ninetoothed.make",
        answer_note={"zh": "要在代码里引用 `ninetoothed.make`。", "en": "Reference `ninetoothed.make` in code."},
        hint={"zh": "它把 arrangement 和 application 接起来。", "en": "It connects arrangement and application."},
        success={"zh": "对。make 是理解完整 kernel 的主入口。", "en": "Correct. make is the main entry for complete kernels."},
        required_fragments=("ninetoothed.make",),
    ),
    Quest(
        slug="jit-contract",
        module="tools",
        kind="answer",
        title={"zh": "jit 参数注解", "en": "jit Parameter Annotations"},
        subtitle={"zh": "把排布写进函数签名。", "en": "Put arrangement in the function signature."},
        concepts=("jit", "Tensor annotation"),
        lesson=(
            LessonBlock(
                {"zh": "jit 写法", "en": "jit style"},
                {
                    "zh": "`@ninetoothed.jit` 可以把 `Tensor(1).tile((BLOCK_SIZE,))` 直接写在参数注解里，适合短小 kernel。",
                    "en": "`@ninetoothed.jit` can put `Tensor(1).tile((BLOCK_SIZE,))` directly in parameter annotations, which is convenient for small kernels.",
                },
                "@ninetoothed.jit\ndef kernel(x: Tensor(1).tile((BLOCK_SIZE,))):\n    ...",
            ),
        ),
        question={"zh": "补全 `decorator = ...`，对应短 kernel 上方的 `@ninetoothed.jit`。", "en": "Complete `decorator = ...`, matching the `@ninetoothed.jit` used above short kernels."},
        expected="ninetoothed.jit",
        answer_note={"zh": "要在代码里引用 `ninetoothed.jit`。", "en": "Reference `ninetoothed.jit` in code."},
        hint={"zh": "它通常以装饰器形式出现。", "en": "It often appears as a decorator."},
        success={"zh": "对。短 kernel 可以用 jit 写得更集中。", "en": "Correct. Small kernels can be written compactly with jit."},
        required_fragments=("ninetoothed.jit",),
    ),
    Quest(
        slug="debug-simulate-arrangement",
        module="tools",
        kind="answer",
        title={"zh": "模拟排布", "en": "Simulate an Arrangement"},
        subtitle={"zh": "调试时先看每个参数会怎么映射。", "en": "During debugging, inspect how each parameter maps."},
        concepts=("debugging", "simulate_arrangement"),
        lesson=(
            LessonBlock(
                {"zh": "官方调试入口", "en": "Official debugging hook"},
                {
                    "zh": "`ninetoothed.debugging.simulate_arrangement(arrangement, tensors)` 可以生成源张量和目标张量，帮助你检查排布关系。这个模块属于可选依赖 `ninetoothed[debugging]`。",
                    "en": "`ninetoothed.debugging.simulate_arrangement(arrangement, tensors)` generates source and target tensors to inspect arrangement mappings. It belongs to optional dependency `ninetoothed[debugging]`.",
                },
                "from ninetoothed.debugging import simulate_arrangement",
            ),
        ),
        question={"zh": "补全调试导入，把排布模拟器引入当前文件。", "en": "Complete the debugging import to bring the arrangement simulator into the file."},
        expected="simulate_arrangement",
        answer_note={"zh": "要在代码里引用 `simulate_arrangement`。", "en": "Reference `simulate_arrangement` in code."},
        hint={"zh": "名字里有 simulate。", "en": "The name contains simulate."},
        success={"zh": "对。复杂排布应该先模拟再写计算。", "en": "Correct. Complex arrangements should be simulated before computation."},
        required_fragments=("simulate_arrangement",),
    ),
    Quest(
        slug="visualize-api",
        module="tools",
        kind="answer",
        title={"zh": "张量可视化", "en": "Tensor Visualization"},
        subtitle={"zh": "把索引地图画出来。", "en": "Draw the index map."},
        concepts=("visualization", "visualize"),
        lesson=(
            LessonBlock(
                {"zh": "visualize", "en": "visualize"},
                {
                    "zh": "`ninetoothed.visualization.visualize(tensor, save_path=...)` 可以把具体化后的张量画成图。更复杂的 `visualize_arrangement` 需要 CUDA 环境和无符号参数。",
                    "en": "`ninetoothed.visualization.visualize(tensor, save_path=...)` draws a concrete tensor. The more complex `visualize_arrangement` requires CUDA and symbol-free parameters.",
                },
                "from ninetoothed.visualization import visualize\nvisualize(tensor, save_path='tensor.png')",
            ),
        ),
        question={"zh": "补全可视化导入，把单个 Tensor 的绘图入口引入当前文件。", "en": "Complete the visualization import to bring in the single-Tensor drawing entry point."},
        expected="visualize",
        answer_note={"zh": "要在代码里引用 `visualize`。", "en": "Reference `visualize` in code."},
        hint={"zh": "名字就是 visualize。", "en": "The name is visualize."},
        success={"zh": "对。能画出来的排布更容易讲清楚。", "en": "Correct. Visual layouts are easier to reason about."},
        required_fragments=("visualize",),
    ),
    Quest(
        slug="aot-api",
        module="tools",
        kind="answer",
        title={"zh": "AOT 导出", "en": "AOT Export"},
        subtitle={"zh": "从 Python kernel 走向工程集成。", "en": "Move from Python kernels toward integration."},
        concepts=("aot", "build", "code generation"),
        lesson=(
            LessonBlock(
                {"zh": "aot 的角色", "en": "Role of aot"},
                {
                    "zh": "`ninetoothed.aot` 可以把 kernel 生成 C++/CUDA 侧可集成的产物。它属于学习后段，不需要一开始就掌握，但要知道有这条路。",
                    "en": "`ninetoothed.aot` can generate C++/CUDA-side artifacts for integration. It is a later-stage tool, but learners should know the path exists.",
                },
                "from ninetoothed.aot import aot",
            ),
        ),
        question={"zh": "补全 AOT 导入，连到 NineToothed 的工程导出入口。", "en": "Complete the AOT import and connect to NineToothed's export entry point."},
        expected="aot",
        answer_note={"zh": "要在代码里引用 `aot`。", "en": "Reference `aot` in code."},
        hint={"zh": "缩写就是 ahead-of-time。", "en": "It abbreviates ahead-of-time."},
        success={"zh": "对。你已经知道从教学 kernel 到工程产物的出口。", "en": "Correct. You know the exit from teaching kernel to engineering artifact."},
        required_fragments=("aot",),
    ),
    Quest(
        slug="jit-vector-add",
        module="kernels",
        kind="code",
        title={"zh": "第一份 JIT kernel", "en": "First JIT Kernel"},
        subtitle={"zh": "真正编辑文件，通过 CUDA 判题。", "en": "Edit a real file and pass the CUDA judge."},
        concepts=("jit", "Tensor annotation", "vector add"),
        requires_cuda=True,
        lesson=(
            LessonBlock(
                {"zh": "你要补哪一行", "en": "The line to complete"},
                {
                    "zh": "参数注解已经完成：每个参数都是按 `block_size` tile 后的局部块。你只需要在 kernel body 里把两个局部块相加并写入 output。",
                    "en": "The parameter annotations are already done: each parameter is a local block tiled by `block_size`. You only need to add the two local blocks and write output.",
                },
            ),
        ),
        question={"zh": "编辑生成的 `vector_add_kernel.py`。", "en": "Edit the generated `vector_add_kernel.py`."},
        hint={"zh": "目标行是 `output = lhs + rhs`。", "en": "The target line is `output = lhs + rhs`."},
        success={"zh": "通过。你写出了第一份 NineToothed CUDA kernel。", "en": "Cleared. You wrote your first NineToothed CUDA kernel."},
        starter_name="vector_add_kernel.py",
        starter=_vector_add_starter(),
        validator="vector_add",
    ),
    Quest(
        slug="make-row-sum",
        module="kernels",
        kind="code",
        title={"zh": "层级张量求行和", "en": "Row Sum with Hierarchical Tensors"},
        subtitle={"zh": "在 application 里循环 K 块。", "en": "Loop over K-blocks inside application."},
        concepts=("make", "application", "ntl.sum"),
        requires_cuda=True,
        lesson=(
            LessonBlock(
                {"zh": "这次不是一行完成", "en": "This one needs a loop"},
                {
                    "zh": "一行可能被切成多个块，所以 application 里要循环 `x.shape[1]`，每次把一个局部块的和累加到 `acc`。",
                    "en": "One row can be split into multiple blocks, so application loops over `x.shape[1]` and accumulates the sum of each local block into `acc`.",
                },
            ),
        ),
        question={"zh": "编辑生成的 `row_sum_kernel.py`。", "en": "Edit the generated `row_sum_kernel.py`."},
        hint={"zh": "循环里应该加 `ntl.sum(x[0, i], axis=-1)`。", "en": "Inside the loop, add `ntl.sum(x[0, i], axis=-1)`."},
        success={"zh": "通过。你已经能处理三层张量里的局部迭代。", "en": "Cleared. You can handle local iteration in a three-level tensor."},
        starter_name="row_sum_kernel.py",
        starter=_row_sum_starter(),
        validator="row_sum",
    ),
    Quest(
        slug="softmax-reduction",
        module="kernels",
        kind="code",
        title={"zh": "稳定 softmax", "en": "Stable Softmax"},
        subtitle={"zh": "max、exp、sum 三个归约组合起来。", "en": "Combine max, exp, and sum reductions."},
        concepts=("ntl.max", "ntl.exp", "ntl.sum", "softmax"),
        requires_cuda=True,
        lesson=(
            LessonBlock(
                {"zh": "数值稳定性", "en": "Numerical stability"},
                {
                    "zh": "softmax 先减去行最大值，再指数化，最后除以指数和。越界位置用 `float('-inf')`，这样不会影响最大值和分母。",
                    "en": "Softmax subtracts the row max, exponentiates, then divides by the exponential sum. Out-of-bounds positions use `float('-inf')`, so they do not affect the max or denominator.",
                },
            ),
        ),
        question={"zh": "编辑生成的 `softmax_kernel.py`。", "en": "Edit the generated `softmax_kernel.py`."},
        hint={"zh": "依次使用 `input_row - ntl.max(input_row)`、`ntl.exp(...)`、`ntl.sum(...)`。", "en": "Use `input_row - ntl.max(input_row)`, then `ntl.exp(...)`, then `ntl.sum(...)`."},
        success={"zh": "通过。你完成了第一个带归约的稳定 softmax kernel。", "en": "Cleared. You completed a stable softmax kernel with reductions."},
        starter_name="softmax_kernel.py",
        starter=_softmax_starter(),
        validator="softmax",
    ),
)

QUESTS_BY_SLUG = {quest.slug: quest for quest in QUESTS}
MODULES_BY_SLUG = {module.slug: module for module in MODULES}
