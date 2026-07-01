from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
try:
    from streamlit_ace import st_ace
except ImportError:  # pragma: no cover - optional UI enhancement.
    st_ace = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VISUAL_CACHE_DIR = (
    Path(os.environ["NTPUZZLES_VISUAL_DIR"])
    if os.environ.get("NTPUZZLES_VISUAL_DIR")
    else Path("/mnt/data/ntpuzzles-visuals")
    if Path("/mnt/data").exists()
    else ROOT / ".ntpuzzles_visuals"
)

from ntpuzzles.curriculum import MODULES, MODULES_BY_SLUG, QUESTS, QUESTS_BY_SLUG, text
from ntpuzzles.game import (
    ANSWER_UNLOCK_ATTEMPTS,
    debrief_text,
    is_unlocked,
    load_progress,
    mark_completed,
    next_quest,
    next_recommendation,
    prepare_solution,
    progress_line,
    quest_number,
    reset_progress,
    set_language,
    solution_explanation_text,
    solution_path,
    validate_solution,
)
from ntpuzzles.levels import PuzzleDependencyError


st.set_page_config(page_title="NineToothed Puzzles", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.4rem; max-width: 1280px; }
    .nt-hero {
        border-left: 5px solid #0f766e;
        padding: 1rem 1.2rem;
        background: #f8fafc;
        margin-bottom: 1rem;
    }
    .nt-muted { color: #475569; }
    .nt-pill {
        display: inline-block;
        padding: .2rem .55rem;
        margin-right: .35rem;
        border: 1px solid #cbd5e1;
        background: #ffffff;
        border-radius: 999px;
        font-size: .82rem;
    }
    .nt-locked { color: #64748b; }
    .nt-open { color: #0f766e; font-weight: 650; }
    .nt-done { color: #166534; font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _py_data(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_py_data(item) for item in value]
    if isinstance(value, list):
        return [_py_data(item) for item in value]
    return value


def _is_scalar(value) -> bool:
    return not isinstance(value, (list, tuple, dict))


def _render_matrix(label: str, value) -> None:
    data = _py_data(value)
    st.caption(label)
    if isinstance(data, list) and data and all(_is_scalar(item) for item in data):
        data = [data]
    if (
        isinstance(data, list)
        and data
        and all(isinstance(row, list) and all(_is_scalar(cell) for cell in row) for row in data)
    ):
        width = max(len(row) for row in data)
        rows = [{str(index): row[index] if index < len(row) else "" for index in range(width)} for row in data]
        st.table(rows)
        return
    st.json(data)


def _render_blocks(label: str, value, lang: str) -> None:
    data = _py_data(value)
    st.caption(label)
    if not (
        isinstance(data, list)
        and data
        and isinstance(data[0], list)
        and data[0]
        and isinstance(data[0][0], list)
    ):
        _render_matrix(label, data)
        return

    for row_index, row in enumerate(data):
        cols = st.columns(len(row))
        for col_index, block in enumerate(row):
            title = (
                f"外层块 ({row_index}, {col_index})"
                if lang == "zh"
                else f"Outer block ({row_index}, {col_index})"
            )
            with cols[col_index]:
                _render_matrix(title, block)


def _load_nt_visuals():
    try:
        from ninetoothed import Tensor
        import matplotlib

        matplotlib.use("Agg", force=True)
        from ninetoothed.visualization import visualize
    except Exception as exc:  # pragma: no cover - depends on optional runtime.
        return None, None, exc
    return Tensor, visualize, None


def _visual_path(quest_slug: str, name: str) -> Path:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in name)
    return VISUAL_CACHE_DIR / f"{quest_slug}-{safe}.png"


def _render_official_tensor_visual(visualize, quest_slug: str, name: str, label: str, tensor, color: str) -> None:
    path = _visual_path(quest_slug, name)
    VISUAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    visualize(tensor, color=color, save_path=path)
    st.image(str(path), caption=label, use_container_width=True)


def _render_official_tensor_gallery(visualize, quest_slug: str, items: list[tuple[str, str, object, str]]) -> None:
    if not items:
        return
    columns = st.columns(min(len(items), 3))
    for index, (name, label, tensor, color) in enumerate(items):
        with columns[index % len(columns)]:
            _render_official_tensor_visual(visualize, quest_slug, name, label, tensor, color)


def _visual_intro(lang: str) -> None:
    st.info(
        "这里优先调用 NineToothed 官方 `visualize(..., save_path=...)` 生成结构图；下方表格只作为索引值补充。"
        if lang == "zh"
        else "This lab uses NineToothed's official `visualize(..., save_path=...)` first; tables below are only numeric index supplements."
    )


def _vector_tile_visual(Tensor, visualize, quest, lang: str) -> None:
    length = st.slider(
        "向量长度" if lang == "zh" else "Vector length",
        min_value=3,
        max_value=16,
        value=10 if quest.slug in {"tile-padding", "block-size-symbol"} else 8,
        key=f"vis-{quest.slug}-length",
    )
    block = st.slider(
        "块大小 / block_size" if lang == "zh" else "Block size",
        min_value=2,
        max_value=8,
        value=4,
        key=f"vis-{quest.slug}-block",
    )
    x = Tensor(1)
    source = x.subs({x: Tensor(shape=(length,))})
    tiled = x.tile((block,)).subs({x: Tensor(shape=(length,))})
    observed = tiled.eval()
    st.code("x = Tensor(1)\nobserved = x.tile((block_size,)).eval({x: Tensor(shape=(length,))})", language="python")
    _render_official_tensor_gallery(
        visualize,
        quest.slug,
        [
            (f"source-{length}", "源 Tensor" if lang == "zh" else "Source Tensor", source, "#0f766e"),
            (f"tile-{length}-{block}", "tile 后的嵌套 Tensor" if lang == "zh" else "Nested Tensor after tile", tiled, "#2563eb"),
        ],
    )
    _render_matrix(
        "每一行是一个外层块；行内元素是该程序实例看到的局部索引，-1 表示越界。"
        if lang == "zh"
        else "Each row is one outer block; row elements are local source indices, and -1 marks padding.",
        observed,
    )
    if quest.slug == "program-local-block":
        program = st.slider(
            "选择程序实例" if lang == "zh" else "Choose program instance",
            min_value=0,
            max_value=max(0, len(_py_data(observed)) - 1),
            value=min(1, max(0, len(_py_data(observed)) - 1)),
            key=f"vis-{quest.slug}-program",
        )
        _render_matrix(
            "当前程序实例的局部块" if lang == "zh" else "Local block for the selected program",
            _py_data(observed)[program],
        )


def _matrix_tile_visual(Tensor, visualize, quest, lang: str) -> None:
    rows = st.slider("矩阵行数" if lang == "zh" else "Matrix rows", 2, 8, 4, key=f"vis-{quest.slug}-rows")
    cols = st.slider("矩阵列数" if lang == "zh" else "Matrix cols", 2, 10, 6, key=f"vis-{quest.slug}-cols")
    tile_r = st.slider("内层块行数" if lang == "zh" else "Inner block rows", 1, 4, 2, key=f"vis-{quest.slug}-tile-r")
    tile_c = st.slider("内层块列数" if lang == "zh" else "Inner block cols", 1, 5, 3, key=f"vis-{quest.slug}-tile-c")
    x = Tensor(shape=(rows, cols))
    tiled = x.tile((tile_r, tile_c))
    st.code("x = Tensor(shape=(rows, cols))\ntiled = x.tile((tile_r, tile_c))", language="python")
    st.write(
        f"{'外层块网格' if lang == 'zh' else 'Outer block grid'}: `{tuple(tiled.shape)}` · "
        f"{'内层局部块' if lang == 'zh' else 'Inner local block'}: `{tuple(tiled.dtype.shape)}`"
    )
    if quest.slug == "flatten-inner":
        flattened = tiled
        flattened.dtype = tiled.dtype.flatten()
        st.write(
            f"{'flatten 后局部块形状' if lang == 'zh' else 'Local block shape after flatten'}: "
            f"`{tuple(flattened.dtype.shape)}`"
        )
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [
                (f"matrix-{rows}-{cols}", "源 Tensor" if lang == "zh" else "Source Tensor", x, "#0f766e"),
                (f"tile-{rows}-{cols}-{tile_r}-{tile_c}", "tile 后" if lang == "zh" else "After tile", tiled, "#2563eb"),
                (f"flatten-{rows}-{cols}-{tile_r}-{tile_c}", "flatten 内层后" if lang == "zh" else "After flattening inner block", flattened, "#9333ea"),
            ],
        )
    else:
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [
                (f"matrix-{rows}-{cols}", "源 Tensor" if lang == "zh" else "Source Tensor", x, "#0f766e"),
                (f"tile-{rows}-{cols}-{tile_r}-{tile_c}", "tile 后的嵌套 Tensor" if lang == "zh" else "Nested Tensor after tile", tiled, "#2563eb"),
            ],
        )
    _render_blocks(
        "外层块网格中的每个小表都是一个局部块。" if lang == "zh" else "Each small table in the outer grid is one local block.",
        tiled.eval(),
        lang,
    )


def _meta_visual(Tensor, visualize, quest, lang: str) -> bool:
    slug = quest.slug
    if slug in {"tensor-rank", "subs-shape"}:
        rows = st.slider("行数" if lang == "zh" else "Rows", 1, 6, 2, key=f"vis-{slug}-rows")
        cols = st.slider("列数" if lang == "zh" else "Cols", 1, 8, 3, key=f"vis-{slug}-cols")
        x = Tensor(2)
        concrete = x.subs({x: Tensor(shape=(rows, cols))})
        st.code("x = Tensor(2)\nconcrete = x.subs({x: Tensor(shape=(rows, cols))})", language="python")
        st.write(f"`Tensor(2)` rank = `{len(x.shape)}`; concrete shape = `{tuple(concrete.shape)}`")
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [(f"tensor2-{rows}-{cols}", "具体化后的 Tensor" if lang == "zh" else "Concrete Tensor", concrete, "#0f766e")],
        )
        _render_matrix("eval 后看到的是源索引地图" if lang == "zh" else "eval reveals the source-index map", concrete.eval())
        return True
    if slug == "eval-vector":
        length = st.slider("向量长度" if lang == "zh" else "Vector length", 2, 16, 5, key=f"vis-{slug}-length")
        x = Tensor(1)
        concrete = x.subs({x: Tensor(shape=(length,))})
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [(f"vector-{length}", "具体化后的一维 Tensor" if lang == "zh" else "Concrete 1D Tensor", concrete, "#0f766e")],
        )
        _render_matrix("一维 index map" if lang == "zh" else "1D index map", concrete.eval())
        return True
    if slug in {
        "tile-even",
        "tile-padding",
        "block-size-symbol",
        "arrange-three-vectors",
        "program-local-block",
        "application-block",
    }:
        _vector_tile_visual(Tensor, visualize, quest, lang)
        return True
    if slug in {"tile-2d-shape", "flatten-inner", "visualize-ready", "matmul-output-grid"}:
        _matrix_tile_visual(Tensor, visualize, quest, lang)
        return True
    if slug == "unsqueeze-expand":
        length = st.slider("原向量长度" if lang == "zh" else "Original vector length", 2, 8, 4, key=f"vis-{slug}-length")
        rows = st.slider("广播后的行数" if lang == "zh" else "Expanded rows", 2, 6, 3, key=f"vis-{slug}-rows")
        x = Tensor(1)
        arranged = x.unsqueeze(0).expand((rows, -1))
        concrete_source = x.subs({x: Tensor(shape=(length,))})
        concrete_arranged = arranged.subs({x: Tensor(shape=(length,))})
        st.code("arranged = x.unsqueeze(0).expand((rows, -1))", language="python")
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [
                (f"source-{length}", "源 Tensor" if lang == "zh" else "Source Tensor", concrete_source, "#0f766e"),
                (f"expanded-{length}-{rows}", "unsqueeze + expand 后" if lang == "zh" else "After unsqueeze + expand", concrete_arranged, "#2563eb"),
            ],
        )
        _render_matrix("广播后的 index map" if lang == "zh" else "Expanded index map", concrete_arranged.eval())
        return True
    if slug == "squeeze-axis":
        x = Tensor(shape=(1, 4))
        y = x.squeeze(0)
        st.code("y = x.squeeze(0)", language="python")
        st.write(f"`{tuple(x.shape)}` -> `{tuple(y.shape)}`")
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [
                ("squeeze-source", "源 Tensor" if lang == "zh" else "Source Tensor", x, "#0f766e"),
                ("squeeze-result", "squeeze 后" if lang == "zh" else "After squeeze", y, "#2563eb"),
            ],
        )
        _render_matrix("squeeze 后的 index map" if lang == "zh" else "Index map after squeeze", y.eval())
        return True
    if slug == "ravel-observe":
        x = Tensor(shape=(2, 3))
        st.code("y = x.ravel()", language="python")
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [
                ("ravel-source", "源 Tensor" if lang == "zh" else "Source Tensor", x, "#0f766e"),
                ("ravel-result", "ravel 后" if lang == "zh" else "After ravel", x.ravel(), "#2563eb"),
            ],
        )
        _render_matrix("原始 2D index map" if lang == "zh" else "Original 2D index map", x.eval())
        _render_matrix("ravel 后的一维读取顺序" if lang == "zh" else "1D read order after ravel", x.ravel().eval())
        return True
    if slug == "pad-border":
        x = Tensor(shape=(2, 2))
        padded = x.pad(((1, 0), (0, 1)))
        st.code("padded = x.pad(((1, 0), (0, 1)))", language="python")
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [
                ("pad-source", "源 Tensor" if lang == "zh" else "Source Tensor", x, "#0f766e"),
                ("pad-result", "pad 后" if lang == "zh" else "After pad", padded, "#2563eb"),
            ],
        )
        _render_matrix("pad 后 -1 是补出来的边界" if lang == "zh" else "-1 marks padded boundary cells", padded.eval())
        return True
    if slug == "slice-index":
        x = Tensor(shape=(3, 4, 2))
        sliced = x[:, 1, :]
        st.code("sliced = x[:, 1, :]", language="python")
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [
                ("slice-source", "源 Tensor" if lang == "zh" else "Source Tensor", x, "#0f766e"),
                ("slice-result", "切片后" if lang == "zh" else "After slicing", sliced, "#2563eb"),
            ],
        )
        _render_matrix("切片后的 index map" if lang == "zh" else "Index map after slicing", sliced.eval())
        return True
    if slug == "permute-map":
        x = Tensor(shape=(2, 3))
        transposed = x.permute((1, 0))
        st.code("transposed = x.permute((1, 0))", language="python")
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [
                ("permute-source", "源 Tensor" if lang == "zh" else "Source Tensor", x, "#0f766e"),
                ("permute-result", "permute 后" if lang == "zh" else "After permute", transposed, "#2563eb"),
            ],
        )
        _render_matrix("原始 index map" if lang == "zh" else "Original index map", x.eval())
        _render_matrix("转置后的 index map" if lang == "zh" else "Permuted index map", transposed.eval())
        return True
    if slug == "outer-shape-rule":
        x = Tensor(shape=(1, 2))
        expanded = x.expand((3, -1))
        st.code("expanded = x.expand((3, -1))", language="python")
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [
                ("outer-source", "源 Tensor" if lang == "zh" else "Source Tensor", x, "#0f766e"),
                ("outer-expanded", "expand 后" if lang == "zh" else "After expand", expanded, "#2563eb"),
            ],
        )
        _render_matrix("对齐后的外层视图" if lang == "zh" else "Aligned outer view", expanded.eval())
        return True
    return False


def render_visual_lab(quest, lang: str) -> None:
    _visual_intro(lang)
    Tensor, visualize, error = _load_nt_visuals()
    if error is not None:
        st.warning(
            f"当前环境还没有安装 ninetoothed，无法运行真实 Tensor 可视化：{error}"
            if lang == "zh"
            else f"ninetoothed is not installed in this environment, so live Tensor visuals are unavailable: {error}"
        )
        return
    if _meta_visual(Tensor, visualize, quest, lang):
        return
    if quest.slug == "matmul-k-blocks":
        k = st.slider("K 维长度" if lang == "zh" else "K length", 2, 16, 6, key=f"vis-{quest.slug}-k")
        block = st.slider("BLOCK_SIZE_K", 1, 8, 3, key=f"vis-{quest.slug}-block")
        count = (k + block - 1) // block
        st.write(f"{'K 方向循环次数' if lang == 'zh' else 'K-loop iterations'}: `{count}`")
        x = Tensor(1)
        tiled = x.tile((block,)).subs({x: Tensor(shape=(k,))})
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [(f"kblocks-{k}-{block}", "K 方向 tile 后的块" if lang == "zh" else "K dimension after tiling", tiled, "#2563eb")],
        )
        _render_matrix("每个循环处理的 K 索引" if lang == "zh" else "K indices handled by each loop", [list(range(i * block, min((i + 1) * block, k))) for i in range(count)])
        return
    if quest.module in {"apply", "tools", "kernels"}:
        st.code(
            "\n".join(
                [
                    "arrangement(...) -> local blocks",
                    "application(local_x, local_y, ...) -> ntl operations",
                    "write local result back through the same index map",
                ]
            ),
            language="text",
        )
        st.write(
            "这些题的重点不是 torch 公式，而是 application 只处理排布后的局部块；torch 只在最后作为结果 oracle。"
            if lang == "zh"
            else "The point here is not a torch formula: application code only sees arranged local blocks; torch is only the final oracle."
        )
        x = Tensor(1)
        local_blocks = x.tile((4,)).subs({x: Tensor(shape=(8,))})
        _render_official_tensor_gallery(
            visualize,
            quest.slug,
            [(f"local-blocks-{quest.slug}", "application 收到的局部块结构" if lang == "zh" else "Local block structure seen by application", local_blocks, "#2563eb")],
        )
        if quest.slug in {"ntl-zeros-accumulator", "ntl-full-sentinel", "reduction-sum", "softmax-reduction"}:
            _render_matrix("局部块索引补充" if lang == "zh" else "Local block index supplement", local_blocks.eval())
        return
    st.write(
        "这一关没有单独的可视化实验；学习页和开始做题区域已经覆盖它的核心。"
        if lang == "zh"
        else "This level has no separate visual lab; the lesson and coding area cover its core."
    )


def select_level(slug: str, module_slug: str) -> None:
    st.session_state["selected_module"] = module_slug
    st.session_state["selected_slug"] = slug


def code_editor(source: str, key: str, lang: str) -> str:
    if st_ace is not None:
        edited = st_ace(
            value=source,
            language="python",
            theme="github",
            key=key,
            height=460,
            font_size=14,
            tab_size=4,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            auto_update=True,
        )
        return source if edited is None else edited

    st.warning(
        "当前环境缺少高亮编辑器组件 `streamlit-ace`，已降级为文本框。"
        if lang == "zh"
        else "The highlighted editor component `streamlit-ace` is missing, so this falls back to a text area."
    )
    edited = st.text_area(
        "解题代码" if lang == "zh" else "Solution code",
        value=source,
        height=360,
        key=key,
    )
    with st.expander("高亮预览" if lang == "zh" else "Highlighted preview", expanded=True):
        st.code(edited, language="python")
    return edited


def render_practice_panel(quest, completed: set[str], lang: str) -> None:
    st.markdown("### 开始做题" if lang == "zh" else "### Start Coding")
    if not is_unlocked(quest.slug, completed) and quest.slug not in completed:
        st.warning("请先完成前面的关卡。" if lang == "zh" else "Clear previous levels first.")
        return

    attempts_key = f"attempts-{quest.slug}"
    st.session_state.setdefault(attempts_key, 0)
    path = prepare_solution(quest)
    source = path.read_text(encoding="utf-8")

    st.markdown(f"**{'题目' if lang == 'zh' else 'Task'}**: {text(quest.question, lang)}")
    st.caption(str(path))
    edited = code_editor(source, f"editor-{quest.slug}", lang)

    col_save, col_check = st.columns([0.28, 0.72])
    with col_save:
        if st.button("保存" if lang == "zh" else "Save", use_container_width=True):
            path.write_text(edited, encoding="utf-8")
            st.success("已保存。" if lang == "zh" else "Saved.")
    with col_check:
        if st.button("判题" if lang == "zh" else "Check solution", type="primary", use_container_width=True):
            path.write_text(edited, encoding="utf-8")
            try:
                validate_solution(quest)
            except PuzzleDependencyError as exc:
                st.warning(str(exc))
            except Exception as exc:  # pragma: no cover - UI feedback path.
                st.session_state[attempts_key] += 1
                st.error(str(exc))
                if st.session_state[attempts_key] >= ANSWER_UNLOCK_ATTEMPTS:
                    st.warning(
                        f"已经尝试 {st.session_state[attempts_key]} 次，可以查看完整答案解析。"
                        if lang == "zh"
                        else f"{st.session_state[attempts_key]} attempts so far. You can reveal the full answer explanation."
                    )
            else:
                st.session_state[attempts_key] = 0
                mark_completed(quest.slug)
                st.success(text(quest.success, lang))
                st.rerun()

    with st.expander("提示" if lang == "zh" else "Hint", expanded=False):
        st.info(text(quest.hint, lang))
    if st.session_state[attempts_key] >= ANSWER_UNLOCK_ATTEMPTS:
        with st.expander("完整答案解析" if lang == "zh" else "Full answer explanation", expanded=False):
            st.markdown(solution_explanation_text(quest, lang))


state_lang = st.session_state.get("lang", "zh")
with st.sidebar:
    lang = st.radio("Language / 语言", ("zh", "en"), index=0 if state_lang == "zh" else 1, horizontal=True)
    st.session_state["lang"] = lang
    set_language(lang)
    completed = load_progress()
    st.progress(len(completed) / len(QUESTS), text=progress_line(completed, lang))
    if st.button("Reset progress" if lang == "en" else "重置进度"):
        reset_progress()
        st.rerun()

    st.subheader("Chapters" if lang == "en" else "章节")
    module_options = [module.slug for module in MODULES]
    if st.session_state.get("selected_module") not in module_options:
        st.session_state["selected_module"] = module_options[0]
    selected_module = st.radio(
        "chapter",
        module_options,
        format_func=lambda slug: text(MODULES_BY_SLUG[slug].title, lang),
        label_visibility="collapsed",
        key="selected_module",
    )

completed = load_progress()
module = MODULES_BY_SLUG[selected_module]
recommended = next_quest(completed)

st.markdown(
    f"""
    <div class="nt-hero">
      <h1 style="margin:0;">NineToothed Puzzles</h1>
      <div class="nt-muted">{text(module.title, lang)} · {text(module.summary, lang)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

module_quests = [quest for quest in QUESTS if quest.module == selected_module]
module_done = [quest for quest in module_quests if quest.slug in completed]
stats = st.columns([0.22, 0.22, 0.34, 0.22])
stats[0].metric("Total" if lang == "en" else "总进度", f"{len(completed)}/{len(QUESTS)}")
stats[1].metric("Chapter" if lang == "en" else "本章", f"{len(module_done)}/{len(module_quests)}")
stats[2].metric("Recommended" if lang == "en" else "推荐下一关", next_recommendation(completed, lang).replace("下一关：", "").replace("Next level: ", ""))
stats[3].metric("Mode" if lang == "en" else "模式", "CUDA" if recommended and recommended.requires_cuda else "CPU")
if recommended:
    st.button(
        "Open recommended level" if lang == "en" else "进入推荐关卡",
        type="primary",
        on_click=select_level,
        args=(recommended.slug, recommended.module),
    )

left, right = st.columns([0.34, 0.66], gap="large")

with left:
    st.subheader("Level Map" if lang == "en" else "关卡地图")
    selected_slug = st.session_state.get("selected_slug")
    if selected_slug not in [quest.slug for quest in module_quests]:
        selected_slug = recommended.slug if recommended and recommended.module == selected_module else module_quests[0].slug
        st.session_state["selected_slug"] = selected_slug

    for quest in module_quests:
        status_class = "nt-done" if quest.slug in completed else "nt-open" if is_unlocked(quest.slug, completed) else "nt-locked"
        status_text = (
            ("cleared" if lang == "en" else "已通关")
            if quest.slug in completed
            else ("open" if lang == "en" else "可挑战")
            if is_unlocked(quest.slug, completed)
            else ("locked" if lang == "en" else "锁定")
        )
        if st.button(
            f"{status_text} · {text(quest.title, lang)}",
            key=f"select-{quest.slug}",
            use_container_width=True,
            disabled=not is_unlocked(quest.slug, completed) and quest.slug not in completed,
        ):
            selected_slug = quest.slug
            st.session_state["selected_slug"] = quest.slug
        st.markdown(
            f"<div class='{status_class}' style='font-size:.84rem; margin:-.4rem 0 .45rem .25rem;'>{quest.slug}</div>",
            unsafe_allow_html=True,
        )

quest = QUESTS_BY_SLUG[selected_slug]

with right:
    status = "CUDA" if quest.requires_cuda else "CPU"
    mode = "Code" if quest.kind == "code" else "Fill"
    if lang == "zh":
        mode = "代码" if quest.kind == "code" else "填空"

    st.subheader(text(quest.title, lang))
    st.caption(text(quest.subtitle, lang))
    st.markdown(
        "".join(f"<span class='nt-pill'>{item}</span>" for item in (*quest.concepts, mode, status)),
        unsafe_allow_html=True,
    )

    render_practice_panel(quest, completed, lang)
    st.divider()

    tabs = st.tabs(
        ["Lesson", "Visual Lab", "Debrief", "Reference"]
        if lang == "en"
        else ["学习", "可视化", "复盘", "参考"]
    )

    with tabs[0]:
        for block in quest.lesson:
            st.markdown(f"#### {text(block.title, lang)}")
            st.write(text(block.body, lang))
            if block.code:
                st.code(block.code, language="python")

    with tabs[1]:
        render_visual_lab(quest, lang)

    with tabs[2]:
        if quest.slug in completed:
            st.markdown(debrief_text(quest, lang))
            st.code(str(solution_path(quest)), language="text")
        else:
            st.info(
                "Clear this level to unlock the debrief for the code you wrote."
                if lang == "en"
                else "通关后会解锁本关复盘，用来对照你刚刚写下的代码。"
            )

    with tabs[3]:
        st.markdown("#### CLI" if lang == "en" else "#### 命令行")
        st.code(f"python puzzle_runner.py play {quest.slug} --lang {lang}", language="bash")
        st.markdown("#### Level" if lang == "en" else "#### 关卡信息")
        st.write(f"{quest_number(quest)}/{len(QUESTS)} · {quest.slug}")
        if quest.slug in completed:
            st.markdown("#### Reference output" if lang == "en" else "#### 参考输出")
            if isinstance(quest.expected, (list, tuple, dict)):
                st.json(quest.expected)
            else:
                st.code(repr(quest.expected), language="python")
            st.markdown("#### Full answer explanation" if lang == "en" else "#### 完整答案解析")
            st.markdown(solution_explanation_text(quest, lang))
        else:
            st.info(
                "Reference output appears after the level is cleared."
                if lang == "en"
                else "参考输出会在通关后显示，避免练习变成抄答案。"
            )
        st.markdown("#### Course position" if lang == "en" else "#### 课程位置")
        st.write(text(MODULES_BY_SLUG[quest.module].title, lang))
        st.write(text(MODULES_BY_SLUG[quest.module].summary, lang))
