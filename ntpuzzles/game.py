from __future__ import annotations

import ast
from contextlib import nullcontext
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from .curriculum import MODULES, MODULES_BY_SLUG, QUESTS, QUESTS_BY_SLUG, Quest, text
from .levels import PUZZLES, PUZZLES_BY_SLUG, PuzzleDependencyError

try:  # Rich is part of the package dependencies, but keep a plain fallback.
    from rich.console import Console
    from rich.columns import Columns
    from rich.layout import Layout
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
except ImportError:  # pragma: no cover - fallback for unusual minimal installs.
    Columns = None
    Console = None
    Layout = None
    Markdown = None
    Panel = None
    Prompt = None
    Syntax = None
    Table = None
    Text = None

try:  # prompt_toolkit gives normal shell-like arrow key behavior.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import InMemoryHistory
except ImportError:  # pragma: no cover - optional graceful fallback.
    AutoSuggestFromHistory = None
    InMemoryHistory = None
    PromptSession = None


ROOT = Path(__file__).resolve().parents[1]
PROGRESS_PATH = ROOT / ".ntpuzzles_progress.json"
SOLUTIONS_DIR = ROOT / "solutions"
DEFAULT_LANG = "zh"
SUPPORTED_LANGS = ("zh", "en")
VALIDATOR_BY_QUEST = {
    "jit-vector-add": "vector-add-kernel",
    "make-row-sum": "row-sum-kernel",
    "softmax-reduction": "softmax-kernel",
}
EDIT_START = "# ====================== EDIT BELOW ======================"
EDIT_END = "# ====================== EDIT ABOVE ======================"
ANSWER_UNLOCK_ATTEMPTS = 2

console = Console() if Console else None
_prompt_session = None
_ansi_sequence = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _print(message: str = "") -> None:
    if console:
        console.print(message)
    else:
        print(message)


def _clean_prompt_value(raw: str, default: str | None = None) -> str | None:
    cleaned = _ansi_sequence.sub("", raw).strip()
    if raw.strip() and cleaned == "" and "\x1b[" in raw:
        return None
    return cleaned or (default or "")


def _interactive_stdin() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _get_prompt_session():
    global _prompt_session
    if not (PromptSession and InMemoryHistory and AutoSuggestFromHistory):
        return None
    if not _interactive_stdin():
        return None
    if _prompt_session is None:
        _prompt_session = PromptSession(history=InMemoryHistory(), auto_suggest=AutoSuggestFromHistory())
    return _prompt_session


def _prompt(message: str, default: str | None = None) -> str:
    while True:
        try:
            session = _get_prompt_session()
            if session:
                raw = session.prompt(f"{message}: ", default=default or "")
                cleaned = _clean_prompt_value(raw, default)
            elif Prompt:
                cleaned = _clean_prompt_value(Prompt.ask(message, default=default), default)
            else:
                raw = input(f"{message}{f' [{default}]' if default else ''}: ")
                cleaned = _clean_prompt_value(raw, default)
            if cleaned is not None:
                return cleaned
        except EOFError:
            return "quit"


def terminal_width() -> int:
    if console:
        return console.width
    return shutil.get_terminal_size((100, 24)).columns


def terminal_height() -> int:
    if console:
        return console.height
    return shutil.get_terminal_size((100, 24)).lines


def normalize_lang(lang: str | None) -> str:
    if not lang:
        return DEFAULT_LANG
    lang = lang.lower()
    return lang if lang in SUPPORTED_LANGS else DEFAULT_LANG


def load_state() -> dict:
    if not PROGRESS_PATH.exists():
        return {"completed": [], "language": DEFAULT_LANG, "editor": ""}
    try:
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"completed": [], "language": DEFAULT_LANG, "editor": ""}
    completed = [slug for slug in data.get("completed", []) if slug in QUESTS_BY_SLUG]
    editor = str(data.get("editor") or "").strip()
    return {"completed": completed, "language": normalize_lang(data.get("language")), "editor": editor}


def save_state(state: dict) -> None:
    completed = set(state.get("completed", []))
    ordered = [quest.slug for quest in QUESTS if quest.slug in completed]
    data = {
        "completed": ordered,
        "language": normalize_lang(state.get("language")),
        "editor": str(state.get("editor") or "").strip(),
    }
    PROGRESS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_progress() -> set[str]:
    return set(load_state()["completed"])


def save_progress(completed: set[str]) -> None:
    state = load_state()
    state["completed"] = [quest.slug for quest in QUESTS if quest.slug in completed]
    save_state(state)


def reset_progress() -> None:
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()


def set_language(lang: str) -> None:
    state = load_state()
    state["language"] = normalize_lang(lang)
    save_state(state)


def get_language(lang: str | None = None) -> str:
    return normalize_lang(lang or load_state()["language"])


def set_editor(editor: str | None) -> None:
    state = load_state()
    editor = (editor or "").strip()
    if editor.lower() in {"auto", "default", "none", "clear"}:
        editor = ""
    state["editor"] = editor
    save_state(state)


def get_editor() -> str:
    return str(load_state().get("editor") or "").strip()


def quest_index(slug: str) -> int:
    for index, quest in enumerate(QUESTS):
        if quest.slug == slug:
            return index
    raise KeyError(slug)


def is_unlocked(slug: str, completed: set[str] | None = None) -> bool:
    completed = load_progress() if completed is None else completed
    index = quest_index(slug)
    if index == 0:
        return True
    return QUESTS[index - 1].slug in completed


def next_quest(completed: set[str] | None = None) -> Quest | None:
    completed = load_progress() if completed is None else completed
    for quest in QUESTS:
        if quest.slug not in completed:
            return quest
    return None


def progress_line(completed: set[str] | None = None, lang: str | None = None) -> str:
    completed = load_progress() if completed is None else completed
    lang = get_language(lang)
    if lang == "zh":
        return f"已通关 {len(completed)}/{len(QUESTS)}"
    return f"{len(completed)}/{len(QUESTS)} levels cleared"


def quest_number(quest: Quest) -> int:
    return quest_index(quest.slug) + 1


def next_after(quest: Quest) -> Quest | None:
    index = quest_index(quest.slug)
    if index + 1 >= len(QUESTS):
        return None
    return QUESTS[index + 1]


def next_recommendation(completed: set[str] | None = None, lang: str | None = None) -> str:
    completed = load_progress() if completed is None else completed
    lang = get_language(lang)
    quest = next_quest(completed)
    if quest is None:
        return "全部关卡已通关。" if lang == "zh" else "Every level is cleared."
    if lang == "zh":
        return f"下一关：{quest.slug} · {text(quest.title, lang)}"
    return f"Next level: {quest.slug} · {text(quest.title, lang)}"


def debrief_text(quest: Quest, lang: str | None = None) -> str:
    lang = get_language(lang)
    concepts = ", ".join(f"`{item}`" for item in quest.concepts)
    number = quest_number(quest)
    following = next_after(quest)
    solution = f"`{solution_path(quest).relative_to(ROOT)}`"
    mode = "代码" if quest.kind == "code" else "填空"
    if lang == "zh":
        lines = [
            f"### 通关复盘：第 {number}/{len(QUESTS)} 关",
            "",
            f"你刚刚完成的是 **{text(quest.title, lang)}**。这不是口头问答：判题器读取 {solution}，导入你的代码，并检查它是否真的算出了目标结果。",
            "",
            f"- 核心知识点：{concepts}",
            f"- 本关价值：把 `{mode}` 关卡里的空白补成可执行程序，逼自己用 NineToothed 对象、shape 或 kernel 语义来得到答案。",
            f"- 复查方式：回到 {solution}，把每个 `___` 替换成的表达式和课程片段对照，看它改变的是 shape、index map、local block，还是 language op。",
        ]
        if following:
            lines.append(f"- 下一步：`{following.slug}` 会继续训练 **{text(following.title, lang)}**。")
        else:
            lines.append("- 下一步：你已经完成全部关卡，可以开始把这些模式迁移到自己的 kernel。")
        return "\n".join(lines)

    mode = "code" if quest.kind == "code" else "fill-in"
    lines = [
        f"### Clear Debrief: Level {number}/{len(QUESTS)}",
        "",
        f"You just cleared **{text(quest.title, lang)}**. This was not a verbal quiz: the judge loaded {solution}, imported your code, and checked that it computed the required result.",
        "",
        f"- Concepts: {concepts}",
        f"- Why it matters: the `{mode}` level forces the blank to become executable NineToothed code, so the answer comes from tensor, shape, local-block, or kernel semantics.",
        f"- Review move: reopen {solution} and connect every filled `___` with the lesson snippet. Ask whether it changed shape, index mapping, local block data, or a language op.",
    ]
    if following:
        lines.append(f"- Next: `{following.slug}` continues with **{text(following.title, lang)}**.")
    else:
        lines.append("- Next: all levels are cleared; start moving these patterns into your own kernels.")
    return "\n".join(lines)


def _status_label(quest: Quest, completed: set[str], lang: str) -> str:
    if quest.slug in completed:
        return "已通关" if lang == "zh" else "cleared"
    if is_unlocked(quest.slug, completed):
        if next_quest(completed) and next_quest(completed).slug == quest.slug:
            return "当前" if lang == "zh" else "current"
        return "可挑战" if lang == "zh" else "open"
    return "锁定" if lang == "zh" else "locked"


def _module_number(module_slug: str) -> int:
    for index, module in enumerate(MODULES, start=1):
        if module.slug == module_slug:
            return index
    return 0


def _module_short_label(module_slug: str, lang: str) -> str:
    number = _module_number(module_slug)
    return f"第{number}章" if lang == "zh" else f"C{number}"


def _quest_mode_label(quest: Quest, lang: str) -> str:
    mode = "代码" if quest.kind == "code" else "填空"
    if lang == "en":
        mode = "Code" if quest.kind == "code" else "Fill"
    if quest.requires_cuda:
        mode += " / CUDA"
    return mode


def _resolve_map_style(style: str, context: str) -> str:
    if style != "auto":
        return style
    width = terminal_width()
    if width < 72:
        return "compact"
    threshold = 116 if context == "tui" else 104
    if width < threshold:
        return "medium"
    return "full"


def _map_title(lang: str, module_slug: str | None = None) -> str:
    if module_slug is None:
        return "NineToothed Puzzles"
    module = MODULES_BY_SLUG[module_slug]
    return f"{_module_short_label(module_slug, lang)} · {text(module.title, lang)}"


def _print_compact_map(quests: list[Quest], completed: set[str], lang: str, module_slug: str | None) -> None:
    _print(_map_title(lang, module_slug))
    current_module = module_slug or ""
    width = terminal_width()
    for quest in quests:
        if quest.module != current_module:
            current_module = quest.module
            module = MODULES_BY_SLUG[current_module]
            _print()
            _print(f"{_module_short_label(current_module, lang)} {text(module.title, lang)}")
        status = _status_label(quest, completed, lang)
        line = f"[{status}] {quest_number(quest):02d} {quest.slug}"
        if width >= 64:
            line += f" · {text(quest.title, lang)}"
        if quest.requires_cuda:
            line += " · CUDA"
        _print(line)


def module_progress_line(module_slug: str, completed: set[str], lang: str) -> str:
    module_quests = [quest for quest in QUESTS if quest.module == module_slug]
    done = [quest for quest in module_quests if quest.slug in completed]
    if lang == "zh":
        return f"{_module_short_label(module_slug, lang)} 进度 {len(done)}/{len(module_quests)}"
    return f"{_module_short_label(module_slug, lang)} progress {len(done)}/{len(module_quests)}"


def print_course_overview(completed: set[str] | None = None, lang: str | None = None) -> None:
    completed = load_progress() if completed is None else completed
    lang = get_language(lang)
    recommended = next_quest(completed)
    if Table and console and terminal_width() >= 72:
        table = Table(title="Course Overview" if lang == "en" else "课程总览", expand=True)
        table.add_column("Chapter" if lang == "en" else "章节", no_wrap=True)
        table.add_column("Progress" if lang == "en" else "进度", no_wrap=True)
        table.add_column("Next" if lang == "en" else "下一关")
        table.add_column("Status" if lang == "en" else "状态", no_wrap=True)
        for module in MODULES:
            quests = [quest for quest in QUESTS if quest.module == module.slug]
            done = [quest for quest in quests if quest.slug in completed]
            next_in_module = next((quest for quest in quests if quest.slug not in completed), None)
            if next_in_module is None:
                next_label = "完成" if lang == "zh" else "complete"
                status = "已完成" if lang == "zh" else "done"
            else:
                next_label = f"{quest_number(next_in_module):02d} {next_in_module.slug}"
                status = "当前" if recommended and recommended.module == module.slug else "待解锁"
                if lang == "en":
                    status = "current" if recommended and recommended.module == module.slug else "locked"
            table.add_row(
                f"{_module_short_label(module.slug, lang)} · {text(module.title, lang)}",
                f"{len(done)}/{len(quests)}",
                next_label,
                status,
            )
        console.print(table)
        return

    _print("课程总览" if lang == "zh" else "Course Overview")
    for module in MODULES:
        quests = [quest for quest in QUESTS if quest.module == module.slug]
        done = [quest for quest in quests if quest.slug in completed]
        next_in_module = next((quest for quest in quests if quest.slug not in completed), None)
        next_label = "完成" if next_in_module is None and lang == "zh" else "complete" if next_in_module is None else f"{quest_number(next_in_module):02d} {next_in_module.slug}"
        _print(f"{_module_short_label(module.slug, lang)} {len(done)}/{len(quests)} · {next_label}")


def print_map(
    completed: set[str] | None = None,
    lang: str | None = None,
    *,
    style: str = "auto",
    context: str = "cli",
    module_slug: str | None = None,
) -> None:
    completed = load_progress() if completed is None else completed
    lang = get_language(lang)
    quests = [quest for quest in QUESTS if module_slug is None or quest.module == module_slug]
    resolved_style = _resolve_map_style(style, context)

    if resolved_style == "compact" or not (Table and console):
        _print_compact_map(quests, completed, lang, module_slug)
        return

    if Table and console:
        table = Table(title=_map_title(lang, module_slug), expand=True)
        table.add_column("Status" if lang == "en" else "状态", style="bold")
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("Level" if lang == "en" else "关卡")
        table.add_column("Ch" if resolved_style == "medium" else ("Chapter" if lang == "en" else "章节"), no_wrap=resolved_style == "medium")
        table.add_column("Mode" if lang == "en" else "类型", no_wrap=True)
        for quest in quests:
            module = MODULES_BY_SLUG[quest.module]
            chapter = _module_short_label(quest.module, lang) if resolved_style == "medium" else text(module.title, lang)
            table.add_row(
                _status_label(quest, completed, lang),
                str(quest_number(quest)),
                f"{quest.slug}\n{text(quest.title, lang)}",
                chapter,
                _quest_mode_label(quest, lang),
            )
        console.print(table)
        return


def print_dashboard(completed: set[str] | None = None, lang: str | None = None) -> None:
    completed = load_progress() if completed is None else completed
    lang = get_language(lang)
    recommended = next_quest(completed)
    lines = [progress_line(completed, lang), next_recommendation(completed, lang)]
    if recommended:
        lines.append(module_progress_line(recommended.module, completed, lang))
    if Panel and console:
        console.print(Panel.fit("\n".join(lines), title="Dashboard" if lang == "en" else "学习面板", border_style="cyan"))
    else:
        for line in lines:
            _print(line)

    if recommended is None:
        print_course_overview(completed, lang)
        return
    print_map(completed, lang, style="compact", context="tui", module_slug=recommended.module)
    _print()
    _print("输入 map 查看章节总览，map all 查看全部关卡。" if lang == "zh" else "Type map for chapter overview, or map all for every level.")


def _chapter_table(completed: set[str], lang: str, module_slug: str) -> Table | str:
    quests = [quest for quest in QUESTS if quest.module == module_slug]
    if not (Table and console):
        return "\n".join(
            f"[{_status_label(quest, completed, lang)}] {quest_number(quest):02d} {quest.slug} · {text(quest.title, lang)}"
            for quest in quests
        )
    table = Table(expand=True, show_lines=False)
    table.add_column("#", justify="right", no_wrap=True, width=3)
    table.add_column("状态" if lang == "zh" else "Status", no_wrap=True, width=8)
    table.add_column("关卡" if lang == "zh" else "Level")
    table.add_column("类型" if lang == "zh" else "Mode", no_wrap=True, width=12)
    for quest in quests:
        table.add_row(
            str(quest_number(quest)),
            _status_label(quest, completed, lang),
            f"{quest.slug}\n{text(quest.title, lang)}",
            _quest_mode_label(quest, lang),
        )
    return table


def render_home_screen(completed: set[str] | None = None, lang: str | None = None, message: str = "") -> None:
    completed = load_progress() if completed is None else completed
    lang = get_language(lang)
    if not (console and Layout and Panel) or not _interactive_stdin():
        print_dashboard(completed, lang)
        _print()
        _print(command_hint(lang))
        return

    recommended = next_quest(completed)
    module_slug = recommended.module if recommended else MODULES[-1].slug
    module = MODULES_BY_SLUG[module_slug]
    chapter_table = _chapter_table(completed, lang, module_slug)
    dashboard_lines = [
        progress_line(completed, lang),
        next_recommendation(completed, lang),
        module_progress_line(module_slug, completed, lang),
        editor_summary(lang),
    ]
    if message:
        dashboard_lines.extend(["", message])
    width = terminal_width()
    height = max(1, terminal_height() - 1)
    header_size = 3 if height >= 8 else 2
    footer_size = 4 if width < 84 and height >= 14 else 3 if height >= 8 else 2
    main_layout = Layout(name="root")
    main_layout.split_column(
        Layout(name="header", size=header_size),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=footer_size),
    )
    title = "边学边做的 NineToothed 闯关课" if lang == "zh" else "A learn-by-doing NineToothed course"
    main_layout["header"].update(Panel(f"[bold]NineToothed Puzzles[/bold]\n{title}", border_style="cyan"))

    if width >= 108:
        main_layout["body"].split_row(Layout(name="left", size=42), Layout(name="right", ratio=1))
        main_layout["left"].update(Panel("\n".join(dashboard_lines), title="学习面板" if lang == "zh" else "Dashboard", border_style="green"))
        main_layout["right"].update(Panel(chapter_table, title=f"{_module_short_label(module_slug, lang)} · {text(module.title, lang)}", border_style="blue"))
    else:
        top_size = 7 if height >= 18 else 5
        main_layout["body"].split_column(Layout(name="top", size=top_size), Layout(name="bottom", ratio=1))
        main_layout["top"].update(Panel("\n".join(dashboard_lines), title="学习面板" if lang == "zh" else "Dashboard", border_style="green"))
        main_layout["bottom"].update(Panel(chapter_table, title=f"{_module_short_label(module_slug, lang)} · {text(module.title, lang)}", border_style="blue"))

    footer = command_hint(lang)
    if lang == "zh":
        footer += "\nEnter 进入下一关；map 看章节总览；editor vim 切换编辑器；quit 退出"
    else:
        footer += "\nEnter opens the next level; map shows chapters; editor vim changes editor; quit exits"
    main_layout["footer"].update(Panel(footer, border_style="magenta"))
    console.clear()
    console.print(main_layout, height=height)


def command_hint(lang: str) -> str:
    width = terminal_width()
    if lang == "zh":
        if width < 76:
            return "命令：Enter 下一关 / map 总览 / map all\n      play <关卡> / review [关卡] / editor vim / lang zh|en / reset / quit"
        return "命令：Enter 下一关 / map 总览 / map all / play <关卡> / review [关卡] / editor vim / lang zh|en / reset / quit"
    if width < 76:
        return "Commands: Enter next / map overview / map all\n          play <slug> / review [slug] / editor vim / lang zh|en / reset / quit"
    return "Commands: Enter next / map overview / map all / play <slug> / review [slug] / editor vim / lang zh|en / reset / quit"


def normalize_answer(value):
    if isinstance(value, tuple):
        return [normalize_answer(item) for item in value]
    if isinstance(value, list):
        return [normalize_answer(item) for item in value]
    if isinstance(value, str):
        return value.strip()
    return value


def parse_answer(raw: str):
    raw = raw.strip()
    try:
        return normalize_answer(ast.literal_eval(raw))
    except (SyntaxError, ValueError):
        if "," in raw and not raw.startswith("["):
            return [normalize_answer(part) for part in raw.split(",")]
        return normalize_answer(raw)


def _match_answer(parsed, expected) -> bool:
    parsed = normalize_answer(parsed)
    expected = normalize_answer(expected)
    if isinstance(parsed, str) and isinstance(expected, str):
        return parsed.lower() == expected.lower()
    return parsed == expected


def answer_is_correct(quest: Quest, raw: str) -> bool:
    parsed = parse_answer(raw)
    accepted = quest.accepted or (quest.expected,)
    return any(_match_answer(parsed, expected) for expected in accepted)


def _inferred_required_fragments(quest: Quest) -> tuple[str, ...]:
    if quest.required_fragments:
        return quest.required_fragments
    mapping = {
        "Tensor": "Tensor(",
        "Tensor.eval": ".eval",
        "eval": ".eval",
        "Tensor.subs": ".subs",
        "Symbol": "Symbol(",
        "constexpr": "constexpr=True",
        "tile": ".tile",
        "squeeze": ".squeeze",
        "flatten": ".flatten",
        "ravel": ".ravel",
        "pad": ".pad",
        "unsqueeze": ".unsqueeze",
        "expand": ".expand",
        "getitem": "[",
        "slicing": "[",
        "permute": ".permute",
        "Tensor(other=...)": "other=0",
        "ntl.sum": "ntl.sum",
        "ninetoothed.language": "ntl.",
    }
    fragments: list[str] = []
    for concept in quest.concepts:
        fragment = mapping.get(concept)
        if fragment and fragment not in fragments:
            fragments.append(fragment)
    return tuple(fragments)


def _fill_starter_body(quest: Quest) -> str:
    starters = {
        "tensor-rank": """\
def solve():
    x = ___  # create the rank-2 symbolic tensor
    return len(x.shape)
""",
        "eval-vector": """\
def solve():
    x = Tensor(1)
    observed = ___  # evaluate x after substituting shape (5,)
    return observed.tolist()
""",
        "subs-shape": """\
def solve():
    x = Tensor(2)
    y = ___  # substitute x with a concrete Tensor(shape=(2, 3))
    return list(y.shape)
""",
        "tile-even": """\
def solve():
    x = Tensor(1)
    tiled = ___  # tile x into blocks of size 2
    return tiled.eval({x: Tensor(shape=(6,))}).tolist()
""",
        "tile-padding": """\
def solve():
    x = Tensor(1)
    tiled = ___  # tile x into blocks of size 2
    return tiled.eval({x: Tensor(shape=(5,))}).tolist()
""",
        "tile-2d-shape": """\
def solve():
    x = Tensor(shape=(4, 6))
    tiled = ___  # tile into 2x3 inner blocks
    return [*list(tiled.shape), *list(tiled.dtype.shape)]
""",
        "flatten-inner": """\
def solve():
    x = Tensor(shape=(4, 4))
    tiled = x.tile((2, 2))
    tiled.dtype = ___  # flatten the inner block only
    return list(tiled.eval().shape)
""",
        "unsqueeze-expand": """\
def solve():
    x = Tensor(1)
    arranged = ___  # unsqueeze at axis 0, then expand to 3 rows
    return arranged.eval({x: Tensor(shape=(4,))}).tolist()
""",
        "squeeze-axis": """\
def solve():
    x = Tensor(shape=(1, 4))
    y = ___  # remove axis 0
    return list(y.shape)
""",
        "ravel-observe": """\
def solve():
    x = Tensor(shape=(2, 3))
    y = ___  # call ravel and observe the current index map
    return y.eval().tolist()
""",
        "pad-border": """\
def solve():
    x = Tensor(shape=(2, 2))
    padded = ___  # pad top by 1 and right by 1
    return padded.eval().tolist()
""",
        "slice-index": """\
def solve():
    x = Tensor(shape=(3, 4, 2))
    sliced = ___  # take x[:, 1, :]
    return sliced.eval().tolist()
""",
        "permute-map": """\
def solve():
    x = Tensor(shape=(2, 3))
    transposed = ___  # swap axes 0 and 1
    return transposed.eval().tolist()
""",
        "block-size-symbol": """\
def solve():
    block_size = ___  # create a constexpr Symbol named "block_size"
    x = Tensor(1)
    tiled = x.tile((block_size,))
    concrete = tiled.subs({x: Tensor(shape=(10,)), block_size: 4})
    return concrete.shape[0]
""",
        "block-size-helper": """\
def solve():
    BLOCK_SIZE = ___  # use the helper for tunable block sizes
    return "block_size"
""",
        "visualize-ready": """\
def solve():
    x = Tensor(2)
    tiled = x.tile((2, 2))
    concrete = ___  # substitute x with Tensor(shape=(4, 4))
    return [list(concrete.shape), list(concrete.dtype.shape)]
""",
        "arrange-three-vectors": """\
def solve():
    block_size = Symbol("block_size", constexpr=True)
    x = Tensor(1)
    arranged = ___  # arrange x with the shared vector tile shape
    concrete = arranged.subs({x: Tensor(shape=(8,)), block_size: 4})
    return concrete.shape[0]
""",
        "program-local-block": """\
def solve():
    block_size = Symbol("block_size", constexpr=True)
    x = Tensor(1)
    arranged = x.tile((block_size,))
    observed = arranged.eval({x: Tensor(shape=(8,)), block_size: 4})
    return ___  # return the local block for program instance 1
""",
        "outer-shape-rule": """\
def solve():
    x = Tensor(shape=(1, 2))
    expanded = ___  # expand x to a 3x2 outer view
    return "expand" if list(expanded.shape) == [3, 2] else "not-yet"
""",
        "matmul-output-grid": """\
def solve():
    output = Tensor(shape=(4, 8))
    output_tiled = ___  # tile output with block shape (2, 4)
    return list(output_tiled.shape)
""",
        "matmul-k-blocks": """\
def solve():
    k = 6
    block_size_k = 3
    return ___  # number of K-block loop iterations
""",
        "application-block": """\
def solve():
    x = Tensor(1).tile((4,))
    local_block = ___  # inspect the inner tensor seen by one program
    return list(local_block.shape)
""",
        "padding-other-zero": """\
def solve():
    x = ___  # construct Tensor(2, other=0) for row-sum padding
    return 0
""",
        "reduction-sum": """\
def solve():
    reducer = ___  # choose the ninetoothed.language reduction for sum
    return "ntl.sum" if reducer is ntl.sum else "not-yet"
""",
        "ntl-zeros-accumulator": """\
def application(output):
    accumulator = ___  # e.g. ntl.zeros(output.shape, dtype=ntl.float32)


def solve():
    return "ntl.zeros"
""",
        "ntl-full-sentinel": """\
def application(q):
    sentinel = ___  # e.g. ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)


def solve():
    return "ntl.full"
""",
        "ntl-dot-contract": """\
def application(lhs, rhs, output):
    product = ___  # local block matmul


def solve():
    return "ntl.dot"
""",
        "ntl-where-mask": """\
def application(mask, value, output):
    selected = ___  # masked selection


def solve():
    return "ntl.where"
""",
        "tensor-offsets": """\
def application(input, seed, p, output):
    mask = ___  # build a random mask from seed and input offsets


def solve():
    x = Tensor(1).tile((4,))
    marker = x.dtype  # keep this a real Tensor exercise, not a bare string.
    return "offsets" if marker.shape == (4,) else "not-yet"
""",
        "make-contract": """\
def solve():
    builder = ___  # choose the API that combines arrangement and application
    return "ninetoothed.make" if builder is ninetoothed.make else "not-yet"
""",
        "jit-contract": """\
def solve():
    decorator = ___  # choose the API used as @ninetoothed.jit
    return "ninetoothed.jit" if decorator is ninetoothed.jit else "not-yet"
""",
        "debug-simulate-arrangement": """\
def solve():
    from ninetoothed.debugging import ___  # import the arrangement simulator
    return "simulate_arrangement" if callable(simulate_arrangement) else "not-yet"
""",
        "visualize-api": """\
def solve():
    from ninetoothed.visualization import ___  # import the single-tensor visualizer
    return "visualize" if callable(visualize) else "not-yet"
""",
        "aot-api": """\
def solve():
    from ninetoothed.aot import ___  # import the AOT export helper
    return "aot" if callable(aot) else "not-yet"
""",
    }
    return starters.get(quest.slug, """\
def solve():
    # TODO: replace None with a small NineToothed program or expression.
    return None
""")


FILL_ANSWER_REPLACEMENTS: dict[str, str] = {
    "tensor-rank": "Tensor(2)",
    "eval-vector": "x.eval({x: Tensor(shape=(5,))})",
    "subs-shape": "x.subs({x: Tensor(shape=(2, 3))})",
    "tile-even": "x.tile((2,))",
    "tile-padding": "x.tile((2,))",
    "tile-2d-shape": "x.tile((2, 3))",
    "flatten-inner": "tiled.dtype.flatten()",
    "unsqueeze-expand": "x.unsqueeze(0).expand((3, -1))",
    "squeeze-axis": "x.squeeze(0)",
    "ravel-observe": "x.ravel()",
    "pad-border": "x.pad(((1, 0), (0, 1)))",
    "slice-index": "x[:, 1, :]",
    "permute-map": "x.permute((1, 0))",
    "block-size-symbol": 'Symbol("block_size", constexpr=True)',
    "block-size-helper": "block_size()",
    "visualize-ready": "tiled.subs({x: Tensor(shape=(4, 4))})",
    "arrange-three-vectors": "x.tile((block_size,))",
    "program-local-block": "observed[1].tolist()",
    "outer-shape-rule": "x.expand((3, -1))",
    "matmul-output-grid": "output.tile((2, 4))",
    "matmul-k-blocks": "(k + block_size_k - 1) // block_size_k",
    "application-block": "x.dtype",
    "padding-other-zero": "Tensor(2, other=0)",
    "reduction-sum": "ntl.sum",
    "ntl-zeros-accumulator": "ntl.zeros(output.shape, dtype=ntl.float32)",
    "ntl-full-sentinel": 'ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)',
    "ntl-dot-contract": "ntl.dot(lhs, rhs)",
    "ntl-where-mask": "ntl.where(mask, value, 0)",
    "tensor-offsets": "ntl.rand(seed, input.offsets()) > p",
    "make-contract": "ninetoothed.make",
    "jit-contract": "ninetoothed.jit",
    "debug-simulate-arrangement": "simulate_arrangement",
    "visualize-api": "visualize",
    "aot-api": "aot",
}


CODE_ANSWER_REPLACEMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "jit-vector-add": (
        ("output = lhs", "output = lhs + rhs"),
    ),
    "make-row-sum": (
        ("acc += 0", "acc += ntl.sum(x[0, i], axis=-1)"),
    ),
    "softmax-reduction": (
        ("row_minus_max = input_row", "row_minus_max = input_row - ntl.max(input_row)"),
        ("numerator = row_minus_max", "numerator = ntl.exp(row_minus_max)"),
        ("denominator = 1", "denominator = ntl.sum(numerator)"),
    ),
}


CONCEPT_EXPLANATIONS: dict[str, dict[str, str]] = {
    "Tensor": {
        "zh": "`Tensor` 是符号索引地图，不是已经存好数据的数组；它描述元素位置之间如何映射。",
        "en": "`Tensor` is a symbolic index map, not stored data; it describes how positions map.",
    },
    "Tensor.eval": {
        "zh": "`eval` 把符号张量具体化成可观察的索引数组，适合调试排布是否符合预期。",
        "en": "`eval` materializes a symbolic tensor into observable source indices, which is useful for debugging arrangement.",
    },
    "Tensor.subs": {
        "zh": "`subs` 代入具体符号或形状，但结果仍是 NineToothed Tensor，可以继续做元操作。",
        "en": "`subs` substitutes concrete symbols or shapes while keeping a NineToothed Tensor for further meta-operations.",
    },
    "tile": {
        "zh": "`tile` 把张量拆成外层块网格和内层局部块；外层通常对应程序实例。",
        "en": "`tile` splits a tensor into an outer block grid and inner local blocks; the outer grid often maps to programs.",
    },
    "padding": {
        "zh": "边界不足一整块时，`eval` 中默认用 `-1` 表示越界；真实 kernel 可用 `other` 指定越界读值。",
        "en": "When a boundary block is partial, `eval` shows out-of-bounds positions as `-1`; real kernels can set boundary load values with `other`.",
    },
    "dtype": {
        "zh": "嵌套张量的 `dtype` 可以看作每个外层块内部的局部张量视图。",
        "en": "For nested tensors, `dtype` acts like the local tensor view inside each outer block.",
    },
    "flatten": {
        "zh": "`flatten` 改的是局部块内部形状，不改变外层程序网格。",
        "en": "`flatten` changes the local block shape, not the outer program grid.",
    },
    "unsqueeze": {
        "zh": "`unsqueeze` 增加长度为 1 的轴，常用于后续广播对齐。",
        "en": "`unsqueeze` adds a size-1 axis, often before broadcasting alignment.",
    },
    "expand": {
        "zh": "`expand` 扩展视图以对齐外层形状，不表示复制真实数据。",
        "en": "`expand` extends a view to align outer shapes; it does not mean copying stored data.",
    },
    "squeeze": {
        "zh": "`squeeze` 移除长度为 1 的轴，让后续局部索引更直接。",
        "en": "`squeeze` removes a size-1 axis and makes later local indexing simpler.",
    },
    "pad": {
        "zh": "`pad` 会改变索引地图的边界区域，配合 `eval` 可以直接看到补出来的位置。",
        "en": "`pad` changes the boundary region of an index map; `eval` reveals padded positions directly.",
    },
    "permute": {
        "zh": "`permute` 重排维度，本质是改变索引地图的读法。",
        "en": "`permute` reorders dimensions; conceptually it changes how the index map is read.",
    },
    "Symbol": {
        "zh": "`Symbol(..., constexpr=True)` 表示编译期常量，常用于块大小。",
        "en": "`Symbol(..., constexpr=True)` represents a compile-time constant, commonly used for block sizes.",
    },
    "block_size": {
        "zh": "`block_size()` 是可调块大小入口；学习早期可先用显式 Symbol，调优时再切换。",
        "en": "`block_size()` is the tunable block-size hook; explicit Symbols are easier while learning.",
    },
    "ninetoothed.language": {
        "zh": "`ninetoothed.language as ntl` 提供应用函数里的局部计算操作，类似 Triton language。",
        "en": "`ninetoothed.language as ntl` provides local computation operations inside application functions, similar to Triton language.",
    },
    "ntl.sum": {
        "zh": "`ntl.sum` 对局部块做归约，是 row-sum、softmax 等 kernel 的基础。",
        "en": "`ntl.sum` reduces local blocks and is a foundation for row-sum, softmax, and similar kernels.",
    },
    "ntl.zeros": {
        "zh": "`ntl.zeros` 常用于创建累加器初值。",
        "en": "`ntl.zeros` is commonly used to create accumulator initial values.",
    },
    "ntl.full": {
        "zh": "`ntl.full` 创建指定值的局部张量，最大值归约常用 `-inf` 做哨兵。",
        "en": "`ntl.full` creates a local tensor filled with a chosen value; max reductions often use `-inf` as a sentinel.",
    },
    "ntl.dot": {
        "zh": "`ntl.dot` 是局部小块矩阵乘的核心操作。",
        "en": "`ntl.dot` is the core operation for local block matrix multiplication.",
    },
    "ntl.where": {
        "zh": "`ntl.where` 根据 mask 在两个值之间选择，常用于边界、dropout 和 causal mask。",
        "en": "`ntl.where` selects between two values by mask, often for boundaries, dropout, and causal masks.",
    },
    "offsets": {
        "zh": "`offsets()` 给出局部元素位置，随机 mask 和位置相关逻辑会用到它。",
        "en": "`offsets()` gives local element positions, useful for random masks and position-dependent logic.",
    },
    "make": {
        "zh": "`ninetoothed.make` 把 arrangement、application 和参数模板组装成 kernel。",
        "en": "`ninetoothed.make` combines arrangement, application, and parameter templates into a kernel.",
    },
    "jit": {
        "zh": "`ninetoothed.jit` 把排布直接写在参数注解里，适合短小 kernel。",
        "en": "`ninetoothed.jit` puts arrangement directly in parameter annotations, convenient for small kernels.",
    },
}


def mark_completed(slug: str) -> None:
    state = load_state()
    completed = set(state["completed"])
    completed.add(slug)
    state["completed"] = [quest.slug for quest in QUESTS if quest.slug in completed]
    save_state(state)


def solution_path(quest: Quest) -> Path:
    filename = quest.starter_name or f"{quest.slug}.py"
    return SOLUTIONS_DIR / filename


def reference_solution_source(quest: Quest) -> str:
    if quest.kind == "code":
        source = quest.starter
        for old, new in CODE_ANSWER_REPLACEMENTS.get(quest.slug, ()):
            source = source.replace(old, new)
        return textwrap.dedent(source).strip()
    source = _fill_starter_body(quest)
    replacement = FILL_ANSWER_REPLACEMENTS.get(quest.slug)
    if replacement is not None:
        source = source.replace("___", replacement, 1)
    return textwrap.dedent(source).strip()


def _expected_text(quest: Quest) -> str:
    if quest.expected is None:
        return "CUDA judge output must match the reference framework result."
    return repr(normalize_answer(quest.expected))


def _concept_lines(quest: Quest, lang: str) -> list[str]:
    lines: list[str] = []
    for concept in quest.concepts:
        detail = CONCEPT_EXPLANATIONS.get(concept)
        if detail:
            lines.append(f"- `{concept}`: {text(detail, lang)}")
    if not lines:
        lines.append(
            "- 这一关考察的是把课程片段转成可执行代码。"
            if lang == "zh"
            else "- This level asks you to turn the lesson snippet into executable code."
        )
    return lines


def solution_explanation_text(quest: Quest, lang: str | None = None) -> str:
    lang = get_language(lang)
    solution = reference_solution_source(quest)
    expected = _expected_text(quest)
    lesson_summary = text(quest.lesson[0].body, lang) if quest.lesson else ""
    hint = text(quest.hint, lang) if quest.hint else ""
    if lang == "zh":
        lines = [
            f"## 完整答案：{text(quest.title, lang)}",
            "",
            "### 参考实现",
            "",
            "```python",
            solution,
            "```",
            "",
            "### 为什么这样写",
            "",
            lesson_summary,
            "",
            f"本题目标不是记住最终值，而是让 `solve()` 通过 NineToothed API 自己算出结果。判题器会导入解题文件，执行 `solve()`，并检查返回值是否等于 `{expected}`。",
        ]
        if hint:
            lines.extend(["", f"提示对应的关键一步是：{hint}"])
        lines.extend(["", "### 关键知识点", "", *_concept_lines(quest, lang)])
        lines.extend(
            [
                "",
                "### 如何确认自己真正懂了",
                "",
                "把参考实现里的每个 NineToothed 调用说清楚：它改变的是维度数、具体 shape、索引地图、局部块，还是应用函数里的局部计算。能说清楚这一点，就不是在抄答案。",
            ]
        )
        return "\n".join(lines)

    lines = [
        f"## Full Answer: {text(quest.title, lang)}",
        "",
        "### Reference implementation",
        "",
        "```python",
        solution,
        "```",
        "",
        "### Why this works",
        "",
        lesson_summary,
        "",
        f"The goal is not to memorize the final value. `solve()` should compute it through NineToothed APIs. The judge imports the solution file, runs `solve()`, and checks that the return value equals `{expected}`.",
    ]
    if hint:
        lines.extend(["", f"The hint points to this key step: {hint}"])
    lines.extend(["", "### Key concepts", "", *_concept_lines(quest, lang)])
    lines.extend(
        [
            "",
            "### Check your understanding",
            "",
            "Explain each NineToothed call in the reference: did it change rank, concrete shape, index map, local block view, or local computation? If you can explain that, you are not just copying.",
        ]
    )
    return "\n".join(lines)


def render_solution_explanation(quest: Quest, lang: str) -> None:
    body = solution_explanation_text(quest, lang)
    if Panel and console:
        console.print(Panel(Markdown(body) if Markdown else body, title="完整答案" if lang == "zh" else "Full Answer", border_style="yellow"))
    else:
        _print(body)


def _with_edit_markers(source: str, *, note: str | None = None) -> str:
    if EDIT_START in source:
        return source
    note = note or "Replace every ___ or TODO in this region, then save the file."
    return f"{EDIT_START}\n# {note}\n{source.rstrip()}\n{EDIT_END}\n"


def _generic_fill_starter(quest: Quest) -> str:
    code_blocks = "\n\n".join(block.code for block in quest.lesson if block.code)
    lesson_code = textwrap.indent(code_blocks or "# No starter snippet for this level.", "# ")
    body = _with_edit_markers(
        _fill_starter_body(quest),
        note="Only edit the blanks below. Make solve() compute the answer.",
    )
    return f'''\
from ninetoothed import Symbol, Tensor, block_size
import ninetoothed
import ninetoothed.language as ntl


# Level: {quest.slug}
# Complete solve() as a small program. Do not type the answer into the terminal;
# make this file compute and return the requested value.
#
# Useful lesson snippet:
{lesson_code}

{body}
'''


def prepare_solution(quest: Quest) -> Path:
    SOLUTIONS_DIR.mkdir(exist_ok=True)
    path = solution_path(quest)
    if not path.exists():
        starter = (
            _with_edit_markers(quest.starter, note="Edit TODOs in this kernel file.")
            if quest.starter
            else _generic_fill_starter(quest)
        )
        path.write_text(textwrap.dedent(starter), encoding="utf-8")
    else:
        source = path.read_text(encoding="utf-8")
        if EDIT_START not in source and ("___" in source or "TODO" in source):
            path.write_text(_with_edit_markers(source), encoding="utf-8")
    return path


def _split_editor_command(configured: str) -> list[str] | None:
    if configured:
        try:
            command = shlex.split(configured)
        except ValueError:
            return None
        return command or None
    return None


def _editor_command() -> list[str] | None:
    saved = _split_editor_command(get_editor())
    if saved:
        return saved
    configured = _split_editor_command(os.environ.get("VISUAL") or os.environ.get("EDITOR") or "")
    if configured:
        return configured
    if os.name == "nt":
        return ["notepad"]
    for candidate in ("nano", "vim", "vi"):
        if shutil.which(candidate):
            return [candidate]
    return None


def editor_summary(lang: str) -> str:
    saved = get_editor()
    if saved:
        return f"当前编辑器：{saved}" if lang == "zh" else f"Current editor: {saved}"
    env_editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if env_editor:
        return f"当前编辑器：{env_editor}（来自环境变量）" if lang == "zh" else f"Current editor: {env_editor} (from environment)"
    detected = _editor_command()
    if detected:
        label = " ".join(detected)
        return f"当前编辑器：{label}（自动检测）" if lang == "zh" else f"Current editor: {label} (auto-detected)"
    return "当前没有找到编辑器。" if lang == "zh" else "No editor found."


def open_solution_editor(path: Path, lang: str) -> bool:
    if not _interactive_stdin():
        return False
    command = _editor_command()
    if not command:
        _print(
            f"没有找到编辑器。请设置 EDITOR，或手动编辑：{path}"
            if lang == "zh"
            else f"No editor found. Set EDITOR or edit manually: {path}"
        )
        return False
    _print(
        f"正在打开编辑器：{' '.join(command)} {path}"
        if lang == "zh"
        else f"Opening editor: {' '.join(command)} {path}"
    )
    if any("{file}" in part for part in command):
        command = [part.replace("{file}", str(path)) for part in command]
    else:
        command = [*command, str(path)]
    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError:
        _print(
            f"找不到编辑器命令：{command[0]}。可以输入 editor vim 或 editor auto 重新选择。"
            if lang == "zh"
            else f"Editor command not found: {command[0]}. Type editor vim or editor auto to choose again."
        )
        return False
    if completed.returncode != 0:
        _print(
            f"编辑器退出码：{completed.returncode}"
            if lang == "zh"
            else f"Editor exited with code {completed.returncode}"
        )
    return True


def _load_solution_module(path: Path):
    module_name = f"_ntpuzzles_solution_{path.stem}_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_solution(quest: Quest):
    path = prepare_solution(quest)
    module = _load_solution_module(path)
    if quest.kind != "code":
        return _validate_fill_solution(quest, module, path)
    if quest.validator == "vector_add":
        return _validate_vector_add(module)
    if quest.validator == "row_sum":
        return _validate_row_sum(module)
    if quest.validator == "softmax":
        return _validate_softmax(module)
    raise ValueError(f"no solution validator for {quest.slug}")


def _validate_fill_solution(quest: Quest, module, path: Path):
    source = path.read_text(encoding="utf-8")
    meaningful_source = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    missing = [
        fragment
        for fragment in _inferred_required_fragments(quest)
        if fragment not in meaningful_source
    ]
    if missing:
        raise AssertionError(
            "缺少必要 API / missing required API: "
            f"{', '.join(missing)}. "
            "这是代码填空题：请编辑解题文件里的 ___，不要只返回字面答案。"
        )
    if not hasattr(module, "solve"):
        raise AssertionError("solution must define solve()")
    actual = normalize_answer(module.solve())
    accepted = quest.accepted or (quest.expected,)
    if not any(_match_answer(actual, expected) for expected in accepted):
        raise AssertionError(f"solve() returned {actual!r}")
    return True


def _torch_cuda():
    try:
        import torch
    except ImportError as exc:
        raise PuzzleDependencyError("install torch to run CUDA levels") from exc
    if not torch.cuda.is_available():
        raise PuzzleDependencyError("CUDA is not available")
    return torch


def _validate_vector_add(module):
    torch = _torch_cuda()
    if not hasattr(module, "kernel"):
        raise AssertionError("solution must define kernel")
    lhs = torch.randn(257, device="cuda")
    rhs = torch.randn(257, device="cuda")
    output = torch.empty_like(lhs)
    module.kernel(lhs, rhs, output, block_size=64)
    expected = lhs + rhs
    if not torch.allclose(output, expected):
        raise AssertionError("output does not match lhs + rhs")
    return True


def _validate_row_sum(module):
    torch = _torch_cuda()
    if not hasattr(module, "build_kernel"):
        raise AssertionError("solution must define build_kernel()")
    kernel = module.build_kernel()
    x = torch.randn((9, 31), device="cuda")
    y = torch.empty((9, 1), device="cuda")
    kernel(x, y, block_size=8)
    expected = torch.sum(x, dim=-1, keepdim=True)
    if not torch.allclose(y, expected, atol=1e-5):
        raise AssertionError("output does not match torch.sum(x, dim=-1, keepdim=True)")
    return True


def _validate_softmax(module):
    torch = _torch_cuda()
    if not hasattr(module, "kernel"):
        raise AssertionError("solution must define kernel")
    x = torch.randn((7, 23), device="cuda")
    output = torch.empty_like(x)
    module.kernel(x, output, block_size=x.shape[-1])
    expected = torch.softmax(x, dim=-1)
    if not torch.allclose(output, expected, atol=1e-5):
        raise AssertionError("output does not match torch.softmax(x, dim=-1)")
    return True


def _header(title: str, subtitle: str = "") -> None:
    if Panel and console:
        console.print(Panel.fit(f"[bold]{title}[/bold]\n{subtitle}", border_style="cyan"))
    else:
        _print(title)
        if subtitle:
            _print(subtitle)


def _render_markdown(body: str) -> None:
    if Markdown and console:
        console.print(Markdown(body))
    else:
        _print(body)


def render_debrief(quest: Quest, lang: str) -> None:
    body = debrief_text(quest, lang)
    if Panel and console:
        console.print(Panel(Markdown(body) if Markdown else body, title="Debrief" if lang == "en" else "通关复盘", border_style="magenta"))
    else:
        _print(body)


def _render_code(code: str) -> None:
    if not code:
        return
    if console:
        console.print(Panel(code, title="code", border_style="blue"))
    else:
        _print(code)


def _solution_preview(path: Path, *, max_lines: int = 72) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = lines[:max_lines]
    head.append(f"# ... {len(lines) - max_lines} more lines not shown; open the file to edit.")
    return "\n".join(head)


def render_practice_workspace(quest: Quest, path: Path, lang: str) -> None:
    body = "\n".join(
        [
            f"**{text(quest.title, lang)}**",
            "",
            text(quest.question, lang),
            "",
            f"`{path}`",
            "",
            ("改文件中 EDIT BELOW / EDIT ABOVE 之间的 `___` 或 TODO。保存退出后会自动判题。"
             if lang == "zh"
             else "Edit the ___ or TODO between EDIT BELOW / EDIT ABOVE. Save and exit; the judge runs after that."),
            "",
            ("命令：Enter 判题 / edit 打开编辑器 / refresh 刷新 / hint 提示 / answer 答案解析 / quit 退出"
             if lang == "zh"
             else "Commands: Enter judge / edit open editor / refresh / hint / answer explanation / quit"),
        ]
    )
    code = _solution_preview(path)
    code_renderable = Syntax(code, "python", line_numbers=True, word_wrap=False) if Syntax else code
    if Panel and console:
        task_panel = Panel(Markdown(body) if Markdown else body, title="题目" if lang == "zh" else "Task", border_style="green")
        code_panel = Panel(code_renderable, title="解题文件" if lang == "zh" else "Solution File", border_style="blue")
        if Columns and terminal_width() >= 118:
            console.print(Columns([task_panel, code_panel], equal=True, expand=True))
        else:
            console.print(task_panel)
            console.print(code_panel)
        return
    _print(body)
    _print(code)


def _pause(lang: str) -> bool:
    message = "继续按 Enter，输入 q 退出" if lang == "zh" else "Press Enter to continue, or q to quit"
    raw = _prompt(message, default="")
    return raw.lower() not in {"q", "quit", "exit"}


def show_quest_lesson(quest: Quest, lang: str) -> bool:
    module = MODULES_BY_SLUG[quest.module]
    _header(text(quest.title, lang), text(quest.subtitle, lang))
    _print(f"{text(module.title, lang)} / {'CUDA' if quest.requires_cuda else 'CPU'}")
    _print()
    for block in quest.lesson:
        if Panel and console:
            console.print(Panel(text(block.body, lang), title=text(block.title, lang), border_style="green"))
        else:
            _print(text(block.title, lang))
            _print(text(block.body, lang))
        _render_code(block.code)
        if not _pause(lang):
            return False
    return True


def _direct_answer_warning(path: Path, lang: str) -> str:
    if lang == "zh":
        return (
            f"这里不是输入答案的位置。请编辑 {path}，把 ___ 补成代码，"
            "保存后回到这里直接按 Enter 判题。如果你在做第一题，应该让代码构造 "
            "Tensor(2)，而不是在终端输入 2。"
        )
    return (
        f"This prompt is not for typing the answer. Edit {path}, replace ___ with code, "
        "save it, then press Enter here to judge. If this is the first level, the code should "
        "construct Tensor(2), not type 2 in the terminal."
    )


def _judge_solution(quest: Quest, lang: str) -> bool:
    try:
        validate_solution(quest)
    except PuzzleDependencyError as exc:
        _print(f"依赖缺失：{exc}" if lang == "zh" else f"Dependency missing: {exc}")
        return False
    except Exception as exc:
        _print(f"还没过：{exc}" if lang == "zh" else f"Not cleared yet: {exc}")
        return False
    _print()
    _print(f"[bold green]{text(quest.success, lang)}[/bold green]" if console else text(quest.success, lang))
    mark_completed(quest.slug)
    render_debrief(quest, lang)
    return True


def _offer_answer_if_unlocked(attempts: int, lang: str) -> None:
    if attempts < ANSWER_UNLOCK_ATTEMPTS:
        return
    _print(
        f"已经尝试 {attempts} 次。可以输入 answer 查看完整参考答案和原理解释。"
        if lang == "zh"
        else f"{attempts} attempts so far. Type answer for the full reference solution and explanation."
    )


def _answer_locked_message(lang: str) -> str:
    return (
        f"先尝试至少 {ANSWER_UNLOCK_ATTEMPTS} 次；如果仍然卡住，再输入 answer 看完整讲解。"
        if lang == "zh"
        else f"Try at least {ANSWER_UNLOCK_ATTEMPTS} times first; if you are still stuck, type answer for the full explanation."
    )


def play_fill_quest(quest: Quest, lang: str) -> bool:
    if not show_quest_lesson(quest, lang):
        return False
    path = prepare_solution(quest)
    _print()
    render_practice_workspace(quest, path, lang)
    attempts = 0
    if open_solution_editor(path, lang):
        _print("编辑器已关闭，开始判题。" if lang == "zh" else "Editor closed; judging now.")
        if _judge_solution(quest, lang):
            return True
        attempts += 1
        _offer_answer_if_unlocked(attempts, lang)
        render_practice_workspace(quest, path, lang)
    while True:
        raw = _prompt("check", default="")
        if raw.lower() in {"quit", "q", "exit"}:
            return False
        if raw.lower() == "hint":
            _print(text(quest.hint, lang))
            continue
        if raw.lower() == "path":
            _print(str(path))
            continue
        if raw.lower() == "answer":
            if attempts >= ANSWER_UNLOCK_ATTEMPTS:
                render_solution_explanation(quest, lang)
            else:
                _print(_answer_locked_message(lang))
            continue
        if raw.lower() in {"edit", "e"}:
            open_solution_editor(path, lang)
            render_practice_workspace(quest, path, lang)
            if _judge_solution(quest, lang):
                return True
            attempts += 1
            _offer_answer_if_unlocked(attempts, lang)
            continue
        if raw.lower() in {"refresh", "r"}:
            render_practice_workspace(quest, path, lang)
            continue
        if raw.strip():
            _print(_direct_answer_warning(path, lang))
            continue
        if _judge_solution(quest, lang):
            return True
        attempts += 1
        _offer_answer_if_unlocked(attempts, lang)


def play_code_quest(quest: Quest, lang: str) -> bool:
    if not show_quest_lesson(quest, lang):
        return False
    path = prepare_solution(quest)
    _print()
    render_practice_workspace(quest, path, lang)
    attempts = 0
    if open_solution_editor(path, lang):
        _print("编辑器已关闭，开始判题。" if lang == "zh" else "Editor closed; judging now.")
        if _judge_solution(quest, lang):
            return True
        attempts += 1
        _offer_answer_if_unlocked(attempts, lang)
        render_practice_workspace(quest, path, lang)
    while True:
        raw = _prompt("check", default="")
        if raw.lower() in {"quit", "q", "exit"}:
            return False
        if raw.lower() == "hint":
            _print(text(quest.hint, lang))
            continue
        if raw.lower() == "path":
            _print(str(path))
            continue
        if raw.lower() == "answer":
            if attempts >= ANSWER_UNLOCK_ATTEMPTS:
                render_solution_explanation(quest, lang)
            else:
                _print(_answer_locked_message(lang))
            continue
        if raw.lower() in {"edit", "e"}:
            open_solution_editor(path, lang)
            render_practice_workspace(quest, path, lang)
            if _judge_solution(quest, lang):
                return True
            attempts += 1
            _offer_answer_if_unlocked(attempts, lang)
            continue
        if raw.lower() in {"refresh", "r"}:
            render_practice_workspace(quest, path, lang)
            continue
        if raw.strip():
            _print("输入 edit 打开文件，或直接按 Enter 判题。" if lang == "zh" else "Type edit to open the file, or press Enter to judge.")
            continue
        if _judge_solution(quest, lang):
            return True
        attempts += 1
        _offer_answer_if_unlocked(attempts, lang)


def play_quest(slug: str | None = None, lang: str | None = None, editor: str | None = None) -> int:
    if editor is not None:
        set_editor(editor)
    lang = get_language(lang)
    completed = load_progress()
    quest = QUESTS_BY_SLUG.get(slug) if slug else next_quest(completed)
    if quest is None:
        _print("全部关卡已通关。" if lang == "zh" else "All levels are cleared.")
        return 0
    if quest.slug not in completed and not is_unlocked(quest.slug, completed):
        _print(f"{quest.slug} 还未解锁，请先完成前面的关卡。" if lang == "zh" else f"{quest.slug} is locked. Clear previous levels first.")
        return 2
    cleared = play_code_quest(quest, lang) if quest.kind == "code" else play_fill_quest(quest, lang)
    return 0 if cleared else 1


def show_quest(slug: str, lang: str | None = None) -> int:
    lang = get_language(lang)
    quest = QUESTS_BY_SLUG.get(slug)
    if quest is None:
        _print(f"Unknown level: {slug}")
        return 2
    _header(text(quest.title, lang), text(quest.subtitle, lang))
    for block in quest.lesson:
        _print(f"- {text(block.title, lang)}: {text(block.body, lang)}")
        _render_code(block.code)
    _print(text(quest.question, lang))
    return 0


def review_quest(slug: str | None = None, lang: str | None = None) -> int:
    lang = get_language(lang)
    completed = load_progress()
    cleared = [quest for quest in QUESTS if quest.slug in completed]
    if not cleared:
        _print("还没有已通关关卡，先完成第一关。" if lang == "zh" else "No cleared levels yet. Clear the first level first.")
        return 1

    if slug is None:
        _print("已通关关卡：" if lang == "zh" else "Cleared levels:")
        for quest in cleared:
            _print(f"- {quest.slug} · {text(quest.title, lang)}")
        slug = _prompt("选择要复盘的关卡" if lang == "zh" else "Review which level", default=cleared[-1].slug).strip()

    quest = QUESTS_BY_SLUG.get(slug)
    if quest is None:
        _print(f"Unknown level: {slug}")
        return 2
    if quest.slug not in completed:
        _print("只能复盘已通关关卡。" if lang == "zh" else "Only cleared levels can be reviewed.")
        return 2

    show_quest(quest.slug, lang)
    _print()
    render_debrief(quest, lang)
    _print()
    render_solution_explanation(quest, lang)
    _print()
    _print(("解题文件：" if lang == "zh" else "Solution file: ") + str(solution_path(quest)))
    return 0


def check_builtin(slug: str | None, *, all_: bool, skip_cuda: bool, lang: str | None = None) -> int:
    _ = get_language(lang)
    if all_:
        puzzles = PUZZLES
    else:
        if slug in VALIDATOR_BY_QUEST:
            slug = VALIDATOR_BY_QUEST[slug]
        puzzle = PUZZLES_BY_SLUG.get(slug or "")
        if puzzle is None:
            _print("Choose a validator slug, or use --all.")
            return 2
        puzzles = (puzzle,)

    ok = True
    for puzzle in puzzles:
        if skip_cuda and puzzle.requires_cuda:
            _print(f"SKIP {puzzle.slug}: CUDA")
            continue
        try:
            result = puzzle.run()
        except PuzzleDependencyError as exc:
            _print(f"SKIP {puzzle.slug}: {exc}")
            ok = ok and not puzzle.requires_cuda
            continue
        except Exception as exc:
            _print(f"FAIL {puzzle.slug}: {exc}")
            ok = False
            continue
        _print(f"{'PASS' if result.passed else 'FAIL'} {puzzle.slug}")
        for observation in result.observations:
            _print(f"  - {observation}")
        ok = ok and result.passed
    return 0 if ok else 1


def start_game(lang: str | None = None, editor: str | None = None) -> int:
    if lang:
        set_language(lang)
    if editor is not None:
        set_editor(editor)
    lang = get_language(lang)
    screen = console.screen(hide_cursor=False) if console and _interactive_stdin() else nullcontext()
    message = ""
    with screen:
        while True:
            completed = load_progress()
            render_home_screen(completed, lang, message)
            command = _prompt("ntpuzzles", default="").strip()
            message = ""
            if command == "":
                play_quest(lang=lang)
            elif command in {"quit", "q", "exit"}:
                return 0
            elif command == "map":
                if console:
                    console.clear()
                print_course_overview(load_progress(), lang)
                if not _pause(lang):
                    return 0
                continue
            elif command in {"map all", "map full", "map compact"}:
                if console:
                    console.clear()
                print_map(load_progress(), lang, style="compact", context="cli")
                if not _pause(lang):
                    return 0
                continue
            elif command.startswith("map "):
                _, module_slug = command.split(maxsplit=1)
                if module_slug not in MODULES_BY_SLUG:
                    message = "未知章节。可用章节：" + ", ".join(module.slug for module in MODULES) if lang == "zh" else "Unknown chapter. Available: " + ", ".join(module.slug for module in MODULES)
                    continue
                if console:
                    console.clear()
                print_map(load_progress(), lang, style="compact", context="tui", module_slug=module_slug)
                if not _pause(lang):
                    return 0
                continue
            elif command.startswith("lang "):
                _, selected = command.split(maxsplit=1)
                set_language(selected)
                lang = get_language(selected)
                message = "语言已切换。" if lang == "zh" else "Language changed."
            elif command == "editor":
                message = editor_summary(lang) + ("\n输入 editor vim 切换到 vim；输入 editor auto 回到自动检测。" if lang == "zh" else "\nType editor vim to switch to vim; type editor auto to return to auto-detection.")
            elif command.startswith("editor "):
                _, selected = command.split(maxsplit=1)
                set_editor(selected)
                message = editor_summary(lang)
            elif command == "reset":
                confirm = _prompt("输入 yes 确认重置" if lang == "zh" else "Type yes to reset", default="no")
                if confirm.lower() == "yes":
                    reset_progress()
                    message = "进度已重置。" if lang == "zh" else "Progress reset."
                else:
                    message = "已取消重置。" if lang == "zh" else "Reset cancelled."
            elif command.startswith("play "):
                _, slug = command.split(maxsplit=1)
                play_quest(slug, lang)
            elif command == "review":
                review_quest(lang=lang)
            elif command.startswith("review "):
                _, slug = command.split(maxsplit=1)
                review_quest(slug, lang)
            else:
                message = "未知命令。" if lang == "zh" else "Unknown command."
