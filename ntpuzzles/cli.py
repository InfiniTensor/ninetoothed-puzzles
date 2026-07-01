from __future__ import annotations

import argparse

from .curriculum import MODULES_BY_SLUG, QUESTS, text
from .game import (
    check_builtin,
    get_language,
    is_unlocked,
    load_progress,
    play_quest,
    print_map,
    reset_progress,
    review_quest,
    set_editor,
    set_language,
    show_quest,
    start_game,
)


def _add_language_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lang", choices=("zh", "en"), help="Language / 语言")


def _add_editor_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--editor", help="Editor command for level files, e.g. vim, nano, 'code --wait', or auto.")


def _cmd_list(args: argparse.Namespace) -> int:
    if args.lang:
        set_language(args.lang)
    lang = get_language(args.lang)
    print_map(load_progress(), lang, style=args.style, context="cli", module_slug=args.chapter)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    if args.lang:
        set_language(args.lang)
    return show_quest(args.slug, args.lang)


def _cmd_play(args: argparse.Namespace) -> int:
    if args.lang:
        set_language(args.lang)
    if args.editor is not None:
        set_editor(args.editor)
    return play_quest(args.slug, args.lang)


def _cmd_review(args: argparse.Namespace) -> int:
    if args.lang:
        set_language(args.lang)
    return review_quest(args.slug, args.lang)


def _cmd_start(args: argparse.Namespace) -> int:
    return start_game(args.lang, args.editor)


def _cmd_check(args: argparse.Namespace) -> int:
    return check_builtin(args.slug, all_=args.all, skip_cuda=args.skip_cuda, lang=args.lang)


def _cmd_reset(_: argparse.Namespace) -> int:
    reset_progress()
    print("Progress reset.")
    return 0


def _cmd_syllabus(args: argparse.Namespace) -> int:
    if args.lang:
        set_language(args.lang)
    lang = get_language(args.lang)
    for module_slug in dict.fromkeys(quest.module for quest in QUESTS):
        module = MODULES_BY_SLUG[module_slug]
        print()
        print(text(module.title, lang))
        print(text(module.summary, lang))
        for quest in [item for item in QUESTS if item.module == module_slug]:
            marker = "*" if is_unlocked(quest.slug) else "-"
            print(f"  {marker} {quest.slug:22} {text(quest.title, lang)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ntpuzzles",
        description="NineToothed bilingual learning game.",
    )
    _add_language_arg(parser)
    _add_editor_arg(parser)
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start the interactive TUI course.")
    _add_language_arg(start_parser)
    _add_editor_arg(start_parser)
    start_parser.set_defaults(func=_cmd_start)

    list_parser = subparsers.add_parser("list", help="Show the level map.")
    list_parser.add_argument("--style", choices=("auto", "full", "medium", "compact"), default="auto")
    list_parser.add_argument("--chapter", choices=tuple(MODULES_BY_SLUG), help="Only show one chapter/module.")
    _add_language_arg(list_parser)
    list_parser.set_defaults(func=_cmd_list)

    syllabus_parser = subparsers.add_parser("syllabus", help="Show the course syllabus.")
    _add_language_arg(syllabus_parser)
    syllabus_parser.set_defaults(func=_cmd_syllabus)

    show_parser = subparsers.add_parser("show", help="Show one lesson without playing it.")
    show_parser.add_argument("slug")
    _add_language_arg(show_parser)
    show_parser.set_defaults(func=_cmd_show)

    play_parser = subparsers.add_parser("play", help="Play one unlocked level.")
    play_parser.add_argument("slug", nargs="?")
    _add_language_arg(play_parser)
    _add_editor_arg(play_parser)
    play_parser.set_defaults(func=_cmd_play)

    review_parser = subparsers.add_parser("review", help="Review a cleared level and its debrief.")
    review_parser.add_argument("slug", nargs="?")
    _add_language_arg(review_parser)
    review_parser.set_defaults(func=_cmd_review)

    check_parser = subparsers.add_parser("check", help="Run environment validators without the game.")
    check_parser.add_argument("slug", nargs="?")
    check_parser.add_argument("--all", action="store_true", help="Run every validator.")
    check_parser.add_argument("--skip-cuda", action="store_true", help="Skip CUDA validators.")
    _add_language_arg(check_parser)
    check_parser.set_defaults(func=_cmd_check)

    reset_parser = subparsers.add_parser("reset", help="Reset saved game progress.")
    reset_parser.set_defaults(func=_cmd_reset)

    parser.set_defaults(func=_cmd_start)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
