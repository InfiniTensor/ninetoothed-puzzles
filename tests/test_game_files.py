from ntpuzzles import game
from ntpuzzles.cli import build_parser
from ntpuzzles.curriculum import QUESTS_BY_SLUG


def test_prepare_solution_creates_marked_fill_file(tmp_path, monkeypatch):
    monkeypatch.setattr(game, "SOLUTIONS_DIR", tmp_path)
    quest = QUESTS_BY_SLUG["tensor-rank"]

    path = game.prepare_solution(quest)
    source = path.read_text(encoding="utf-8")

    assert path == tmp_path / "tensor-rank.py"
    assert game.EDIT_START in source
    assert game.EDIT_END in source
    assert "from ninetoothed import Symbol, Tensor, block_size" in source
    assert "def solve():" in source


def test_prepare_solution_creates_marked_code_file(tmp_path, monkeypatch):
    monkeypatch.setattr(game, "SOLUTIONS_DIR", tmp_path)
    quest = QUESTS_BY_SLUG["jit-vector-add"]

    path = game.prepare_solution(quest)
    source = path.read_text(encoding="utf-8")

    assert path.name == "vector_add_kernel.py"
    assert game.EDIT_START in source
    assert game.EDIT_END in source
    assert "@ninetoothed.jit" in source
    assert "TODO" in source


def test_cli_parser_defaults_to_start_game():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.func.__name__ == "_cmd_start"


def test_cli_parser_accepts_common_commands():
    parser = build_parser()

    list_args = parser.parse_args(["list", "--chapter", "meta", "--style", "compact"])
    assert list_args.command == "list"
    assert list_args.chapter == "meta"

    play_args = parser.parse_args(["play", "tensor-rank", "--editor", "vim", "--lang", "zh"])
    assert play_args.command == "play"
    assert play_args.slug == "tensor-rank"
    assert play_args.editor == "vim"
