# NineToothed Puzzles

`ninetoothed-puzzles` is a bilingual learn-by-doing course for the NineToothed
DSL. It is designed as a small learning platform, not a test script: students
read a short lesson, inspect code, complete a fill-in programming file, unlock
the next level, and eventually edit real CUDA kernel solution files.

The original notebook-style tutorial is still available in
[`ninetoothed-puzzles.py`](ninetoothed-puzzles.py).

## Start The Course

```bash
python puzzle_runner.py --lang zh
```

Use a specific editor, for example Vim:

```bash
python puzzle_runner.py --editor vim --lang zh
```

or:

```bash
python puzzle_runner.py --lang en
```

The default language is Chinese. You can also change language inside the
terminal course:

```text
lang zh
lang en
```

Progress is saved in `.ntpuzzles_progress.json`. Every level generates an
editable Python file under `solutions/`; both are local state and ignored by
git.

For students, the TUI is the recommended terminal entry point: it uses a
persistent full-screen dashboard, reserves the bottom line for commands, focuses
on the current chapter in narrow terminals, and uses `map` for a six-chapter
overview instead of dumping every level at once. `map all` is available when the student really wants the full
compact map. The CLI subcommands remain useful for scripting, checking, jumping
to a known level, and sharing precise commands in docs or CI.

The TUI uses `prompt_toolkit` when available, so arrow keys behave like a normal
shell prompt for command editing/history instead of printing escape characters.

## Terminal Commands

```bash
python puzzle_runner.py                 # start the TUI course
python puzzle_runner.py start --lang zh # start explicitly
python puzzle_runner.py list --lang zh  # level map
python puzzle_runner.py list --style compact --lang zh
python puzzle_runner.py list --chapter meta --lang zh
python puzzle_runner.py syllabus --lang en
python puzzle_runner.py show tile-even --lang zh
python puzzle_runner.py play tensor-rank --lang zh
python puzzle_runner.py review tensor-rank --lang zh
python puzzle_runner.py reset
```

Inside the TUI, the main loop is:

```text
Enter              # open the next unlocked programming level
play <slug>        # jump to an unlocked level
review [slug]      # revisit a cleared level and its debrief
editor             # show the current editor
editor vim         # use Vim for generated solution files
editor auto        # go back to auto-detecting VISUAL/EDITOR/nano/vim/vi
map                # inspect the six-chapter overview
map all            # inspect every level in compact form
map meta           # inspect one chapter/module by slug
lang zh|en         # switch language
```

The important interaction is the generated programming file. Students fill the
`___` blanks in `solutions/<level>.py`, return to the TUI or web page, and run
the judge. A passing level unlocks a debrief so the student can connect their
code to the NineToothed concept it practiced.

When a level starts in the TUI, the course renders the task and current solution
file together. Wide terminals use a left/right split; narrow terminals stack
task and code vertically. The starter file is opened automatically with
`$VISUAL`, `$EDITOR`, or a detected terminal editor such as `nano`/`vim`/`vi`.
Save and exit the editor to run the judge. Type `edit` inside the level to open
the file again.

Every level has a hint and a full answer explanation. Students who already know
the concept can solve directly; students who are close can use `hint`; after two
failed judge attempts the TUI suggests `answer`, which reveals the reference
implementation and a principle-level explanation. The web UI follows the same
rule with an expandable full answer after repeated failures.

Environment validators are still available, but they are not the learning
experience:

```bash
python puzzle_runner.py check --all --skip-cuda
python puzzle_runner.py check --all
```

CUDA levels compare against torch only as a familiar numerical oracle. The
student-facing task is still NineToothed: arrange symbolic tensors, reason about
local blocks, and write `ninetoothed.language` application code.

## Web Learning Platform

For a browser-based experience:

```bash
python -m pip install -e ".[runtime,ui]"
python -m streamlit run ui/app.py
```

The web UI uses the same course data and progress file as the terminal. It has:

- Chinese/English language switching.
- Chapter navigation.
- A prominent coding panel directly on the level page, with save and judge
  buttons visible without opening a tab.
- Python syntax highlighting through the browser editor when `streamlit-ace` is
  installed.
- Lesson, visual lab, debrief, and reference tabs for supporting material.
- NineToothed official `visualize(..., save_path=...)` images for `eval`,
  `tile`, `expand`, `pad`, `permute`, local blocks, and program grids, with
  small numeric tables as supplements.
- Per-level debriefs after clearing a programming task.
- Recommended next level and chapter progress.
- Progress tracking.
- Browser-editable fill-in code files.
- Reference output hidden until the level is cleared.
- CUDA code challenge generation and judging.

## Course Path

The current course has 37 checkpoints across 6 chapters.

| Chapter | Focus |
| --- | --- |
| Symbolic Tensors | `Tensor`, rank, shape substitution, `eval`, `Symbol` |
| Meta-Operation Maps | `tile`, padding, nested tensors, `flatten`, `unsqueeze`, `squeeze`, `ravel`, `pad`, `expand`, slicing, `permute` |
| Arrangement and Program Grids | `block_size()`, constexpr block sizes, outer shape, program instances, visualization-ready `subs`, matmul block relationships |
| Application Functions | local blocks, `other`, `ninetoothed.language`, reductions |
| Language Ops and Tools | `ntl.zeros`, `ntl.full`, `ntl.dot`, `ntl.where`, `offsets()`, `make`, `jit`, debugging, visualization, AOT |
| Real Kernels | editable CUDA challenges for vector add, row sum, and softmax |

## Repository Layout

```text
ninetoothed-puzzles.py   # Existing notebook-style tutorial.
puzzle_runner.py         # Course launcher.
ntpuzzles/curriculum.py  # Bilingual course content and level metadata.
ntpuzzles/game.py        # Terminal game, progress, answer checking, code judging.
ntpuzzles/levels.py      # Environment/reference validators.
ui/app.py                # Streamlit learning platform.
tests/                   # Fast unit tests for curriculum and starter generation.
```

## Adding A Level

1. Add the bilingual learner-facing level in `ntpuzzles/curriculum.py`.
2. Put the level in a chapter and choose a fill-in level or a CUDA code level.
3. For code levels, add a starter file template and a validator in
   `ntpuzzles/game.py`.
4. Keep `ntpuzzles/levels.py` for environment/reference smoke checks.
5. Verify:

```bash
python puzzle_runner.py syllabus --lang zh
python puzzle_runner.py play <slug> --lang zh
python -m pytest
python puzzle_runner.py check --all --skip-cuda
```
