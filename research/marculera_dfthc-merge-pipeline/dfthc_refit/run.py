"""Run the dfthcsos pipeline from a recipe: train -> [lbfgs] -> round -> [polish] -> refine -> [refit] -> build and verify.

    python notebooks/dfthc-refit/run.py recipes/merged_10x120x60.toml [--out DIR] [--scale 0.001]

A recipe (TOML) names a public seed, how many groups to merge, the final shape, a list of
training steps, the rounding and the circuits; see `recipes/`. Each training step is one
`middle.py` run with a fresh optimizer: it drops to `groups` (if fewer than before) and fits.
Any other key of a step is passed through as a `middle.py` option (`jitter = 0.003` becomes
`--jitter 0.003`).

Every step writes its result into the run directory and is skipped when that file exists, so an
interrupted run (a Kaggle session that hit its time limit) resumes where it stopped. `--scale`
multiplies every step count, for a quick smoke test of a recipe.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import WORK  # noqa: E402

STEP_KEYS = ('fade', 'fit', 'warmup', 'stage_fit', 'stage_steps', 'final_steps', 'fit_steps')   # scaled by --scale


def option(key: str, value) -> list[str]:
    flag = '--' + key.replace('_', '-')
    if value is True:
        return [flag]
    if isinstance(value, list):
        return [flag, *map(str, value)]
    return [flag, str(value)]


def plan(recipe: dict, out: Path, scale: float) -> list[tuple[str, Path, list[str]]]:
    """The (name, output, command) of every step, in order."""
    def steps(n: int) -> int:
        return max(2, round(n * scale))

    R, B, C_final = recipe['shape']
    C = recipe.get('start_copies', C_final)  # the copies after merging, before any step cuts them
    commands, previous, groups = [], None, None
    for i, step in enumerate(recipe['train']):
        step = dict(step)
        target = step.pop('groups')
        bases = step.pop('bases', B)  # grow to the final B at a later step (cheaper drops before it)
        cut = step.pop('copies', None)  # fade copies down to this many (in each group's SVD basis)
        if cut is not None:
            C = cut
            step['copies'] = cut
        result = out / f'train_{i}_{target}x{bases}x{C}.npz'
        cmd = ['middle.py', str(target), str(bases), str(C)]
        if previous is None:
            cmd += ['--seed', recipe['seed'], '--merge', str(recipe.get('merge', 1))]
        else:
            cmd += ['--init', str(previous)]
            if target < groups:
                cmd += ['--stages', str(target)]
        for key, value in step.items():
            cmd += option(key, steps(value) if key in STEP_KEYS else value)
        cmd += ['--print-every', str(steps(recipe.get('print_every', 10000))), '--out', str(result)]
        commands.append((f'train {i}', result, cmd))
        previous, groups = result, target
    if (groups, bases, C) != (R, B, C_final):
        raise ValueError(f'the last training step ends at {groups} groups and {C} copies, not shape = {recipe["shape"]}')

    if 'lbfgs' in recipe:  # float64 L-BFGS polish of the continuous fit before rounding
        fitted = out / 'lbfgs.npz'
        cmd = ['lbfgs_fit.py', str(previous), str(fitted)]
        for key, value in recipe['lbfgs'].items():
            cmd += option(key, value)
        commands.append(('lbfgs', fitted, cmd))
        previous = fitted
    rounded, refined, built = out / 'rounded.npz', out / 'refined.npz', out / 'submission'
    rnd = dict(recipe.get('round', {}))
    cmd = ['quantize.py', str(previous), str(rounded)]
    for key, value in rnd.items():
        cmd += option(key, steps(value) if key in STEP_KEYS else value)
    commands.append(('round', rounded, cmd))
    brot = str(rnd.get('brot', 10))
    if 'polish' in recipe:  # exact coordinate descent on the rounded words
        polished = out / 'polished.npz'
        cmd = ['polish_words.py', str(rounded), str(polished), '--brot', brot]
        for key, value in recipe['polish'].items():
            cmd += option(key, value)
        commands.append(('polish', polished, cmd))
        rounded = polished
    cmd = ['refine.py', str(rounded), str(refined), '--brot', brot]
    for key, value in recipe.get('refine', {}).items():
        cmd += option(key, value)
    commands.append(('refine', refined, cmd))
    final = refined
    if 'refit' in recipe:  # spend the error budget left after rounding on lambda_eff
        final = out / 'refit.npz'
        cmd = ['refit_weights.py', str(refined), str(final)]
        for key, value in recipe['refit'].items():
            cmd += option(key, value)
        commands.append(('refit', final, cmd))
    bld = recipe.get('build', {})
    commands.append(('build', out / 'result.json',
                     ['build.py', str(final), str(built), '--brot', brot,
                      '--qrom', bld.get('qrom', 'four'), '--shots', str(bld.get('shots', 64)),
                      '--json', str(out / 'result.json')]))
    return commands


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('recipe', type=Path)
    ap.add_argument('--out', type=Path, help='run directory (default: WORK/<recipe name>)')
    ap.add_argument('--scale', type=float, default=1.0, help='multiply every step count (smoke tests)')
    ap.add_argument('--dry-run', action='store_true', help='print the commands only')
    a = ap.parse_args()
    if not a.recipe.exists() and (HERE / a.recipe).exists():   # e.g. recipes/x.toml next to this file
        a.recipe = HERE / a.recipe
    recipe = tomllib.loads(a.recipe.read_text())
    out = a.out or WORK / 'recipes' / a.recipe.stem
    out.mkdir(parents=True, exist_ok=True)
    commands = plan(recipe, out, a.scale)
    (out / 'commands.txt').write_text(''.join(shlex.join(c) + '\n' for _, _, c in commands))
    for name, result, cmd in commands:
        if result.exists():
            print(f'== {name}: {result.name} exists, skipped', flush=True)
            continue
        print(f'== {name}: {shlex.join(cmd)}', flush=True)
        if not a.dry_run and subprocess.run([sys.executable, '-u', str(HERE / cmd[0]), *cmd[1:]]).returncode:
            sys.exit(f'== {name} failed; earlier results are kept, rerun to resume')
    if not a.dry_run:
        print(json.dumps(json.loads((out / 'result.json').read_text()), indent=1))


if __name__ == '__main__':
    main()
