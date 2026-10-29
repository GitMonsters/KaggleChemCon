from starter.task3_circuits.build_circuits import build_submission
from starter.task1_factorization.solution_io import BASELINES_DIRECTORY
from verifier.reference import load_systems
systems = load_systems()
path = BASELINES_DIRECTORY / systems['Fe2S2-20o30e']['baseline']
try:
    res = build_submission('Fe2S2-20o30e', path, overwrite=True)
    print(res['score'])
except Exception as e:
    import traceback
    traceback.print_exc()
