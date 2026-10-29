import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'starter_kit'))
sys.path.insert(0, os.path.join(os.getcwd(), 'starter_kit', 'dfthc'))

from starter.task3_circuits.build_circuits import build_submission
from starter.task1_factorization.solution_io import BASELINES_DIRECTORY
from verifier.reference import load_systems

systems = load_systems()
try:
    path = BASELINES_DIRECTORY / systems['FeMoCo76o113e']['baseline']
    print(path)
    res = build_submission('FeMoCo76o113e', path, overwrite=True)
    print(res['score'])
    print('done')
except Exception as e:
    import traceback
    traceback.print_exc()
