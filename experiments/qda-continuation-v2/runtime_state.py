"""Bind experiment resumes and evaluation reuse to their executable dependencies."""
import pathlib
from contracts import sha

HERE = pathlib.Path(__file__).resolve().parent
LOCAL = (
    'contracts.py', 'runtime_state.py', 'optimizers.py', 'train_continuation.py',
    'prepare_experiment.py', 'planning.py', 'evaluate_panel.py', 'gpu_preflight.py', 'launch.py',
    'vendor/polora/polora/optim.py', 'vendor/polora/polora/utils.py',
)
EXTERNAL = (
    '/data/work/asr-pilot/long-context-pilot/train.py',
    '/data/work/asr-pilot/long-context-pilot/model_paths.py',
    '/data/work/asr-pilot/optimized_kernels.py',
    '/data/work/asr-evaluation/periodic/code/evaluate.py',
    '/data/work/asr-evaluation/periodic/code/number_wer.py',
    '/data/work/asr-datasets/qda-50h-v2/code/qda_release_dataset.py',
)

def runtime_hashes():
    return {str(p): sha(p) for p in [*(HERE / f for f in LOCAL), *(pathlib.Path(f) for f in EXTERNAL)]}

def evaluation_hashes():
    paths = [HERE / 'evaluate_panel.py', HERE / 'runtime_state.py']
    paths += [pathlib.Path(p) for p in EXTERNAL if 'evaluation/' in p or 'model_paths.py' in p or 'qda_release_dataset.py' in p]
    return {str(p): sha(p) for p in paths}
