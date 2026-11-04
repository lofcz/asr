"""Resume is tied to the exact executable dependencies and immutable model revision."""
import pathlib
from contracts import sha
HERE=pathlib.Path(__file__).resolve().parent


def runtime_hashes():
    local=('contracts.py','targets.py','events.py','gates.py','prepare.py','model.py','train_restart.py','runtime_state.py','schedule.py','dataset.py','parent_state.py','checkpoint_archive.py','disk_retention.py','run_full.py','long_evaluation.py')
    external=('/data/work/asr-pilot/long-context-pilot/train.py','/data/work/asr-pilot/long-context-pilot/model_paths.py',
              '/data/work/asr-pilot/optimized_kernels.py')
    return {str(p):sha(p) for p in [*(HERE/n for n in local),*(pathlib.Path(n) for n in external)]}


def evaluation_hashes():
    paths=[HERE/'final_report.py',HERE/'evaluate_panel.py',HERE/'gates.py',HERE/'scoring.py',pathlib.Path('/data/work/asr-evaluation/periodic/code/evaluate.py'),
           pathlib.Path('/data/work/asr-evaluation/periodic/code/number_wer.py')]
    return {str(p):sha(p) for p in paths}
