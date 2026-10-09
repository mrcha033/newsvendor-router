"""Finish the saved study by reproducing the historical CPU inference device."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'src'))
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import torch
from newsvendor import sequence
from newsvendor.io import digest, jsonl, lines, read, require, write
from study_demand import check_reference, evaluate, summarize

torch.set_num_threads(4)
root = Path('results/l40s-demand-observations-v1')
report = read(root / 'partial.json')
config = report['config']
require(sorted(map(int, report['runs'])) == config['seeds'], 'All registered training must be complete')
require(not (root / 'report.json').exists(), 'Preserve completed report')
for s, run in report['runs'].items():
    for name, expected in run['rawHashes'].items():
        require(digest((root / s / name).read_bytes()) == expected, 'Changed saved run')
started = time.perf_counter()
rows = [r for r in lines(Path(config['dataset']) / 'inputs.jsonl') if r['component'] == 'retail' and r['split'] == 'dev']
ids = {r['id'] for r in rows}
labels = {r['id']: r['target'] for r in lines(Path(config['dataset']) / 'labels.jsonl') if r['id'] in ids}
development = sequence.samples(rows, labels, config['demand']['minHistory'])
model = sequence.DemandEncoder().eval()
payload = torch.load(config['deployedCheckpoint'], weights_only=True, map_location='cpu', mmap=True)
model.load_state_dict({k.removeprefix('demand.'): v for k,v in payload['weights'].items() if k.startswith('demand.')})
deployed = evaluate(model, development, config['demand']['observationNodes'])
jsonl(root / 'deployed-cpu.jsonl', deployed)
check_reference(deployed, lines(config['deployedPredictions']))
jsonl(root / 'deployed.jsonl', deployed)
report['deployed'] = {'metrics': summarize(deployed), 'device': 'cpu', 'predictionsIdentical': True, 'rawHash': digest((root / 'deployed.jsonl').read_bytes()), 'checkpointHash': digest(Path(config['deployedCheckpoint']).read_bytes())}
report['completion'] = {'reason': 'Training and matched GPU controls completed for all seeds. Final exact historical-reference check failed when CPU-origin predictions were rerun on CUDA; recompute on the original CPU device.', 'trainingRepeated': False, 'trainingCheckpointChanged': False, 'seconds': time.perf_counter()-started, 'scriptHash': digest(Path(__file__).read_bytes()), 'originalFailure': read(root / 'job.json'), 'originalRecordedSeconds': read(root / 'progress.json')['seconds'], 'peakGpuBytes': None}
write(root / 'report.json', report)
write(root / 'completion.json', {'stage': 'complete', 'allRegisteredSeeds': config['seeds'], 'cpuReferenceExact': True, 'testUsed': False})
print(report['completion'], flush=True)
