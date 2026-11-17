#!/usr/bin/env python3
"""Benchmark BLAKE3/SHA256 at b=15 and save the six median rows to flock.csv."""
import argparse
import csv
import io
from pathlib import Path
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ['backend', 'hash', 'log2_batches', 'packed_mu', 'r', 'security_policy']
METRICS = ['witness_ms', 'commit_ms', 'zerocheck_ms', 'lincheck_ms', 'open_ms',
           'rs_open_ms', 'total_prove_ms', 'k_hashes_per_s', 'verify_ms', 'proof_kb']
BACKENDS = [('hyperfold-weighted', '4'), ('hyperfold-weighted', '8'), ('ligerito', '2')]


def summarize(trials):
    row = {key: trials[0][key] for key in IDENTITY}
    for metric in METRICS:
        values = []
        for trial in trials:
            if metric == 'rs_open_ms':
                # Ligerito already includes ring-switch in open_ms.
                ring = trial['ring_switch_ms']
                value = float(trial['open_ms']) + (0 if ring == 'fused-in-open' else float(ring))
            else:
                value = float(trial[metric])
            values.append(value)
        row[metric + '_median'] = statistics.median(values)
    row['ring_switch_ms_median'] = (
        'fused-in-open' if row['backend'] == 'ligerito'
        else statistics.median(float(t['ring_switch_ms']) for t in trials)
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/flock.csv')
    args = parser.parse_args()
    subprocess.run(['cargo', 'build', '--release', '--locked', '--bin', 'bench_flock'],
                   cwd=ROOT, check=True)
    summaries = []
    for hash_name in ['blake3', 'sha256']:
        print(f'Running {hash_name}: b=15, r=4/8, 32 threads, 1 warmup + 3 trials', flush=True)
        result = subprocess.run(
            [str(ROOT / 'target/release/bench_flock'), '--hash', hash_name,
             '--log2-batches', '15', '--r', '4,8', '--threads', '32',
             '--warmup', '1', '--iters', '3'],
            cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True,
        )
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert len(rows) == 9
        assert all(r['hash'] == hash_name and r['log2_batches'] == '15' for r in rows)
        for backend, arity in BACKENDS:
            trials = [r for r in rows if r['backend'] == backend and r['r'] == arity]
            assert len(trials) == 3 and {t['trial'] for t in trials} == {'0', '1', '2'}
            summaries.append(summarize(trials))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=args.out.parent, newline='', delete=False) as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summaries)
    Path(output.name).replace(args.out)
    print(f'Saved {len(summaries)} configurations to {args.out}')


if __name__ == '__main__':
    main()
