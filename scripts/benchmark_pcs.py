#!/usr/bin/env python3
"""One configuration per process; certify schedule before timing, keep all trials."""
import argparse, json, pathlib, subprocess
from derive_params import certify_schedule
p=argparse.ArgumentParser()
p.add_argument('--mu',default='20');p.add_argument('--r',default='2,4,8');p.add_argument('--c',default='2,4')
p.add_argument('--threads',default='32');p.add_argument('--security-bits',default='100')
p.add_argument('--base-field-bits',default='128');p.add_argument('--mu0')
p.add_argument('--distance-mode',default='analytic',choices=['analytic','primal-dual'])
p.add_argument('--matrix',default='specialized',choices=['specialized','general'])
p.add_argument('--warmup',default='1');p.add_argument('--iters',default='3')
p.add_argument('--out',required=True);a=p.parse_args()
out=pathlib.Path(a.out);out.mkdir(exist_ok=False,parents=True)
subprocess.run(['cargo','build','--release','--bin','bench_pcs'],check=True)
for mu in a.mu.split(','):
 for r in a.r.split(','):
  for c in a.c.split(','):
   name=f'mu{mu}-r{r}-c{c}'
   common=['target/release/bench_pcs','--mu',mu,'--r',r,'--c',c,'--threads',a.threads,
           '--base-field-bits',a.base_field_bits,'--security-bits',a.security_bits,
           '--distance-mode',a.distance_mode,'--matrix',a.matrix]
   if a.mu0 is not None: common+=['--mu0',a.mu0]
   schedule=out/(name+'.json')
   with (out/(name+'.log')).open('w') as log:
    subprocess.run(common+['--iters','0','--schedule-json',str(schedule)],stdout=log,stderr=log,check=True)
    certificates=[certify_schedule(row) for row in json.loads(schedule.read_text())]
    (out/(name+'-certificate.json')).write_text(json.dumps(certificates,indent=2)+'\n')
    subprocess.run(common+['--warmup',a.warmup,'--iters',a.iters,'--csv',str(out/(name+'.csv'))],stdout=log,stderr=log,check=True)
