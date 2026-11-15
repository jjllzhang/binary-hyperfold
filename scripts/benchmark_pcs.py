#!/usr/bin/env python3
"""One configuration per process; raw CSV outputs. No data selection or pooling."""
import argparse, os, pathlib, subprocess
p=argparse.ArgumentParser();p.add_argument('--mu',default='20');p.add_argument('--r',default='2,4,8');p.add_argument('--c',default='2,4');p.add_argument('--threads',default='32');p.add_argument('--out',required=True);a=p.parse_args()
out=pathlib.Path(a.out);out.mkdir(exist_ok=False,parents=True)
env=os.environ.copy();env['RUSTFLAGS']='-C target-cpu=native'
subprocess.run(['cargo','build','--release','--bin','bench_pcs'],env=env,check=True)
for mu in a.mu.split(','):
 for r in a.r.split(','):
  for c in a.c.split(','):
   name=f'mu{mu}-r{r}-c{c}'
   with (out/(name+'.log')).open('w') as log:
    subprocess.run(['target/release/bench_pcs','--mu',mu,'--r',r,'--c',c,'--threads',a.threads,'--csv',str(out/(name+'.csv'))],stdout=log,stderr=log,check=True)
