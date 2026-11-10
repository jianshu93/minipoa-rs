#!/usr/bin/env python3
"""Deterministic long-read, repeat, RNA, thread and GFA round-trip stress checks."""
import argparse,json,random,tempfile
from pathlib import Path
from differential import call,validate_msa,validate_gfa

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--rust',type=Path,required=True);ap.add_argument('--cases',type=int,default=20);ap.add_argument('--json',type=Path)
    a=ap.parse_args();binary=a.rust.resolve();r=random.Random(9917);checks=0
    with tempfile.TemporaryDirectory(prefix='minipoa-seeded-') as tmp:
        tmp=Path(tmp)
        for case in range(a.cases):
            base=''.join(r.choice('ACGT') for _ in range(r.randint(800,2400)))
            if case%4==0:base=base[:200]+'ATAT'*100+base[200:]
            reads=[base,base[:350]+''.join(r.choice('ACGT') for _ in range(160))+base[350:],base[:250]+base[430:]]
            seq=''.join(r.choice('ACGT') if r.random()<.03 else c for c in base)
            reads.append(seq[:500]+'NNNNN'+seq[500:])
            if case%3==0:reads=[s.replace('T','U') for s in reads]
            path=tmp/'input.fa';path.write_text(''.join(f'>r{i}\n{s.lower()}\n' for i,s in enumerate(reads)))
            expected={f'r{i}':s for i,s in enumerate(reads)}
            for progressive in (False,True):
                flags=['-S','-k11','-w5','--win-max=100','--win-band=40','--max-occ=20','--tr-dist=0']+(['-p'] if progressive else [])
                for mode in (1,2):
                    one=call(binary,path,[*flags,'-t1',f'-r{mode}']);many=call(binary,path,[*flags,'-t4',f'-r{mode}'])
                    assert one==many,'worker count changed output';checks+=1
                    (validate_msa if mode==1 else validate_gfa)(one,expected);checks+=1
            # Reimport native GFA and add a fifth read, verifying every old/new path.
            gfa=tmp/'graph.gfa';gfa.write_bytes(call(binary,path,['-r2']))
            addition=tmp/'new.fa';addition.write_text(f'>extra\n{reads[0]}\n')
            output=call(binary,addition,['-i',str(gfa),'-S','-t4','-r2'])
            validate_gfa(output,{**expected,'extra':reads[0]});checks+=1
            print(f'{case+1}/{a.cases} long-read fixtures; {checks} checks',flush=True)
    report={'seed':9917,'fixtures':a.cases,'checks':checks,'failures':0}
    if a.json:a.json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
