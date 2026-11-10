#!/usr/bin/env python3
"""Compare the Rust CLI to pinned upstream; verify MSA/GFA sequence preservation.
Only unseeded cases require byte parity: SIMD minimizers use a different hash.
"""
import argparse, gzip, json, random, subprocess, tempfile, time
from pathlib import Path


def call(binary, path, flags, *, rust=True):
    # The Rust CLI requires --input; upstream C++ still uses a positional file.
    input_args = ['--input', str(path)] if rust else [str(path)]
    result = subprocess.run([str(binary), *input_args, *flags], capture_output=True, timeout=180)
    if result.returncode:
        raise AssertionError(f"{binary} {flags}: exit {result.returncode}: {result.stderr.decode(errors='replace')}")
    return result.stdout


def sequences(output):
    return [line for line in output.decode().splitlines() if not line.startswith('>')]


def validate_msa(output, expected):
    lines=output.decode().splitlines()
    rows={lines[i][1:].split()[0]:lines[i+1] for i in range(0,len(lines),2)}
    assert len(rows)==len(expected), 'MSA omitted a read'
    assert len(set(map(len,rows.values())))<=1, 'unequal MSA widths'
    for name, seq in expected.items():
        assert rows[name].replace('-','')==seq, f'MSA sequence changed: {name}'


def validate_gfa(output, expected):
    nodes={}; edges=set(); paths={}
    for line in output.decode().splitlines():
        f=line.split('\t')
        if f[0]=='S': nodes[f[1]]=f[2]
        elif f[0]=='L':
            assert f[2]==f[4]=='+' and f[5]=='0M'
            edges.add((f[1],f[3]))
        elif f[0]=='P': paths[f[1]]=[] if not f[2] else [s[:-1] for s in f[2].split(',')]
    assert len(paths)==len(expected), 'GFA omitted a read'
    for name,path in paths.items():
        assert ''.join(nodes[n] for n in path)==expected[name], f'GFA path changed: {name}'
        assert all((a,b) in edges for a,b in zip(path,path[1:])), 'missing GFA link'


def mutate(rng, seq):
    out=[]
    for c in seq:
        r=rng.random()
        if r<0.04: continue
        if r<0.10: out.append(rng.choice('ACGT'))
        else: out.append(c)
        if rng.random()<0.04: out.append(rng.choice('ACGT'))
    return ''.join(out) or 'A'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--original', type=Path, required=True)
    ap.add_argument('--rust',type=Path,required=True)
    ap.add_argument('--cases',type=int,default=100)
    ap.add_argument('--seed',type=int,default=7319)
    ap.add_argument('--json',type=Path)
    args=ap.parse_args(); args.original=args.original.resolve();args.rust=args.rust.resolve()
    rng=random.Random(args.seed); root=Path(__file__).resolve().parents[1]
    comparisons=0; invariant_checks=0; failures=[]; started=time.monotonic()
    fixtures=[['ACGT'], ['ACGT','ACGT','ACGT'],['ACGT','AGT','ACGGT','ACCT'],['AAAA','AAA','AAAAA'],['NACGTN','NACCTN','NNACGTN'],['A','C','G','T'],['ACGT','TGCA','GATTACA']]
    for _ in range(args.cases):
        base=''.join(rng.choice('ACGT') for _ in range(rng.randint(1,180)))
        fixtures.append([base]+[mutate(rng,base) for _ in range(rng.randint(1,7))])
    with tempfile.TemporaryDirectory(prefix='minipoa-diff-') as tmp:
        tmp=Path(tmp)
        for index,reads in enumerate(fixtures):
            path=tmp/'reads.fa'; path.write_text(''.join(f'>r{i}\n{s}\n' for i,s in enumerate(reads)))
            expected={f'r{i}':s for i,s in enumerate(reads)}
            variants=[[],['-f0'],['-B'],['-M3','-X5','-O6','-E1']]
            if index<10: variants.append(['-m',str(root/'test-data/HOXD70.mtx'),'-O105','-E12'])
            for variant in variants:
                for mode in range(3):
                    flags=[*variant,f'-r{mode}']
                    old=call(args.original,path,flags,rust=False);new=call(args.rust,path,flags);comparisons+=1
                    if old!=new:
                        failures.append({'case':index,'flags':flags,'reads':reads,'original':old.decode(),'rust':new.decode()})
                    if mode==1: validate_msa(new,expected);invariant_checks+=1
                    if mode==2: validate_gfa(new,expected);invariant_checks+=1
            if index<25:
                for flags in (['-S','-k5','-w3','--tr-dist=0'],['-S','-p','-k5','-w3','--tr-dist=0','-t4']):
                    validate_msa(call(args.rust,path,[*flags,'-r1']),expected)
                    validate_gfa(call(args.rust,path,[*flags,'-r2']),expected)
                    invariant_checks+=2
            if index%25==0: print(f'{index+1}/{len(fixtures)} fixtures; {comparisons} comparisons; {len(failures)} differences',flush=True)
        # The supplied mitochondrial data exercises longer bands and seeded windows.
        data=root/'test-data/mtDNA.fasta'
        records={};name=None
        for line in data.read_text().splitlines():
            if line.startswith('>'): name=line[1:].split()[0];records[name]=''
            else: records[name]+=line.upper()
        for flags in ([],['-B'],['-S'],['-S','-p','-t4']):
            for mode in range(3):
                new=call(args.rust,data,[*flags,f'-r{mode}'])
                if mode==1: validate_msa(new,records);invariant_checks+=1
                if mode==2: validate_gfa(new,records);invariant_checks+=1
                if '-S' not in flags:
                    old=call(args.original,data,[*flags,f'-r{mode}'],rust=False);comparisons+=1
                    if old!=new:failures.append({'case':'mtDNA','flags':[*flags,f'-r{mode}'],'original_bytes':len(old),'rust_bytes':len(new)})
    report={'seed':args.seed,'fixtures':len(fixtures),'exact_comparisons':comparisons,'invariant_checks':invariant_checks,'differences':len(failures),'seconds':round(time.monotonic()-started,2),'failures':failures[:20]}
    if args.json: args.json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='failures'},indent=2))
    return bool(failures)

if __name__=='__main__':raise SystemExit(main())
