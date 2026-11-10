#!/usr/bin/env python3
"""Build pinned upstream in a scratch directory, using one worker for parity.
On macOS ARM, compile an x86_64 AVX2 reference to execute under Rosetta.
The shim only replaces the two OpenMP worker-management calls; DP is unchanged.
"""
import argparse, concurrent.futures, platform, subprocess
from pathlib import Path

COMMIT='0fc1e2a44839e2b6e978591fd7085c11ed95474a'

def run(cmd):
    subprocess.run([str(x) for x in cmd],check=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    args=ap.parse_args(); root=args.directory.resolve();root.mkdir(parents=True,exist_ok=True)
    source=root/'source'
    if not source.exists():run(['git','clone','https://github.com/NCl3-lhd/minipoa.git',source])
    run(['git','-C',source,'checkout','--detach',COMMIT])
    include=root/'include';include.mkdir(exist_ok=True)
    (include/'omp.h').write_text('static inline void omp_set_num_threads(int n) {(void)n;}\nstatic inline int omp_get_thread_num(void) {return 0;}\n')
    obj=root/'objects';obj.mkdir(exist_ok=True)
    arch=['-arch','x86_64'] if platform.system()=='Darwin' and platform.machine()=='arm64' else []
    cc='clang' if platform.system()=='Darwin' else 'gcc'
    cxx='clang++' if platform.system()=='Darwin' else 'g++'
    flags=[*arch,'-O3','-mavx2','-DENABLE_AVX2','-I'+str(include),'-I'+str(source/'include'),'-Wno-deprecated-declarations','-Wno-unused-result']
    def build(p):
        run([cxx if p.suffix=='.cpp' else cc,*flags,'-std=c++17' if p.suffix=='.cpp' else '-std=c11','-c',p,'-o',obj/(p.name+'.o')])
    sources=list((source/'src').glob('*.cpp'))+list((source/'src').glob('*.c'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(build,sources))
    binary=root/'minipoa'
    run([cxx,*arch,'-O3',*[obj/(p.name+'.o') for p in sources],'-lz','-o',binary])
    run([binary,'--version']);print(binary)

if __name__=='__main__':main()
