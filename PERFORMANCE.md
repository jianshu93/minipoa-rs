# Previous eight-lane whole-program performance

These measurements predate the separate AVX-512 kernel. They do not measure AVX-512 throughput.

Measured November 9, 2026 on macOS 15.8.1, Apple Silicon. These are full CLI timings, including input reading, alignment, consensus generation, and process startup. Every run uses one thread and consensus output. Five interleaved repetitions follow one warmup; values below are median seconds.

The new native ARM Rust build is **3.05–4.37× faster than the original Rust port** on these six workload/mode combinations. It uses AVX2/NEON score updates and insertion prefix scans, a reusable worker-local DP arena, and Rayon for seeded parallel windows.

**It is still 1.50–1.72× slower than C++ when both x86 builds run through Rosetta.** Native ARM Rust beats emulated C++ in this environment, but that comparison mixes architectures. These measurements do not confirm a speed advantage over native C++ on native x86 hardware.

| Workload / mode | C++ x86 AVX2, Rosetta (s) | Rust x86 AVX2, Rosetta (s) | Rust x86 time / C++ time | Native ARM Rust (s) | Native ARM speedup over old Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| mtDNA.fasta / unseeded | 0.115 | 0.197 | 1.71× | 0.070 | 3.49× |
| mtDNA.fasta / seeded | 0.160 | 0.240 | 1.50× | 0.094 | 3.71× |
| synthetic-3kb-24.fa / unseeded | 0.082 | 0.133 | 1.63× | 0.046 | 4.03× |
| synthetic-3kb-24.fa / seeded | 0.101 | 0.169 | 1.68× | 0.061 | 3.39× |
| synthetic-10kb-12.fa / unseeded | 0.210 | 0.357 | 1.70× | 0.121 | 4.37× |
| synthetic-10kb-12.fa / seeded | 0.142 | 0.244 | 1.72× | 0.087 | 3.05× |

All six cases produced identical consensus output across all three executables. The seeded algorithms can differ on other inputs because their minimizer schemes differ. The earlier baseline used three interleaved repetitions. Generated timing reports are not included in the source tree; the reproduction command below writes a fresh report under `target/`.

The synthetic 3 kb dataset has 24 reads (71,975 bases); the 10 kb dataset has 12 reads (119,967 bases). The included mtDNA dataset has 393 reads (146,561 bases). The benchmark script generates synthetic inputs deterministically from seeds 2141 and 8171.

## SIMD and threading verification

The DP uses `wide::i32x8`: one AVX2 register on x86 and two NEON registers on ARM. It vectorizes predecessor match/deletion transitions, substitution profile updates, affine insertion propagation with a three-stage prefix maximum, and adaptive-band score maxima. A conservative bound selects scalar i64 scoring when i32 arithmetic cannot safely represent the workload. Traceback and graph insertion retain their sequential dependencies.

Actual release DP disassembly (`minipoa::align::poa_inner`) contains:

- ARM: NEON `smax.4s`, `add.4s`, and `ext.16b`.
- x86: AVX2 `vpmaxsd` and `vpaddd` with lane permutation instructions.

SIMD is therefore present in the alignment kernel as well as in minimizer extraction. Rayon now handles independent seeded windows through a reusable bounded pool; graph insertion remains sequential. The one-thread timings above do not measure Rayon scaling. Tests cover identical results across 1/2/4/16 workers and pool reuse.

## Reproduce

Build ARM Rust with `cargo build --release --locked`. For the x86 reference comparison on this ARM Mac:

```sh
rustup target add x86_64-apple-darwin --toolchain 1.89.0
RUSTFLAGS="-C target-feature=+avx2" CARGO_PROFILE_RELEASE_LTO=false cargo +1.89.0 build --release --locked --target x86_64-apple-darwin
python3 scripts/build_oracle.py /tmp/minipoa-oracle
python3 scripts/performance.py --cpp /tmp/minipoa-oracle/minipoa --rust-x86 target/x86_64-apple-darwin/release/minipoa --rust-native target/release/minipoa --data-dir /tmp/minipoa-perf --rounds 5 --json target/performance-results.json
```

The C++ build uses `-O3 -mavx2` and its original DP code. Its OpenMP shim forces one worker. The x86 Rust build uses Rust 1.89.0, explicit AVX2, and LTO disabled. Native ARM Rust uses Rust 1.102 nightly with thin LTO. A previous x86-64-v3/thin-LTO Rust build aborted at startup under Rosetta; the cause has not been isolated, so that build is excluded from timings.

The `scalar-dp` Cargo feature forces the retained i64 DP reference for differential validation. The original extractor-only **3.65×** SIMD/scalar result remains a separate microbenchmark and is not a C++ speed comparison.
