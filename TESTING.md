# Previous eight-lane validation report

This report describes the baseline copied into this separate AVX-512 source tree. New sixteen-lane validation results are recorded in [AVX512-VALIDATION.md](AVX512-VALIDATION.md).

Validated on November 9, 2026. Upstream reference: minipoa 1.5.3, commit `0fc1e2a44839e2b6e978591fd7085c11ed95474a`. No automated tests were present in upstream; its mtDNA and HOXD70 fixtures are included with the port.

## Automated checks

The Rust suite has **48 tests** (25 unit, 20 I/O and CLI integration, 3 regressions), plus a compiled library documentation example.

- 1,100 deterministic sequence pairs checked against an independent full affine-gap DP scorer. These cover varied scoring, ambiguous bases, empty queries, and zero gap costs.
- 560 direct SIMD i32 versus scalar i64 traceback comparisons on branched DAGs, varied scoring and bands; 2,500 affine-gap vector prefix scan cases; score-overflow fallback and worker-local arena reuse/restoration checks.
- Four Rayon pool tests cover worker bounds, ordering, pool reuse/growth, and deterministic error propagation.
- Static/adaptive bands, weighted graph consensus, cycles, traceback boundaries, seed window splitting, and identical seeded alignment across 1/2/4/16 workers.
- SIMD minimizer positions compared with the dependency's scalar reference over multiple lengths, k values through 31, and windows through 32767.
- Exact minimizer values independently reconstructed from the original bases, including repeated sequences. Internal truncated ntHash ranking keys are never used as anchor identities. Super-window rehashing is unnecessary because the port compares lossless 2k-bit values in u64.
- Repeat intervals/filters, occurrence filters, chaining, progressive ordering, RNA and ambiguous-run boundaries.
- FASTA/FASTQ, comments, CRLF, empty reads, multiline qualities, gzip concatenation/corruption, stdin, all CLI flags, invalid arguments, nucleotide/protein matrices, GFA round trips, graph extension, missing references, repeated-node paths, cycles, and unsupported orientations.
- A regression confirms rendering an MSA leaves the library graph ready for further alignment.
- `cargo fmt --check` and Clippy with warnings denied.

## Build configurations executed

| Configuration | Result |
| --- | --- |
| Native ARM NEON, debug and release, Rust 1.102 nightly | 48 tests passed |
| Native ARM NEON, Rust 1.89.0 minimum supported version | 48 tests passed |
| ARM Rust 1.89.0 with `scalar,scalar-dp` fallback features | 48 tests passed |
| x86_64 AVX2 (`+avx2`, LTO disabled), release, Rust 1.89.0, through Rosetta | 48 tests passed |
| x86_64 baseline CPU without AVX2, `scalar,scalar-dp` features, Rust 1.89.0, through Rosetta | 48 tests passed |
| Library documentation example | Passed |

These are macOS runs. Linux/Windows execution and native x86 hardware execution have not been performed locally.

## C++ differential tests

The final deterministic run uses seed 9027, 307 short-read fixtures (300 generated plus 7 hand-written), and the supplied 393-read mtDNA dataset.

| Check | Result |
| --- | --- |
| Exact C++/Rust output comparisons | 3,720 passed, zero differences |
| MSA/GFA sequence and topology checks | 2,584 passed |
| Longer seeded fixtures, including repeats, 160-base insertions, 180-base deletions, N, RNA, guide order, worker counts, GFA extension | 20 fixtures, 180 checks passed |

Exact comparison covers consensus, MSA and GFA under default bands, full DP, adaptive bands, custom scores, and HOXD70. Seeded runs use a different minimizer scheme and are checked for sequence preservation, graph links, determinism, and thread-count equivalence rather than byte identity with C++.

The validation scripts can generate machine-readable reports with `--json`; generated reports are not included in the source tree. An earlier independent run (seed 7319) also passed 1,320 comparisons and 984 consistency checks; it is not added to the final counts above.

The C++ reference executes its original AVX2 DP. On this ARM Mac it runs as x86_64 through Rosetta. A small header supplies `omp_set_num_threads` and `omp_get_thread_num` to force one worker; the alignment algorithms and scoring code are unchanged. Timing comparisons through Rosetta appear in PERFORMANCE.md and do not establish native x86 performance.

## Reproduce

From the crate directory:

```sh
RUSTFLAGS="-C target-cpu=native" cargo test --locked --all-targets
RUSTFLAGS="-C target-cpu=native" cargo test --locked --release --all-targets
RUSTFLAGS="-C target-cpu=native" cargo test --locked --doc
cargo test --locked --all-targets --features scalar,scalar-dp
cargo fmt --check
RUSTFLAGS="-C target-cpu=native" cargo clippy --locked --all-targets -- -D warnings
RUSTFLAGS="-C target-cpu=native" cargo build --locked --release
python3 scripts/build_oracle.py /tmp/minipoa-oracle
python3 scripts/differential.py --original /tmp/minipoa-oracle/minipoa --rust target/release/minipoa --cases 300 --seed 9027 --json target/differential-results.json
python3 scripts/stress_seeded.py --rust target/release/minipoa --cases 20 --json target/seeded-stress-results.json
RUSTFLAGS="-C target-cpu=native" cargo run --locked --release --example minimizer_bench -- 1000000 20
```

Building the oracle requires git, a C/C++ compiler, zlib, AVX2, and Rosetta when running on Apple Silicon. Python checks require Python 3 and no third-party Python packages. The provided GitHub Actions workflow adds Linux/macOS stable/MSRV jobs; those hosted jobs have not been run here.

## Performance and limits

The minimizer microbenchmark checks identical SIMD/scalar positions before timing. For 1,000,000 DNA bases, k=19, w=10, 20 rounds, one Apple Silicon thread:

| Operation | Time |
| --- | --- |
| Dependency scalar position extraction | 3.849 ns/base |
| Dependency SIMD position extraction | 1.054 ns/base |
| Full Rust sketch, including validation, packing and exact values | 1.928 ns/base |

Full-program timings before and after DP vectorization are in [PERFORMANCE.md](PERFORMANCE.md). Native ARM DP disassembly contains NEON `smax.4s`, `add.4s`, and `ext.16b` instructions.

The extraction-only speedup was **3.65×**. Timings vary with hardware and compiler. This does not establish end-to-end speed or memory parity with C++. The DP fast path uses eight i32 lanes through AVX2 or NEON with a conservative overflow guard and scalar i64 fallback. Worker-local matrix arenas are reused, and a persistent Rayon pool processes windows. Read paths and formatted output are retained in memory. Chromosome-scale, very high-depth, and other operating-system workloads remain unvalidated.

Intentional upstream corrections: zero-gap-open traceback succeeds, invalid inputs return errors, single-base GFA path segments and empty GFA paths are preserved, explicit GFA links are honored, and the requested minimizer filter ratio is retained. Reverse GFA orientation and nonzero overlaps are explicitly unsupported.
