# AVX-512 validation

Validated November 9, 2026. This is the separate 0.2.0 source tree. Native AVX-512 execution and timing have **not** been performed locally; the available host is ARM macOS. The included Linux driver performs those checks on your machine.

| Check | Result |
| --- | --- |
| Portable sixteen-lane execution on ARM, debug | 54 tests passed |
| Portable sixteen-lane execution on ARM, release | 54 tests passed |
| Default eight-lane execution in this source tree | 51 tests passed |
| Library documentation example | Passed |
| Formatting and all-feature Clippy, warnings denied | Passed |
| Native Linux x86_64 AVX-512 release cross-compilation | Passed; ELF executable produced |
| Linux AVX-512 unit/integration test executable compilation | Passed with `--no-run`; not executed |
| Rust 1.89.0 native AVX-512 code cross-compilation, x86_64 macOS | Passed; not executed |
| Exact default-eight versus portable-sixteen output comparisons | 3,276 passed; zero differences |
| Eight/sixteen MSA and GFA invariant checks | 2,176 passed |
| Portable-sixteen versus original C++ output comparisons | 3,720 passed; zero differences |
| C++ corpus MSA/GFA invariant checks | 2,584 passed |

The new tests include 2,400 sixteen-lane prefix/tail chains, including unreachable carries and large extensions; 1,080 branched graph/vector-boundary traceback comparisons; and row maxima that must exclude neighboring padding. Existing tests also cover 560 SIMD/scalar DAG tracebacks, 1,100 independent affine-score comparisons, the overflow fallback, arena reuse, Rayon worker ordering, and malformed I/O.

The backend comparison uses 136 deterministic fixtures covering lengths around 8/16 boundaries, rows and predecessor intersections offset by eight, empty/RNA reads, seeded windows, worker counts, and the supplied mtDNA dataset. Timings were disabled for portable execution. Generated backend and C++ comparison reports are not included in the source tree.

The Linux cross-build used Rust 1.102 nightly, a GNU x86_64 Linux linker, and:

```text
-C target-cpu=x86-64 -C target-feature=+avx2,+avx512f,+avx512bw
```

Inspection of `minipoa::align::poa_inner` confirms native 512-bit code: 15 ZMM `vpmaxsd` instructions, 9 ZMM `vpaddd` instructions, and 4 ZMM `vpermt2d` instructions. This verifies code generation rather than hardware execution. The compiler flags used for this cross-build are recorded above.

AVX512BW is required alongside AVX512F by the pinned `wide` 1.7.1 dependency's native compilation branches. F-only builds fail in that dependency; the supplied Linux driver checks and enables F+BW explicitly.

The original eight-lane source tree was left unchanged. The copied eight-lane helper in this separate tree uses a lower shift-padding sentinel so unreachable carries cannot introduce artificial prefix candidates. It preserves reachable scores and output compatibility, as checked above.

Run the native experiment from [AVX512.md](AVX512.md). No AVX-512 speedup is claimed until native measurements are available.
