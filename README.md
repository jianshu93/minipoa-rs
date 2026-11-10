# minipoa-rs
Rust reimplementation of [minipoa](https://github.com/NCl3-lhd/minipoa) algorithm.

Includes a Rust library and `minipoa` executable. The alignment and graph algorithms are implemented in Rust.

## Build and run

Requires Rust 1.89 or newer. On Apple Silicon, NEON works with the default target. On x86_64, enable AVX2 by building for the local CPU:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release --locked
./target/release/minipoa --input test-data/mtDNA.fasta > consensus.fasta
./target/release/minipoa --input test-data/mtDNA.fasta -S -r1 -t4 > alignment.fasta
./target/release/minipoa --input test-data/mtDNA.fasta -S -r2 > graph.gfa
./target/release/minipoa --input additional.fasta -i graph.gfa -S -r2 > extended.gfa
./target/release/minipoa --help
```

A binary built with `target-cpu=native` should be used on a CPU with the same instruction support. For a build that permits the dependency's portable fallback:

```sh
cargo build --release --locked --features scalar
```

The `scalar` feature permits compilation without AVX2/NEON; on supported targets the dependency may still select SIMD. It is not a force-scalar benchmark switch. Add `--features scalar-dp` to force the retained i64 DP reference kernel for validation or comparison.

## Features

- Affine-gap global sequence-to-DAG alignment with weighted deterministic traceback.
- AVX2/NEON DP score updates, affine-gap prefix scans, and band maxima through `wide::i32x8`; conservative score bounds select an i64 fallback for extreme scores.
- A reusable, contiguous DP arena per executing worker.
- Compact static and adaptive bands; `-f0` disables banding. If a band cannot produce a traceback, retry with full DP.
- Actual [`simd-minimizers`](https://github.com/rust-seq/simd-minimizers) extraction, exact packed k-mer identities, occurrence and tandem-repeat filters, sparse chaining, and drift-based windows.
- Seed windows processed by a reusable bounded Rayon pool with deterministic output order.
- Minimizer-based progressive guide order with `-p`.
- Consensus FASTA (`-r0`), MSA FASTA (`-r1`), and GFA1 (`-r2`).
- FASTA, multiline FASTQ, gzip (including concatenated members), and stdin (`-`).
- DNA, RNA, custom nucleotide/protein substitution matrices, incremental GFA input.
- GFA syntax validated using the established [`gfa`](https://docs.rs/gfa/0.10.1/gfa/) crate.

The Rust CLI uses clap 4.5 and requires `--input PATH`; positional input paths are not accepted. Read stdin with `--input -`. Use `--help` for the full option list and `--version` for the version and selected DP backend. Value-taking options accept separate values, `--option=value`, and attached short values such as `-t4`. Existing short options remain available, including `-i` for an initial GFA graph. Hyphenated long options are shown in help; legacy underscore names such as `--inc_fp`, `--gap_open`, and `--progressive_poa` remain aliases. Upstream C++ still takes a positional input path.

Scoring and alignment defaults retain upstream behavior. Positive mismatch/gap penalties are normalized to negative values. `-t` parallelizes independent seeded windows; it does not parallelize progressive read insertion.

## Library example

```rust
use minipoa::{align_sequences, Params, Sequence};

let reads = vec![
    Sequence { name: "a".into(), seq: b"ACGT".to_vec(), ..Default::default() },
    Sequence { name: "b".into(), seq: b"ACCT".to_vec(), ..Default::default() },
];
let params = Params::default();
let mut graph = align_sequences(&params, &reads)?;
let (consensus, node_ids, coverage) = graph.consensus(&params);
let msa = graph.msa(&params, &reads)?;
let gfa = graph.gfa(&params, &reads)?;
# Ok::<(), String>(())
```

Set `Params::is_rna` when using the library to request U in RNA output. `Params::matrix` controls scores in the library; changing the convenience score fields alone does not rebuild it. The CLI rebuilds the default matrix after parsing score flags. Lower-level graph fields are public for inspection; mutations must preserve DAG and alignment-column invariants. `align::align` expects individual-node topological ranks, obtainable with `graph.toposort(false)`.

## Compatibility and limits

Unseeded consensus/MSA/GFA matched the pinned AVX2 C++ reference byte for byte in 3,720 comparisons. See [TESTING.md](TESTING.md) for scope and reproducible commands.

Initial seed matching is query-wide against the complete graph consensus; windows are chosen after chaining. Exact packed u64 k-mer identities are used for this global step, so no rehashing of super-windows is needed.

Seeded and progressive results can differ: the SIMD crate uses forward ntHash selection with leftmost ties, while upstream uses a Wang hash and emits repeated ties. Protein matrices use graph DP without DNA seeding or guide reordering. The Rust implementation honors `--mm-filter-ratio`; upstream initialization silently overwrites it with 0.25.

GFA input supports forward, acyclic H/S/L/P graphs with explicit sequences and zero overlaps. Reverse-oriented graphs, inversions, nonzero overlaps, and other record kinds produce explicit errors. GFA links are respected, including graphs without read paths; upstream only reconstructs edges from P records. GFA does not carry minipoa's substitution-column groups, so exporting and reimporting a graph need not preserve MSA column layout.

Malformed FASTQ/matrices/GFA are errors. Empty read records are supported; a completely empty sequence input is an error. Empty GFA paths use `*`. A zero gap-open traceback bug inherited from C++ has been fixed. Negative match arguments are normalized to positive scores.

The fast DP kernel uses eight i32 lanes: one AVX2 vector on x86 or two NEON vectors on ARM. An overflow guard preserves exact scoring by falling back to scalar i64 DP. Read paths and final text output remain in memory, unlike upstream's temporary path file. Whole-program timing results and their architecture limits are in [PERFORMANCE.md](PERFORMANCE.md); peak-memory equivalence and chromosome-scale workloads have not been validated. As with upstream, very repetitive sequences with repeat/occurrence filtering disabled can generate many seed matches. Guide ordering uses O(N²) storage.

## Validation and benchmark

```sh
RUSTFLAGS="-C target-cpu=native" cargo test --locked --all-targets
RUSTFLAGS="-C target-cpu=native" cargo test --locked --release --all-targets
cargo test --locked --all-targets --features scalar,scalar-dp
cargo fmt --check
RUSTFLAGS="-C target-cpu=native" cargo clippy --locked --all-targets -- -D warnings
RUSTFLAGS="-C target-cpu=native" cargo run --locked --release --example minimizer_bench -- 1000000 20
```

A single-thread Apple Silicon run measured **3.65× faster minimizer extraction** than the same dependency's scalar reference (1.054 vs 3.849 ns/base). Complete extraction including validation, packing, and exact values took 1.928 ns/base. This measures extraction, not whole-aligner speed. Raw output is in [minimizer-benchmark.txt](minimizer-benchmark.txt). Full-program comparisons against C++ and the original Rust port are in [PERFORMANCE.md](PERFORMANCE.md).

MIT licensed; upstream attribution is preserved in [LICENSE](LICENSE).
