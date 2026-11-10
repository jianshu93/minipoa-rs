//! Reproducible single-thread minimizer microbenchmark, including a scalar
//! correctness reference from the same dependency. Timings are descriptive,
//! not assertions: hardware, CPU frequency, and build flags affect results.
use simd_minimizers::packed_seq::{PackedSeqVec, SeqVec};
use std::{hint::black_box, time::Instant};

fn main() {
    let n = std::env::args()
        .nth(1)
        .map(|n| n.parse::<usize>().expect("DNA length"))
        .unwrap_or(1_000_000);
    let rounds = std::env::args()
        .nth(2)
        .map(|n| n.parse::<usize>().expect("rounds"))
        .unwrap_or(15)
        .max(1);
    let mut state = 42u64;
    let dna: Vec<_> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            b"ACGT"[(state & 3) as usize]
        })
        .collect();
    let packed = PackedSeqVec::from_ascii(&dna);
    let builder = simd_minimizers::minimizers(19, 10);
    let scalar = builder.run_scalar_once(packed.as_slice());
    let simd = builder.run_once(packed.as_slice());
    assert_eq!(
        simd, scalar,
        "SIMD positions must equal the dependency's scalar reference"
    );
    let sketch = minipoa::minimizer::sketch(&dna, 19, 10);
    assert_eq!(
        sketch
            .iter()
            .map(|m| (m.end + 1 - m.span) as u32)
            .collect::<Vec<_>>(),
        simd
    );
    let mut positions = Vec::new();
    let start = Instant::now();
    for _ in 0..rounds {
        positions.clear();
        black_box(builder.run_scalar(black_box(packed.as_slice()), &mut positions));
    }
    let scalar_ns = start.elapsed().as_nanos() as f64 / (rounds * n) as f64;
    let start = Instant::now();
    for _ in 0..rounds {
        positions.clear();
        black_box(builder.run(black_box(packed.as_slice()), &mut positions));
    }
    let simd_ns = start.elapsed().as_nanos() as f64 / (rounds * n) as f64;
    let start = Instant::now();
    for _ in 0..rounds {
        black_box(minipoa::minimizer::sketch(black_box(&dna), 19, 10));
    }
    let sketch_ns = start.elapsed().as_nanos() as f64 / (rounds * n) as f64;
    println!(
        "DNA bases: {n}; rounds: {rounds}; k: 19; w: 10; minimizers: {}",
        simd.len()
    );
    println!("Scalar dependency extraction: {scalar_ns:.3} ns/base");
    println!("SIMD dependency extraction: {simd_ns:.3} ns/base");
    println!(
        "SIMD / scalar extraction speedup: {:.2}x",
        scalar_ns / simd_ns
    );
    println!("Full Rust sketch (DNA validation, packing, extraction, exact values): {sketch_ns:.3} ns/base");
}
