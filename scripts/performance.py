#!/usr/bin/env python3
"""Whole-program comparison with deterministic data and interleaved repetitions.

Use architecture-matched binaries where available. On the measurement ARM Mac,
C++ and x86 Rust execute through Rosetta; native ARM Rust is listed separately.
Unseeded outputs must match. Seeded minimizer schemes may differ.
"""
import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path


def reads(path):
    result = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            result.append("")
        else:
            result[-1] += line.strip()
    return result


def generate(path, length, count, seed):
    rng = random.Random(seed)
    template = "".join(rng.choice("ACGT") for _ in range(length))
    sequences = [template]
    for _ in range(count - 1):
        sequence = []
        for base in template:
            value = rng.random()
            if value < 0.006:
                continue
            sequence.append(rng.choice("ACGT") if value < 0.026 else base)
            if rng.random() < 0.006:
                sequence.append(rng.choice("ACGT"))
        sequences.append("".join(sequence))
    path.write_text("".join(f">read{i}\n{s}\n" for i, s in enumerate(sequences)))


def execute(binary, dataset, flags, *, rust=True):
    input_args = ["--input", str(dataset)] if rust else [str(dataset)]
    started = time.perf_counter()
    run = subprocess.run(
        [str(binary), *input_args, *flags], capture_output=True, timeout=300
    )
    elapsed = time.perf_counter() - started
    if run.returncode:
        raise RuntimeError(
            f"{binary}: {run.returncode}: {run.stderr.decode(errors='replace')}"
        )
    return elapsed, hashlib.sha256(run.stdout).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp", type=Path, required=True)
    parser.add_argument("--rust-x86", type=Path)
    parser.add_argument("--rust-native", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    root = Path(__file__).resolve().parents[1]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    short = args.data_dir / "synthetic-3kb-24.fa"
    long = args.data_dir / "synthetic-10kb-12.fa"
    generate(short, 3000, 24, 2141)
    generate(long, 10000, 12, 8171)
    binaries = {
        "cpp_x86": args.cpp.resolve(),
        "rust_native_arm": args.rust_native.resolve(),
    }
    if args.rust_x86:
        binaries["rust_x86"] = args.rust_x86.resolve()
    report = {
        "platform": platform.platform(),
        "rounds": args.rounds,
        "warmups": 1,
        "threads": 1,
        "timing": "whole-process wall seconds, including I/O and startup",
        "notes": [
            "C++ and x86 Rust run through Rosetta on the measurement ARM Mac; these are not native x86 hardware timings.",
            "C++ OpenMP shim forces one thread.",
            "Native ARM Rust vs emulated C++ is not an architecture-matched comparison.",
            "Seeded minimizer schemes differ; no exact output parity is required for seeded timing.",
        ],
        "binary_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in binaries.items()
        },
        "rows": [],
    }
    for dataset in [root / "test-data/mtDNA.fasta", short, long]:
        sequences = reads(dataset)
        for mode, flags in [("unseeded", ["-t1"]), ("seeded", ["-S", "-t1"])]:
            samples = {name: [] for name in binaries}
            digests = {}
            for name, binary in binaries.items():
                _, digests[name] = execute(binary, dataset, flags, rust=name != "cpp_x86")
            for repeat in range(args.rounds):
                names = list(binaries)
                pivot = repeat % len(names)
                names = names[pivot:] + names[:pivot]
                for name in names:
                    elapsed, digest = execute(binaries[name], dataset, flags, rust=name != "cpp_x86")
                    assert digest == digests[name], f"nondeterministic output: {name}"
                    samples[name].append(elapsed)
            same = len(set(digests.values())) == 1
            if mode == "unseeded":
                assert same, "unseeded outputs differ"
            medians = {
                name: round(statistics.median(times), 6)
                for name, times in samples.items()
            }
            row = {
                "dataset": dataset.name,
                "reads": len(sequences),
                "bases": sum(map(len, sequences)),
                "mode": mode,
                "flags": flags,
                "median_seconds": medians,
                "samples_seconds": samples,
                "output_sha256": digests,
                "all_outputs_identical": same,
                "rust_native_time_div_cpp_time": round(
                    medians["rust_native_arm"] / medians["cpp_x86"], 3
                ),
            }
            if "rust_x86" in medians:
                row["rust_x86_time_div_cpp_time"] = round(
                    medians["rust_x86"] / medians["cpp_x86"], 3
                )
            report["rows"].append(row)
            args.json.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ("samples_seconds", "output_sha256")}
                ),
                flush=True,
            )
    print("Saved", args.json)


if __name__ == "__main__":
    main()
