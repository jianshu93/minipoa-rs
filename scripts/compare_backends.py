#!/usr/bin/env python3
"""Check exact eight/sixteen-lane output parity, then time both whole programs.

This script can also compare portable sixteen-lane builds on a non-AVX512 host.
Use --expect-native when testing actual AVX512 hardware; global CPU target flags
are a deployment requirement, so only execute binaries supported by the host.
"""
import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from performance import generate


def execute(binary, path, flags):
    started = time.perf_counter()
    result = subprocess.run(
        [str(binary), "--input", str(path), *flags], capture_output=True, timeout=180
    )
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise AssertionError(
            f"{binary} {flags}: exit {result.returncode}: "
            f"{result.stderr.decode(errors='replace')}"
        )
    return result.stdout, elapsed


def validate_msa(output, expected):
    lines = output.decode().splitlines()
    assert len(lines) == 2 * len(expected), "MSA omitted a read"
    rows = {
        lines[index][1:].split()[0]: lines[index + 1]
        for index in range(0, len(lines), 2)
    }
    assert len(set(map(len, rows.values()))) <= 1, "unequal MSA widths"
    for name, sequence in expected.items():
        assert rows[name].replace("-", "") == sequence, f"MSA changed {name}"


def validate_gfa(output, expected):
    nodes, edges, paths = {}, set(), {}
    for line in output.decode().splitlines():
        fields = line.split("\t")
        if fields[0] == "S":
            nodes[fields[1]] = fields[2]
        elif fields[0] == "L":
            assert fields[2] == fields[4] == "+" and fields[5] == "0M"
            edges.add((fields[1], fields[3]))
        elif fields[0] == "P":
            paths[fields[1]] = (
                [] if fields[2] == "*" else [item[:-1] for item in fields[2].split(",")]
            )
    assert len(paths) == len(expected), "GFA omitted a read"
    for name, path in paths.items():
        assert "".join(nodes[node] for node in path) == expected[name], f"GFA changed {name}"
        assert all(edge in edges for edge in zip(path, path[1:])), "missing GFA link"


def variants(sequence):
    middle = len(sequence) // 2
    insertion = sequence[:middle] + "TGCATGCA" + sequence[middle:]
    deletion = "".join(base for i, base in enumerate(sequence) if i % 11 != 3)
    return [sequence, insertion, deletion]


def fixtures(rng, random_cases):
    yield [""]
    yield ["", "ACGT"]
    yield ["GCANCCTTG", "CTTCA"]
    yield ["ACGU", "AGU", "ACGGU"]
    for length in [1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33,
                   47, 48, 49, 63, 64, 65, 79, 80, 81, 95, 96, 97,
                   111, 127, 128, 129, 501, 1009, 1537]:
        sequence = "".join(rng.choice("ACGT") for _ in range(length))
        yield variants(sequence)
    for _ in range(random_cases):
        sequence = "".join(rng.choice("ACGTN") for _ in range(rng.randint(1, 350)))
        yield variants(sequence)


def benchmark(binary_paths, dataset, rounds):
    timings = []
    for mode, flags in [("unseeded", ["-t1"]), ("seeded", ["-S", "-t1"]),
                        ("seeded-t4", ["-S", "-t4"])]:
        samples = {name: [] for name in binary_paths}
        checksums = {}
        for name, binary in binary_paths.items():
            output, _ = execute(binary, dataset, flags)
            checksums[name] = hashlib.sha256(output).hexdigest()
        assert len(set(checksums.values())) == 1, "backend consensus differs in timing fixture"
        for repeat in range(rounds):
            names = list(binary_paths)
            if repeat % 2:
                names.reverse()
            for name in names:
                output, elapsed = execute(binary_paths[name], dataset, flags)
                assert hashlib.sha256(output).hexdigest() == checksums[name]
                samples[name].append(elapsed)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        timings.append({
            "dataset": dataset.name, "mode": mode, "flags": flags,
            "median_seconds": medians, "samples_seconds": samples,
            "avx512_time_div_avx2_time": medians["avx512"] / medians["avx2"],
            "output_sha256": checksums,
        })
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avx2", type=Path, required=True)
    parser.add_argument("--avx512", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=72619)
    parser.add_argument("--timing-rounds", type=int, default=0)
    parser.add_argument("--expect-native", action="store_true")
    parser.add_argument("--avx2-flags", default="unspecified")
    parser.add_argument("--avx512-flags", default="unspecified")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.cases < 0 or args.timing_rounds < 0:
        parser.error("case count and timing rounds must be nonnegative")
    binaries = {"avx2": args.avx2.resolve(), "avx512": args.avx512.resolve()}
    versions = {}
    for name, binary in binaries.items():
        version = subprocess.run([str(binary), "--version"], capture_output=True, check=True)
        versions[name] = version.stdout.decode().strip()
    if args.expect_native:
        assert "DP backend: avx512-native (16 x i32)" in versions["avx512"], versions["avx512"]
        assert "DP backend: avx2 (8 x i32)" in versions["avx2"], versions["avx2"]
    root = Path(__file__).resolve().parents[1]
    comparisons = invariants = count = 0
    options = [[], ["-f0"], ["-B"], ["-b1", "-f40"], ["-O0", "-E1"],
               ["-S", "-k5", "-w3", "--tr-dist=0", "--win-band=8", "--win-max=40", "-t1"],
               ["-S", "-k5", "-w3", "--tr-dist=0", "--win-band=8", "--win-max=40", "-t4"],
               ["-S", "-p", "-k5", "-w3", "--tr-dist=0", "-t4"]]
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="minipoa-backends-") as temporary:
        temporary = Path(temporary)
        path = temporary / "reads.fa"
        for index, sequences in enumerate(fixtures(random.Random(args.seed), args.cases)):
            count += 1
            path.write_text("".join(f">read{i}\n{s}\n" for i, s in enumerate(sequences)))
            rna = any("U" in sequence for sequence in sequences)
            expected = {f"read{i}": sequence.replace("T", "U") if rna else sequence
                        for i, sequence in enumerate(sequences)}
            for flags in options:
                for mode in range(3):
                    flags_mode = [*flags, f"-r{mode}"]
                    reference, _ = execute(binaries["avx2"], path, flags_mode)
                    candidate, _ = execute(binaries["avx512"], path, flags_mode)
                    assert reference == candidate, f"backend difference: case {index}, flags {flags_mode}, reads {sequences}"
                    comparisons += 1
                    if mode == 1:
                        validate_msa(candidate, expected)
                        invariants += 1
                    elif mode == 2:
                        validate_gfa(candidate, expected)
                        invariants += 1
            if index % 25 == 0:
                print(f"{count} fixtures; {comparisons} exact backend comparisons passed", flush=True)
        data = root / "test-data/mtDNA.fasta"
        for flags in [[], ["-B"], ["-S"], ["-S", "-p", "-t4"]]:
            for mode in range(3):
                left, _ = execute(binaries["avx2"], data, [*flags, f"-r{mode}"])
                right, _ = execute(binaries["avx512"], data, [*flags, f"-r{mode}"])
                assert left == right, f"mtDNA backend difference: {flags}, mode {mode}"
                comparisons += 1
        timing_rows = []
        if args.timing_rounds:
            long_reads = temporary / "synthetic-10kb-12.fa"
            generate(long_reads, 10000, 12, 8171)
            for dataset in [data, long_reads]:
                timing_rows.extend(benchmark(binaries, dataset, args.timing_rounds))
    report = {
        "platform": platform.platform(), "seed": args.seed, "random_cases": args.cases,
        "fixtures": count, "exact_comparisons": comparisons,
        "invariant_checks": invariants, "differences": 0,
        "versions": versions,
        "binary_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in binaries.items()},
        "build_flags": {"avx2": args.avx2_flags, "avx512": args.avx512_flags},
        "seconds": round(time.monotonic() - started, 3),
        "timing_rounds": args.timing_rounds, "timings": timing_rows,
        "timing_note": "Ratio below 1 means the sixteen-lane build was faster on this host. Portable sixteen-lane execution does not validate native AVX512 hardware.",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["fixtures", "exact_comparisons", "invariant_checks", "differences", "seconds"]}, indent=2))
    print("Saved", args.json)


if __name__ == "__main__":
    main()
