#!/usr/bin/env bash
# Native Linux AVX2 / AVX512 correctness and whole-program comparison.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash scripts/test_avx512_linux.sh [OPTIONS]
  --rounds N        Timing repetitions after one warmup (default: 3; 0 skips timings)
  --cases N         Generated correctness fixtures (default: 100)
  --output-dir DIR  Separate builds and JSON reports (default: target/avx512-experiment)
  --cpp PATH        Also compare against an existing upstream C++ minipoa executable
  -h, --help        Show this help

Requires x86_64 Linux with AVX2, AVX512F and AVX512BW, Rust >=1.89, and Python 3.
No upstream source download is performed. AVX2_RUSTFLAGS and AVX512_RUSTFLAGS
may override the controlled default CPU flags. Both builds disable LTO.
USAGE
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rounds=3
cases=100
output="$root/target/avx512-experiment"
cpp=""
while (($#)); do
    case "$1" in
        --rounds|--cases|--output-dir|--cpp)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --rounds) rounds="$2" ;;
                --cases) cases="$2" ;;
                --output-dir) output="$2" ;;
                --cpp) cpp="$2" ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ ! "$rounds" =~ ^[0-9]+$ || ! "$cases" =~ ^[0-9]+$ ]]; then
    echo "--rounds and --cases must be nonnegative integers" >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "Run this driver on x86_64 Linux; portable validation is available through compare_backends.py." >&2
    exit 2
fi
if ! awk '/^flags[[:space:]]*:/ { for (i=1; i<=NF; i++) {if ($i=="avx2") a=1; if ($i=="avx512f") f=1; if ($i=="avx512bw") b=1} found=1; exit } END {exit !(found && a && f && b)}' /proc/cpuinfo; then
    echo "This host does not report AVX2, AVX512F and AVX512BW; native AVX512 binaries must not be executed here." >&2
    exit 2
fi
command -v cargo >/dev/null
command -v rustc >/dev/null
command -v python3 >/dev/null
if [[ -n "$cpp" && ! -x "$cpp" ]]; then
    echo "--cpp must name an executable file: $cpp" >&2
    exit 2
fi
mkdir -p "$output"
output="$(cd "$output" && pwd)"
if [[ -n "$cpp" ]]; then cpp="$(cd "$(dirname "$cpp")" && pwd)/$(basename "$cpp")"; fi
host="$(rustc -vV | sed -n 's/^host: //p')"
avx2_flags="${AVX2_RUSTFLAGS:--C target-cpu=x86-64 -C target-feature=+avx2,-avx512f}"
avx512_flags="${AVX512_RUSTFLAGS:--C target-cpu=x86-64 -C target-feature=+avx2,+avx512f,+avx512bw}"
# An inherited encoded flag list takes priority over RUSTFLAGS in Cargo.
unset CARGO_ENCODED_RUSTFLAGS
rustc -vV > "$output/rustc.txt"
if command -v lscpu >/dev/null; then lscpu > "$output/cpu.txt"; fi
for backend in avx2 avx512; do
    flags="$avx2_flags"
    features=()
    if [[ "$backend" == avx512 ]]; then
        flags="$avx512_flags"
        features=(--features avx512)
    fi
    echo "Building and testing $backend with RUSTFLAGS=$flags"
    env RUSTFLAGS="$flags" CARGO_PROFILE_RELEASE_LTO=false CARGO_TARGET_DIR="$output/$backend" \
        cargo build --manifest-path "$root/Cargo.toml" --locked --release --target "$host" "${features[@]}"
    env RUSTFLAGS="$flags" CARGO_PROFILE_RELEASE_LTO=false CARGO_TARGET_DIR="$output/$backend" \
        cargo test --manifest-path "$root/Cargo.toml" --locked --release --target "$host" --all-targets "${features[@]}"
done
avx2_binary="$output/avx2/$host/release/minipoa"
avx512_binary="$output/avx512/$host/release/minipoa"
"$avx2_binary" --version
"$avx512_binary" --version
python3 "$root/scripts/compare_backends.py" \
    --avx2 "$avx2_binary" --avx512 "$avx512_binary" --expect-native \
    --cases "$cases" --timing-rounds "$rounds" \
    --avx2-flags="$avx2_flags" --avx512-flags="$avx512_flags" \
    --json "$output/backend-comparison.json"
if [[ -n "$cpp" ]]; then
    python3 "$root/scripts/differential.py" --original "$cpp" --rust "$avx512_binary" \
        --cases "$cases" --json "$output/cpp-differential.json"
fi
echo "All requested checks passed. Results: $output/backend-comparison.json"
