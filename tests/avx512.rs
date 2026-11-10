//! Public workflow checks shared by the eight- and sixteen-lane builds.
use minipoa::{align_sequences, Params, Sequence};
use std::collections::{HashMap, HashSet};

fn dna(length: usize) -> Vec<u8> {
    let mut state = 0x5a71_b301_u64;
    (0..length)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            b"ACGT"[(state >> 32) as usize % 4]
        })
        .collect()
}

fn records(sequences: Vec<Vec<u8>>) -> Vec<Sequence> {
    sequences
        .into_iter()
        .enumerate()
        .map(|(index, seq)| Sequence {
            name: format!("read{index}"),
            seq,
            ..Sequence::default()
        })
        .collect()
}

fn validate_msa(output: &str, reads: &[Sequence]) {
    let mut rows = HashMap::new();
    let mut lines = output.lines();
    while let Some(header) = lines.next() {
        let name = header
            .strip_prefix('>')
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap();
        rows.insert(name, lines.next().unwrap());
    }
    assert_eq!(rows.len(), reads.len());
    assert!(
        rows.values()
            .map(|row| row.len())
            .collect::<HashSet<_>>()
            .len()
            <= 1
    );
    for read in reads {
        assert_eq!(
            rows[read.name.as_str()].replace('-', "").as_bytes(),
            read.seq
        );
    }
}

fn validate_gfa(output: &str, reads: &[Sequence]) {
    let mut nodes = HashMap::new();
    let mut links = HashSet::new();
    let mut paths = HashMap::new();
    for line in output.lines() {
        let fields = line.split('\t').collect::<Vec<_>>();
        match fields[0] {
            "S" => {
                nodes.insert(fields[1], fields[2]);
            }
            "L" => {
                links.insert((fields[1], fields[3]));
            }
            "P" => {
                let path = if fields[2] == "*" {
                    Vec::new()
                } else {
                    fields[2]
                        .split(',')
                        .map(|item| item.strip_suffix('+').unwrap())
                        .collect()
                };
                paths.insert(fields[1], path);
            }
            _ => {}
        }
    }
    assert_eq!(paths.len(), reads.len());
    for read in reads {
        let path = &paths[read.name.as_str()];
        let sequence = path.iter().map(|id| nodes[id]).collect::<String>();
        assert_eq!(sequence.as_bytes(), read.seq);
        assert!(path
            .windows(2)
            .all(|edge| links.contains(&(edge[0], edge[1]))));
    }
}

#[test]
fn version_identifies_the_compiled_backend() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_minipoa"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains(env!("CARGO_PKG_VERSION")));
    assert!(text.contains(&format!("DP backend: {}", minipoa::dp_backend())));
    if cfg!(feature = "avx512") && !cfg!(feature = "scalar-dp") {
        assert!(minipoa::dp_backend().contains("16 x i32"));
        assert_eq!(
            minipoa::dp_backend().starts_with("avx512-native"),
            cfg!(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "avx512f"
            ))
        );
    }
}

#[test]
fn lane_boundary_workflows_preserve_every_read() {
    for length in [
        1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79,
    ] {
        let reference = dna(length);
        let split = length / 2;
        let mut insertion = reference[..split].to_vec();
        insertion.extend_from_slice(b"TGCATGCA");
        insertion.extend_from_slice(&reference[split..]);
        let deletion = reference
            .iter()
            .enumerate()
            .filter_map(|(i, &base)| (i % 5 != 2).then_some(base))
            .collect();
        let reads = records(vec![reference, insertion, deletion, Vec::new()]);
        for adaptive in [false, true] {
            let params = Params {
                b: 1,
                f: 40,
                adaptive,
                ..Params::default()
            };
            let mut graph = align_sequences(&params, &reads).unwrap();
            validate_msa(&graph.msa(&params, &reads).unwrap(), &reads);
            validate_gfa(&graph.gfa(&params, &reads).unwrap(), &reads);
        }
    }
}

#[test]
fn seeded_sixteen_lane_workflow_is_identical_across_worker_counts() {
    let reference = dna(1409);
    let mut insertion = reference[..417].to_vec();
    insertion.extend_from_slice(&dna(25));
    insertion.extend_from_slice(&reference[417..]);
    let mut deletion = reference[..803].to_vec();
    deletion.extend_from_slice(&reference[820..]);
    let reads = records(vec![reference, insertion, deletion]);
    for progressive in [false, true] {
        let mut params = Params {
            seeding: true,
            k: 7,
            w: 3,
            tr_dist: 0,
            win_max: 80,
            win_band: 24,
            progressive,
            threads: 1,
            ..Params::default()
        };
        let mut one = align_sequences(&params, &reads).unwrap();
        let expected = (
            one.consensus(&params),
            one.msa(&params, &reads).unwrap(),
            one.gfa(&params, &reads).unwrap(),
        );
        for threads in [2, 4, 16] {
            params.threads = threads;
            let mut many = align_sequences(&params, &reads).unwrap();
            assert_eq!(many.consensus(&params), expected.0);
            assert_eq!(many.msa(&params, &reads).unwrap(), expected.1);
            assert_eq!(many.gfa(&params, &reads).unwrap(), expected.2);
        }
    }
}
