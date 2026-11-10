use clap::error::ErrorKind;
use minipoa::{cli, io, Params};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::Command as Process,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new(extension: &str, data: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "minipoa-rs-test-{}-{}.{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            extension
        ));
        fs::write(&path, data).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn args(options: &[&str]) -> cli::Args {
    cli::parse(options.iter().copied()).unwrap()
}

#[test]
fn fasta_multiline_crlf_comments_and_empty_records() {
    let records =
        io::parse_sequences("\n>alpha annotation words\r\nAC GT\r\nN\r\n>empty\r\n>last\r\nU\r\n")
            .unwrap();
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].name, "alpha");
    assert_eq!(records[0].comment, "annotation words");
    assert_eq!(records[0].seq, b"ACGTN");
    assert!(records[1].seq.is_empty());
    assert_eq!(records[2].seq, b"U");
}

#[test]
fn fastq_multiline_quality_may_start_with_header_marker() {
    let records =
        io::parse_sequences("@alpha words\nAC\nGT\n+alpha\n@@\n++\n@beta\nNN\n+\nII\n").unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].seq, b"ACGT");
    assert_eq!(records[0].qual, b"@@++");
    assert_eq!(records[1].qual, b"II");
}

#[test]
fn malformed_sequence_files_are_errors() {
    for text in [
        "ACGT\n",
        ">\nA\n",
        "@a\nAC\n+\nI\n",
        "@a\nA\n+\nIII\n",
        "@a\nAA\n@b\nA\n+\nI\n",
        ">a\n@b\n",
    ] {
        assert!(io::parse_sequences(text).is_err(), "accepted {text:?}");
    }
    assert!(io::parse_sequences("").unwrap().is_empty());
}

#[test]
fn concatenated_gzip_members_and_magic_detection() {
    let mut bytes = Vec::new();
    for data in [b">a\nAC\n".as_slice(), b">b\nGT\n".as_slice()] {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).unwrap();
        bytes.extend(encoder.finish().unwrap());
    }
    let file = Temp::new("data", &bytes);
    let records = io::read_sequences(&file.0).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[1].seq, b"GT");
    bytes.truncate(bytes.len() - 3);
    let broken = Temp::new("gz", &bytes);
    assert!(io::read_sequences(&broken.0).is_err());
}

#[test]
fn parse_every_upstream_cli_option_and_normalize_penalties() {
    let a = args(&[
        "--input=input.fa",
        "-iold.gfa",
        "--mat_fp=score.mtx",
        "-M3",
        "-X5",
        "-O7",
        "-E1",
        "-t2",
        "-b50",
        "-f0",
        "-SBp",
        "--win-band",
        "60",
        "--win-max=800",
        "-k15",
        "-w5",
        "--mm-filter-ratio=0.4",
        "--max-occ=12",
        "--tr-dist=0",
        "-r2",
        "-V1",
    ]);
    assert_eq!(a.input, PathBuf::from("input.fa"));
    assert_eq!(a.incremental, Some("old.gfa".into()));
    assert_eq!(a.matrix, Some("score.mtx".into()));
    let p = a.params;
    assert_eq!(
        (p.match_score, p.mismatch, p.gap_open, p.gap_ext),
        (3, -5, -7, -1)
    );
    assert_eq!(
        (p.threads, p.b, p.f, p.k, p.w, p.win_band, p.win_max),
        (2, 50, 0, 15, 5, 60, 800)
    );
    assert!(p.adaptive && p.seeding && p.progressive);
    assert_eq!(p.mm_filter_ratio, 0.4);
    assert_eq!((p.max_occ, p.tr_dist, p.result, p.verbose), (12, 0, 2, 1));
    assert_eq!(p.score(0, 0), 3);
    assert_eq!(p.score(0, 1), -5);
    assert_eq!(p.score(4, 0), 0);
}

#[test]
fn cli_canonical_long_names_and_presence_flags() {
    let a = args(&[
        "--input",
        "file.fa",
        "--match",
        "2",
        "--mismatch=-3",
        "--gap-open=-5",
        "--gap-ext=-1",
        "--thread=1",
        "--band-b=3",
        "--band-f=50",
        "--ab-band",
        "--seeding",
        "--progressive-poa",
        "--k-mer=7",
        "--mm-w=2",
        "--result=1",
        "--verbose=2",
        "--inc-fp=old.gfa",
        "--mat-fp=score.mtx",
    ]);
    assert!(a.params.adaptive && a.params.seeding && a.params.progressive);
    assert_eq!((a.params.gap_open, a.params.gap_ext), (-5, -1));
    assert_eq!(
        (a.params.b, a.params.f, a.params.k, a.params.w),
        (3, 50, 7, 2)
    );
    assert_eq!(a.incremental, Some("old.gfa".into()));
    assert_eq!(a.matrix, Some("score.mtx".into()));
    assert_eq!(
        args(&["--input=-reads.fa"]).input,
        PathBuf::from("-reads.fa")
    );
    let default = args(&["--input", "file.fa"]);
    assert!(!default.params.adaptive && !default.params.seeding && !default.params.progressive);
}

#[test]
fn cli_legacy_underscore_aliases_remain_accepted() {
    let a = args(&[
        "--input=file.fa",
        "--gap_open=-5",
        "--gap_ext=-1",
        "--band_b=3",
        "--band_f=50",
        "--ab_band",
        "--progressive_poa",
        "--k_mer=7",
        "--mm_w=2",
        "--inc_fp=old.gfa",
        "--mat_fp=score.mtx",
    ]);
    assert!(a.params.adaptive && a.params.progressive);
    assert_eq!((a.params.gap_open, a.params.gap_ext), (-5, -1));
    assert_eq!(
        (a.params.b, a.params.f, a.params.k, a.params.w),
        (3, 50, 7, 2)
    );
    assert_eq!(a.incremental, Some("old.gfa".into()));
    assert_eq!(a.matrix, Some("score.mtx".into()));
}

#[test]
fn cli_negative_scores_and_valid_range_boundaries() {
    let a = args(&[
        "--input",
        "file.fa",
        "-M",
        "-3",
        "-X",
        "-5",
        "-O",
        "-7",
        "-E",
        "-1",
        "--k-mer=1",
        "--mm-w=32767",
        "--band-b=0",
        "--band-f=0",
        "--win-max=1",
        "--mm-filter-ratio=0",
        "--max-occ=0",
        "--tr-dist=0",
    ]);
    assert_eq!(
        (
            a.params.match_score,
            a.params.mismatch,
            a.params.gap_open,
            a.params.gap_ext
        ),
        (3, -5, -7, -1)
    );
    assert_eq!((a.params.k, a.params.w), (1, 32767));
    assert_eq!(
        args(&["--input=file.fa", "--k-mer=31", "--mm-filter-ratio=1"])
            .params
            .k,
        31
    );
}

#[test]
fn cli_rejects_bad_options_values_and_missing_input() {
    for values in [
        &[][..],
        &["file.fa"],
        &["--", "file.fa"],
        &["--input=file.fa", "extra.fa"],
        &["--input=file.fa", "--", "extra.fa"],
        &["--input"],
        &["--input=file.fa", "--unknown"],
        &["--input=file.fa", "-t0"],
        &["--input=file.fa", "-r3"],
        &["--input=file.fa", "-V3"],
        &["--input=file.fa", "-w0"],
        &["--input=file.fa", "-w32768"],
        &["--input=file.fa", "-k0"],
        &["--input=file.fa", "-k32"],
        &["--input=file.fa", "-b-1"],
        &["--input=file.fa", "--win-max=0"],
        &["--input=file.fa", "--mm-filter-ratio=NaN"],
        &["--input=file.fa", "--mm-filter-ratio=inf"],
        &["--input=file.fa", "--mm-filter-ratio=-0.1"],
        &["--input=file.fa", "--mm-filter-ratio=1.1"],
        &["--input=file.fa", "-M-2147483648"],
        &["--input=file.fa", "-X1000001"],
        &["--input=file.fa", "-M"],
        &["--input=file.fa", "--seeding=true"],
        &["--input=file.fa", "--seeding=false"],
        &["--input=file.fa", "--ab-band=false"],
        &["--input=file.fa", "--progressive-poa=0"],
    ] {
        let error = cli::parse(values.iter().copied()).unwrap_err();
        assert_eq!(
            error.exit_code(),
            2,
            "wrong exit code for {values:?}: {error}"
        );
        assert!(error.use_stderr());
    }
}

#[test]
fn cli_rejects_duplicate_named_options() {
    for values in [
        vec!["--input=a.fa", "--input=b.fa"],
        vec!["--input=a.fa", "-t1", "--thread=2"],
        vec!["--input=a.fa", "--gap-open=-4", "--gap_open=-5"],
        vec!["--input=a.fa", "-iold.gfa", "--inc-fp=new.gfa"],
        vec!["--input=a.fa", "--seeding", "-S"],
    ] {
        let error = cli::parse(values.iter().copied()).unwrap_err();
        assert_eq!(
            error.kind(),
            ErrorKind::ArgumentConflict,
            "{values:?}: {error}"
        );
    }
}

#[test]
fn cli_help_and_version_use_clap_display_errors() {
    for flag in ["-h", "--help"] {
        let error = cli::parse([flag]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DisplayHelp);
        assert_eq!(error.exit_code(), 0);
        assert!(!error.use_stderr());
        let help = error.to_string();
        assert!(help.contains("Usage:"));
        assert!(help.contains("--input"));
        for option in [
            "--gap-open",
            "--band-b",
            "--ab-band",
            "--k-mer",
            "--progressive-poa",
        ] {
            assert!(help.contains(option), "help omitted {option}: {help}");
        }
    }
    for flag in ["-v", "--version"] {
        let error = cli::parse([flag]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DisplayVersion);
        assert_eq!(error.exit_code(), 0);
        assert!(!error.use_stderr());
        assert!(error.to_string().contains(env!("CARGO_PKG_VERSION")));
    }
}

#[test]
fn scoring_matrix_respects_labels_and_row_order() {
    let file=Temp::new("mtx",b"# reordered alphabet\n N G A C T\nT 0 -3 -4 -5 7\nA 0 -2 6 -4 -5\nN 8 0 0 0 0\nC 0 -1 -2 9 -3\nG 0 10 -2 -3 -4\n");
    let mut p = Params::default();
    io::load_matrix(&mut p, &file.0).unwrap();
    assert_eq!(p.score(p.encode(b'a'), p.encode(b'A')), 6);
    assert_eq!(p.score(p.encode(b'g'), p.encode(b'C')), -3);
    assert_eq!(p.score(p.encode(b'U'), p.encode(b'T')), 7);
    assert_eq!(p.encode(b'?'), p.encode(b'N'));
}

#[test]
fn amino_acid_matrix_has_no_nucleotide_remapping() {
    let alphabet = b"ACGTNU";
    let mut text = String::from("A C G T N U\n");
    for &row in alphabet {
        text.push(row as char);
        for &col in alphabet {
            text.push_str(if row == col { " 5" } else { " -4" });
        }
        text.push('\n');
    }
    let file = Temp::new("mtx", text.as_bytes());
    let mut p = Params::default();
    io::load_matrix(&mut p, &file.0).unwrap();
    assert_ne!(p.encode(b'U'), p.encode(b'T'));
    assert_eq!(p.score(p.encode(b'U'), p.encode(b'U')), 5);
}

#[test]
fn rejects_malformed_matrix_instead_of_partial_initialization() {
    for text in [
        "# empty\n",
        "A A\nA 1 2\n",
        "A C\nA 1\nC 1 2\n",
        "A C\nA 1 2\n",
        "A C\nA 1 2\nA 2 1\n",
        "A C\nA 1 x\nC 1 2\n",
    ] {
        let file = Temp::new("mtx", text.as_bytes());
        let mut p = Params::default();
        assert!(
            io::load_matrix(&mut p, &file.0).is_err(),
            "accepted {text:?}"
        );
    }
}

#[test]
fn gfa_paths_include_single_base_segments_and_roundtrip() {
    let file = Temp::new(
        "gfa",
        b"H\tVN:Z:1.0\nS\tone\tA\nS\ttwo\tCG\nL\tone\t+\ttwo\t+\t0M\nP\tread\tone+,two+\t*\n",
    );
    let p = Params::default();
    let (graph, records) = io::read_gfa(&p, &file.0).unwrap();
    assert_eq!(records[0].seq, b"ACG");
    assert_eq!(graph.paths[0].1.len(), 3);
    assert_eq!(graph.consensus(&p).0, b"ACG");
    let out = graph.gfa(&p, &records).unwrap();
    let second = Temp::new("gfa", out.as_bytes());
    let (again, sequences) = io::read_gfa(&p, &second.0).unwrap();
    assert_eq!(again.consensus(&p).0, b"ACG");
    assert_eq!(sequences[0].seq, b"ACG");
}

#[test]
fn gfa_links_without_paths_still_initialize_a_graph() {
    let file = Temp::new("gfa", b"S\ta\tAC\nS\tb\tGT\nL\ta\t+\tb\t+\t0M\n");
    let p = Params::default();
    let (graph, records) = io::read_gfa(&p, &file.0).unwrap();
    assert!(records.is_empty());
    assert_eq!(graph.consensus(&p).0, b"ACGT");
}

#[test]
fn rejects_invalid_gfa_references_cycles_or_unsupported_orientation() {
    for text in [
        "S\ta\tA\nS\ta\tC\n",
        "S\ta\tA\nP\tx\tb+\t*\n",
        "S\ta\tA\nP\tx\ta-\t*\n",
        "S\ta\tA\nL\ta\t+\ta\t+\t0M\n",
        "S\ta\tA\nS\tb\tC\nL\ta\t+\tb\t+\t0M\nL\tb\t+\ta\t+\t0M\n",
        "S\ta\tA\nL\ta\t+\ta\t-\t0M\n",
        "S\ta\tA\nL\ta\t+\ta\t+\t1M\n",
        "S\ta\t*\n",
    ] {
        let file = Temp::new("gfa", text.as_bytes());
        assert!(
            io::read_gfa(&Params::default(), &file.0).is_err(),
            "accepted {text:?}"
        );
    }
}

#[test]
fn executable_consensus_msa_and_gfa_modes() {
    let file = Temp::new("fa", b">a\nACGT\n>b\nACGT\n");
    let bin = env!("CARGO_BIN_EXE_minipoa");
    let out = Process::new(bin)
        .arg("--input")
        .arg(&file.0)
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout, b">consensus sequence\nACGT\n");
    let out = Process::new(bin)
        .arg("--input")
        .arg(&file.0)
        .arg("-r1")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout, b">a \nACGT\n>b \nACGT\n");
    let out = Process::new(bin)
        .arg("--input")
        .arg(&file.0)
        .arg("-r2")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(out.stdout.starts_with(b"H\tVN:Z:1.0\n"));
    assert!(String::from_utf8(out.stdout)
        .unwrap()
        .contains("P\ta\t1+,2+,3+,4+\t*"));
}

#[test]
fn executable_usage_errors_exit_two_and_runtime_errors_exit_one() {
    let bin = env!("CARGO_BIN_EXE_minipoa");
    for values in [
        vec![],
        vec!["reads.fa"],
        vec!["--input=a.fa", "--unknown"],
        vec!["--input"],
        vec!["--input=a.fa", "-t0"],
    ] {
        let output = Process::new(bin).args(values).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
    let file = Temp::new("fq", b"@a\nACGT\n+\nI\n");
    let output = Process::new(bin)
        .arg("--input")
        .arg(&file.0)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("quality length mismatch"));
    let missing = Temp::new("fa", b"");
    fs::remove_file(&missing.0).unwrap();
    let output = Process::new(bin)
        .arg("--input")
        .arg(&missing.0)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(!output.stderr.is_empty());
}

#[test]
fn executable_help_and_version_exit_zero_on_stdout() {
    let bin = env!("CARGO_BIN_EXE_minipoa");
    for flag in ["-h", "--help", "-v", "--version"] {
        let output = Process::new(bin).arg(flag).output().unwrap();
        assert_eq!(output.status.code(), Some(0), "{flag}");
        assert!(output.stderr.is_empty(), "{flag}");
        let text = String::from_utf8(output.stdout).unwrap();
        if flag == "-h" || flag == "--help" {
            assert!(text.contains("Usage:") && text.contains("--input"));
        } else {
            assert!(text.contains(env!("CARGO_PKG_VERSION")));
        }
    }
}

#[cfg(unix)]
#[test]
fn cli_accepts_non_utf8_input_paths() {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};
    let path = OsString::from_vec(b"reads_\xff.fa".to_vec());
    let parsed = cli::parse([OsString::from("--input"), path.clone()]).unwrap();
    assert_eq!(parsed.input, PathBuf::from(path));
}

// Linux filesystems permit byte paths that macOS filesystems reject.
#[cfg(target_os = "linux")]
#[test]
fn executable_accepts_non_utf8_input_paths() {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};
    let mut file = Temp::new("fa", b">a\nACGT\n");
    let mut path = file.0.clone().into_os_string();
    path.push(OsString::from_vec(vec![0xff]));
    let path = PathBuf::from(path);
    fs::rename(&file.0, &path).unwrap();
    file.0 = path;
    let parsed = cli::parse([OsString::from("--input"), file.0.clone().into_os_string()]).unwrap();
    assert_eq!(parsed.input, file.0);
    let output = Process::new(env!("CARGO_BIN_EXE_minipoa"))
        .arg("--input")
        .arg(&file.0)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b">consensus sequence\nACGT\n");
}

#[test]
fn executable_reads_stdin_and_normalizes_rna() {
    use std::process::Stdio;
    let mut child = Process::new(env!("CARGO_BIN_EXE_minipoa"))
        .arg("--input")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b">rna\nacgu\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert_eq!(output.stdout, b">consensus sequence\nACGU\n");
}

#[test]
fn executable_incremental_graph_alignment() {
    let graph = Temp::new("gfa", b"S\ta\tACGT\nP\told\ta+\t*\n");
    let input = Temp::new("fa", b">new\nACGT\n");
    let output = Process::new(env!("CARGO_BIN_EXE_minipoa"))
        .arg("--input")
        .arg(&input.0)
        .arg("-i")
        .arg(&graph.0)
        .arg("-r1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b">old \nACGT\n>new \nACGT\n");
}

#[test]
fn gfa_empty_paths_export_and_reimport() {
    for text in [
        "H\tVN:Z:1.0\nP\tempty\t*\t*\n",
        "H\tVN:Z:1.0\nS\ta\tAC\nP\tempty\t*\t*\nP\tnonempty\ta+\t*\n",
    ] {
        let file = Temp::new("gfa", text.as_bytes());
        let p = Params::default();
        let (graph, records) = io::read_gfa(&p, &file.0).unwrap();
        assert_eq!(records[0].name, "empty");
        assert!(records[0].seq.is_empty());
        assert!(graph.paths[0].1.is_empty());
        let output = graph.gfa(&p, &records).unwrap();
        assert!(output.contains("P\tempty\t*\t*\n"));
        let roundtrip = Temp::new("gfa", output.as_bytes());
        let (loaded, again) = io::read_gfa(&p, &roundtrip.0).unwrap();
        assert_eq!(again, records);
        assert_eq!(loaded.consensus(&p).0, graph.consensus(&p).0);
    }
}

#[test]
fn rejects_gfa_paths_revisiting_single_base_segment() {
    let file = Temp::new("gfa", b"S\ta\tA\nP\tcycle\ta+,a+\t*\n");
    assert!(io::read_gfa(&Params::default(), &file.0)
        .unwrap_err()
        .contains("revisits"));
}

#[test]
fn executable_gzip_stdin_and_empty_record() {
    use std::process::Stdio;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(b">a\nACGT\n>empty\n").unwrap();
    let compressed = encoder.finish().unwrap();
    let mut child = Process::new(env!("CARGO_BIN_EXE_minipoa"))
        .arg("--input")
        .arg("-")
        .arg("-r1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&compressed).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert_eq!(output.stdout, b">a \nACGT\n>empty \n----\n");
}
