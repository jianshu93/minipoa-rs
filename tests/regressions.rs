use minipoa::{align_sequences, Params, Sequence};

fn reads() -> Vec<Sequence> {
    [b"ACGT".as_slice(), b"ACCT".as_slice()]
        .into_iter()
        .enumerate()
        .map(|(i, seq)| Sequence {
            name: format!("r{i}"),
            seq: seq.to_vec(),
            ..Default::default()
        })
        .collect()
}

#[test]
fn rendering_msa_keeps_graph_ready_for_further_alignment() {
    let params = Params::default();
    let reads = reads();
    let mut graph = align_sequences(&params, &reads).unwrap();
    let order = graph.rank.clone();
    let ranks: Vec<_> = graph.nodes.iter().map(|n| n.rank).collect();
    assert!(graph.msa(&params, &reads).unwrap().contains("ACCT"));
    assert_eq!(graph.rank, order);
    assert_eq!(
        graph.nodes.iter().map(|n| n.rank).collect::<Vec<_>>(),
        ranks
    );
    let path = minipoa::align::align(&params, &graph, b"ACGT").unwrap();
    graph.add_path(2, &path).unwrap();
    graph.toposort(false).unwrap();
    assert_eq!(graph.consensus(&params).0, b"ACGT");
}

#[test]
fn zero_gap_open_traceback_regression() {
    let args =
        minipoa::cli::parse(["--input=unused", "-f0", "-M5", "-X-1", "-O0", "-E-1"]).unwrap();
    let reads = ["GCANCCTTG", "CTTCA"]
        .into_iter()
        .enumerate()
        .map(|(i, s)| Sequence {
            name: format!("r{i}"),
            seq: s.as_bytes().to_vec(),
            ..Default::default()
        })
        .collect::<Vec<_>>();
    let mut graph = align_sequences(&args.params, &reads).unwrap();
    let msa = graph.msa(&args.params, &reads).unwrap();
    let rows = msa
        .lines()
        .filter(|s| !s.starts_with('>'))
        .map(|s| s.replace('-', ""))
        .collect::<Vec<_>>();
    assert_eq!(rows, ["GCANCCTTG", "CTTCA"]);
}

#[test]
fn invalid_library_matrix_is_rejected_before_encoding() {
    let params = Params {
        alphabet: Vec::new(),
        matrix: Vec::new(),
        ..Default::default()
    };
    assert!(align_sequences(&params, &reads())
        .unwrap_err()
        .contains("matrix"));
}
