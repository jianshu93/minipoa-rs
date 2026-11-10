//! A portable Rust implementation of minipoa's partial order alignment workflow.
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]
pub mod align;
pub mod cli;
pub mod graph;
pub mod io;
pub mod minimizer;
mod parallel;

pub type Result<T> = std::result::Result<T, String>;

/// Compile-time DP configuration, also printed by the executable's `--version`.
/// Individual extreme-score alignments can still use the guarded i64 fallback.
pub fn dp_backend() -> &'static str {
    if cfg!(feature = "scalar-dp") {
        "scalar-i64 (forced)"
    } else if cfg!(feature = "avx512") {
        if cfg!(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "avx512f"
        )) {
            "avx512-native (16 x i32)"
        } else {
            "avx512-portable (16 x i32; native AVX512F disabled)"
        }
    } else if cfg!(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    )) {
        "avx2 (8 x i32)"
    } else if cfg!(all(target_arch = "aarch64", target_feature = "neon")) {
        "neon (8 x i32)"
    } else {
        "portable (8 x i32)"
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sequence {
    pub name: String,
    pub comment: String,
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub match_score: i32,
    pub mismatch: i32,
    pub gap_open: i32,
    pub gap_ext: i32,
    pub b: usize,
    pub f: usize,
    pub adaptive: bool,
    pub seeding: bool,
    pub k: usize,
    pub w: usize,
    pub progressive: bool,
    pub threads: usize,
    pub win_band: usize,
    pub win_max: usize,
    pub mm_filter_ratio: f64,
    pub max_occ: usize,
    pub tr_dist: usize,
    pub result: u8,
    pub verbose: u8,
    pub alphabet: Vec<u8>,
    pub matrix: Vec<i32>,
    pub is_rna: bool,
}

impl Default for Params {
    fn default() -> Self {
        let mut matrix = vec![0; 25];
        for i in 0..4 {
            for j in 0..4 {
                matrix[i * 5 + j] = if i == j { 2 } else { -4 };
            }
        }
        matrix[24] = 2;
        Self {
            match_score: 2,
            mismatch: -4,
            gap_open: -4,
            gap_ext: -2,
            b: 100,
            f: 40,
            adaptive: false,
            seeding: false,
            k: 19,
            w: 10,
            progressive: false,
            threads: 1,
            win_band: 500,
            win_max: 10000,
            mm_filter_ratio: 0.25,
            max_occ: 0,
            tr_dist: 1000,
            result: 0,
            verbose: 0,
            alphabet: b"ACGTN".to_vec(),
            matrix,
            is_rna: false,
        }
    }
}

impl Params {
    pub fn encode(&self, base: u8) -> usize {
        let base = base.to_ascii_uppercase();
        let base = if self.alphabet.len() <= 5 && base == b'U' {
            b'T'
        } else {
            base
        };
        self.alphabet
            .iter()
            .position(|&x| x == base)
            .unwrap_or_else(|| {
                self.alphabet
                    .iter()
                    .position(|&x| x == b'N')
                    .unwrap_or(self.alphabet.len() - 1)
            })
    }
    pub fn decode(&self, base: usize) -> u8 {
        let c = self.alphabet[base];
        if self.is_rna && self.alphabet.len() <= 5 && c == b'T' {
            b'U'
        } else {
            c
        }
    }
    pub fn score(&self, a: usize, b: usize) -> i32 {
        self.matrix[a * self.alphabet.len() + b]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlignedBase {
    pub node: Option<usize>,
    pub base: usize,
    pub aligned: Option<usize>,
}

impl Params {
    /// Check parameters before allocating matrices or computing minimizers.
    pub fn validate(&self) -> Result<()> {
        let n = self.alphabet.len();
        if n == 0 || n > 256 || self.matrix.len() != n * n {
            return Err("invalid scoring matrix dimensions".into());
        }
        if self.threads == 0 {
            return Err("thread count must be positive".into());
        }
        if !(1..=31).contains(&self.k) || !(1..32768).contains(&self.w) {
            return Err("k must be 1..31 and minimizer window must be 1..32767".into());
        }
        if self.gap_open > 0 || self.gap_ext > 0 {
            return Err("gap penalties must be zero or negative".into());
        }
        if self.result > 2 || self.verbose > 2 {
            return Err("result format and verbosity must be 0..2".into());
        }
        if !self.mm_filter_ratio.is_finite() || !(0.0..=1.0).contains(&self.mm_filter_ratio) {
            return Err("minimizer filter ratio must be between 0 and 1".into());
        }
        if self.win_max == 0 {
            return Err("window maximum must be positive".into());
        }
        Ok(())
    }
}

/// Align sequences in input order, or in minimizer guide-tree order with `-p`.
pub fn align_sequences(params: &Params, sequences: &[Sequence]) -> Result<graph::Graph> {
    params.validate()?;
    let mut graph = graph::Graph::new(params);
    extend_graph(params, &mut graph, sequences, 0)?;
    Ok(graph)
}

fn extend_graph(
    params: &Params,
    graph: &mut graph::Graph,
    sequences: &[Sequence],
    existing: usize,
) -> Result<()> {
    let order = if params.progressive && existing == 0 {
        minimizer::guide_order(params, sequences)
    } else {
        (0..sequences.len()).collect()
    };
    for (index, &id) in order.iter().enumerate().skip(existing) {
        if params.verbose > 0 && index % 10 == 0 {
            eprintln!("[{index}/{}]", sequences.len());
        }
        let path = align::align(params, graph, &sequences[id].seq)?;
        graph.add_path(id, &path)?;
        graph.toposort(false)?;
    }
    Ok(())
}

/// Execute the command-line workflow without writing stdout.
pub fn run(args: cli::Args) -> Result<String> {
    let mut params = args.params;
    let incremental = args.incremental.is_some();
    if let Some(path) = args.matrix {
        io::load_matrix(&mut params, &path)?;
    }
    params.validate()?;
    let (mut graph, mut sequences) = if let Some(path) = args.incremental {
        io::read_gfa(&params, &path)?
    } else {
        (graph::Graph::new(&params), Vec::new())
    };
    let existing = sequences.len();
    sequences.extend(io::read_sequences(&args.input)?);
    if sequences.is_empty() {
        return Err("input contains no sequences".into());
    }
    params.is_rna = params.alphabet.len() <= 5
        && sequences
            .iter()
            .any(|s| s.seq.iter().any(|c| c.eq_ignore_ascii_case(&b'U')));
    let mut alignment_params = params.clone();
    if incremental {
        alignment_params.progressive = false;
    }
    extend_graph(&alignment_params, &mut graph, &sequences, existing)?;
    match params.result {
        0 => Ok(format!(
            ">consensus sequence\n{}\n",
            String::from_utf8(graph.consensus(&params).0).map_err(|e| e.to_string())?
        )),
        1 => graph.msa(&params, &sequences),
        2 => graph.gfa(&params, &sequences),
        _ => unreachable!(),
    }
}
