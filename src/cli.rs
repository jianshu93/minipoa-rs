//! Typed command-line parsing with clap. Sequence input requires `--input`.
use crate::Params;
use clap::{CommandFactory, FromArgMatches, Parser};
use std::{ffi::OsString, path::PathBuf, sync::OnceLock};

#[derive(Clone, Debug)]
pub struct Args {
    pub params: Params,
    pub input: PathBuf,
    pub matrix: Option<PathBuf>,
    pub incremental: Option<PathBuf>,
}

#[derive(Debug, Parser)]
#[command(
    name = "minipoa",
    about = "Rust partial order alignment, consensus, MSA and GFA",
    version = version(),
    disable_version_flag = true,
    after_help = "Examples:\n  minipoa --input reads.fasta > consensus.fasta\n  minipoa --input reads.fasta -S -r1 -t4 > alignment.fasta\n  minipoa --input - -S -r2 > graph.gfa"
)]
struct Options {
    /// FASTA, FASTQ, or gzip-compressed input; use '-' to read stdin
    #[arg(long, value_name = "PATH")]
    input: PathBuf,

    /// Initialize from a forward, acyclic GFA1 graph
    #[arg(short = 'i', long = "inc-fp", alias = "inc_fp", value_name = "PATH")]
    incremental: Option<PathBuf>,

    /// Substitution matrix with alphabet header
    #[arg(short = 'm', long = "mat-fp", alias = "mat_fp", value_name = "PATH")]
    matrix: Option<PathBuf>,

    /// Match score (normalized to a positive value)
    #[arg(short = 'M', long = "match", default_value_t = 2, allow_negative_numbers = true, value_parser = score)]
    match_score: i32,

    /// Mismatch penalty (normalized to a negative value)
    #[arg(short = 'X', long, default_value_t = -4, allow_negative_numbers = true, value_parser = score)]
    mismatch: i32,

    /// Gap open penalty (normalized to a negative value)
    #[arg(short = 'O', long, alias = "gap_open", default_value_t = -4, allow_negative_numbers = true, value_parser = score)]
    gap_open: i32,

    /// Gap extension penalty (normalized to a negative value)
    #[arg(short = 'E', long, alias = "gap_ext", default_value_t = -2, allow_negative_numbers = true, value_parser = score)]
    gap_ext: i32,

    /// Alignment worker threads
    #[arg(short = 't', long = "thread", default_value_t = 1)]
    threads: usize,

    /// Base band width
    #[arg(short = 'b', long = "band-b", alias = "band_b", default_value_t = 100)]
    b: usize,

    /// Band expansion divisor; 0 disables banding
    #[arg(short = 'f', long = "band-f", alias = "band_f", default_value_t = 40)]
    f: usize,

    /// Enable adaptive banding
    #[arg(short = 'B', long = "ab-band", alias = "ab_band")]
    adaptive: bool,

    /// Enable SIMD minimizers and anchor windows
    #[arg(short = 'S', long)]
    seeding: bool,

    /// Maximum cumulative window drift
    #[arg(long, default_value_t = 500)]
    win_band: usize,

    /// Window length upper bound
    #[arg(long, default_value_t = 10000)]
    win_max: usize,

    /// Minimizer k-mer length (1-31)
    #[arg(short = 'k', long = "k-mer", alias = "k_mer", default_value_t = 19)]
    k: usize,

    /// Minimizer window length (1-32767)
    #[arg(short = 'w', long = "mm-w", alias = "mm_w", default_value_t = 10)]
    w: usize,

    /// Build a minimizer-based guide tree
    #[arg(short = 'p', long = "progressive-poa", alias = "progressive_poa")]
    progressive: bool,

    /// Guide tree minimizer filter ratio (0-1)
    #[arg(long, default_value_t = 0.25, allow_negative_numbers = true)]
    mm_filter_ratio: f64,

    /// Maximum anchor minimizer occurrences; 0 permits all
    #[arg(long, default_value_t = 0)]
    max_occ: usize,

    /// Tandem repeat distance; 0 disables filtering
    #[arg(long, default_value_t = 1000)]
    tr_dist: usize,

    /// Output format: 0=consensus FASTA, 1=MSA FASTA, 2=GFA
    #[arg(short = 'r', long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=2))]
    result: u8,

    /// Verbosity (0-2)
    #[arg(short = 'V', long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=2))]
    verbose: u8,
}

fn version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        format!(
            "{} (minipoa 1.5.3 port)\nDP backend: {}",
            env!("CARGO_PKG_VERSION"),
            crate::dp_backend()
        )
    })
}

fn score(value: &str) -> std::result::Result<i32, String> {
    let score: i32 = value.parse().map_err(|_| "expected an integer score")?;
    if !(-1_000_000..=1_000_000).contains(&score) {
        return Err("score magnitude exceeds 1000000".into());
    }
    Ok(score)
}

/// Parse arguments excluding the program name. Help and version use clap's
/// display errors (exit code 0); invalid command lines use exit code 2.
pub fn parse<I, T>(args: I) -> std::result::Result<Args, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    // Upstream uses -v for version and -V for the verbosity value.
    let mut command = Options::command().arg(
        clap::Arg::new("version")
            .short('v')
            .long("version")
            .action(clap::ArgAction::Version)
            .help("Print version and selected DP backend"),
    );
    let matches = command.try_get_matches_from_mut(
        std::iter::once(OsString::from("minipoa")).chain(args.into_iter().map(Into::into)),
    )?;
    let options = Options::from_arg_matches(&matches)?;
    let mut error = |message| command.error(clap::error::ErrorKind::ValueValidation, message);
    if options.threads == 0 {
        return Err(error("thread count must be greater than zero"));
    }
    if !(1..=31).contains(&options.k) {
        return Err(error("k-mer length must be in 1..=31"));
    }
    if !(1..=32767).contains(&options.w) {
        return Err(error("minimizer window length must be in 1..=32767"));
    }
    if options.win_max == 0 {
        return Err(error("window length upper bound must be greater than zero"));
    }
    if !options.mm_filter_ratio.is_finite() || !(0.0..=1.0).contains(&options.mm_filter_ratio) {
        return Err(error("minimizer filter ratio must be between 0 and 1"));
    }
    let mut params = Params {
        match_score: options.match_score.abs(),
        mismatch: -options.mismatch.abs(),
        gap_open: -options.gap_open.abs(),
        gap_ext: -options.gap_ext.abs(),
        threads: options.threads,
        b: options.b,
        f: options.f,
        adaptive: options.adaptive,
        seeding: options.seeding,
        win_band: options.win_band,
        win_max: options.win_max,
        k: options.k,
        w: options.w,
        progressive: options.progressive,
        mm_filter_ratio: options.mm_filter_ratio,
        max_occ: options.max_occ,
        tr_dist: options.tr_dist,
        result: options.result,
        verbose: options.verbose,
        ..Params::default()
    };
    let m = params.alphabet.len();
    for i in 0..m {
        for j in 0..m {
            params.matrix[i * m + j] = if i == j {
                params.match_score
            } else if i == m - 1 || j == m - 1 {
                0
            } else {
                params.mismatch
            };
        }
    }
    Ok(Args {
        params,
        input: options.input,
        matrix: options.matrix,
        incremental: options.incremental,
    })
}
