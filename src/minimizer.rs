//! SIMD DNA minimizers, repeat filtering, guide ordering, and seed windows.
//!
//! `simd-minimizers` processes eight packed sequence chunks in parallel (NEON or
//! AVX2). Its forward ntHash ordering and leftmost ties intentionally replace the
//! C++ implementation's Wang hash and repeated ties. K-mer identities are stored
//! exactly, so hash collisions cannot create a false sequence match.
use crate::{Params, Sequence};
use simd_minimizers::packed_seq::{PackedSeqVec, SeqVec};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Minimizer {
    /// Exact forward packed k-mer value; this is not a probabilistic hash.
    pub value: u64,
    /// Inclusive position of the k-mer's final base in the original sequence.
    pub end: usize,
    pub span: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Anchor {
    pub target_end: usize,
    pub query_end: usize,
    pub span: usize,
    pub adaptive_before: bool,
}

fn is_base(base: u8) -> bool {
    matches!(base.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T' | b'U')
}

/// Extract forward minimizers using the actual SIMD dependency fast path.
///
/// Ambiguous symbols split the sequence into DNA runs. U is normalized to T;
/// lowercase DNA is accepted. Runs shorter than k produce no minimizers. For a
/// partial window at a run boundary, use all available k-mers as one window.
/// k must be in 1..=31 and w in 1..32768; invalid library arguments return
/// an empty sketch (the CLI validates parameters before calling this function).
pub fn sketch(seq: &[u8], k: usize, w: usize) -> Vec<Minimizer> {
    if !(1..=31).contains(&k) || !(1..32768).contains(&w) || seq.len() >= u32::MAX as usize {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0;
    let mut positions = Vec::new();
    while start < seq.len() {
        while start < seq.len() && !is_base(seq[start]) {
            start += 1;
        }
        let mut end = start;
        while end < seq.len() && is_base(seq[end]) {
            end += 1;
        }
        if end - start >= k {
            // packed-seq uses (base >> 1) & 3: case is ignored and U packs
            // identically to T, avoiding a normalization copy of the input.
            let dna = &seq[start..end];
            let packed = PackedSeqVec::from_ascii(dna);
            positions.clear();
            let window = w.min(dna.len() - k + 1);
            let output =
                simd_minimizers::minimizers(k, window).run(packed.as_slice(), &mut positions);
            // Exact 2*k-bit sequence values, not the truncated ntHash ranking
            // keys used inside the SIMD scan. With k<=31 these fit losslessly
            // in u64, so anchor matching needs no super-window rehash pass.
            out.extend(output.pos_and_values_u64().map(|(pos, value)| Minimizer {
                value,
                end: start + pos as usize + k - 1,
                span: k,
            }));
        }
        start = end.saturating_add(1);
    }
    out
}

/// Half-open tandem-repeat intervals, extended by one observed repeat period.
/// Mirrors minipoa's recurrence-distance filter and merges gaps up to 50 bases.
pub fn repeat_intervals(mm: &[Minimizer], len: usize, max_dist: usize) -> Vec<(usize, usize)> {
    if max_dist == 0 {
        return Vec::new();
    }
    let mut last = HashMap::new();
    let mut intervals = Vec::new();
    for seed in mm {
        if let Some(previous) = last.insert(seed.value, seed.end) {
            let period = seed.end.saturating_sub(previous);
            if period > 0 && period <= max_dist {
                let start = previous
                    .saturating_add(1)
                    .saturating_sub(seed.span)
                    .saturating_sub(period);
                let end = seed.end.saturating_add(1).saturating_add(period).min(len);
                intervals.push((start, end));
            }
        }
    }
    intervals.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in intervals {
        if let Some(last) = merged.last_mut() {
            if start <= last.1.saturating_add(50) {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

fn repeat_coverage(mm: &[Minimizer], len: usize, max_dist: usize) -> Vec<usize> {
    let mut prefix = vec![0; len + 1];
    for (start, end) in repeat_intervals(mm, len, max_dist) {
        prefix[start + 1..=end].fill(1);
    }
    for i in 0..len {
        prefix[i + 1] += prefix[i];
    }
    prefix
}

fn overlaps_repeat(mm: Minimizer, prefix: &[usize]) -> bool {
    prefix[mm.end + 1] > prefix[mm.end + 1 - mm.span]
}

/// Collect exact, forward-strand matches after occurrence and repeat filtering.
pub fn collect_anchors(params: &Params, query: &[u8], target: &[u8]) -> Vec<Anchor> {
    if params.alphabet.len() > 5 {
        return Vec::new();
    }
    let qmm = sketch(query, params.k, params.w);
    let tmm = sketch(target, params.k, params.w);
    let qcov = repeat_coverage(&qmm, query.len(), params.tr_dist);
    let tcov = repeat_coverage(&tmm, target.len(), params.tr_dist);
    let mut qindex: BTreeMap<u64, Vec<Minimizer>> = BTreeMap::new();
    let mut tindex: BTreeMap<u64, Vec<Minimizer>> = BTreeMap::new();
    for mm in qmm {
        qindex.entry(mm.value).or_default().push(mm);
    }
    for mm in tmm {
        tindex.entry(mm.value).or_default().push(mm);
    }
    let mut anchors = Vec::new();
    for (value, tgroup) in tindex {
        let Some(qgroup) = qindex.get(&value) else {
            continue;
        };
        if params.max_occ > 0 && (tgroup.len() > params.max_occ || qgroup.len() > params.max_occ) {
            continue;
        }
        for t in tgroup {
            if overlaps_repeat(t, &tcov) {
                continue;
            }
            for &q in qgroup {
                if overlaps_repeat(q, &qcov) {
                    continue;
                }
                anchors.push(Anchor {
                    target_end: t.end,
                    query_end: q.end,
                    span: params.k,
                    adaptive_before: false,
                });
            }
        }
    }
    anchors.sort_unstable_by_key(|a| (a.target_end, a.query_end));
    anchors.dedup();
    anchors
}

fn chain_score(a: Anchor, b: Anchor, k: usize) -> Option<i32> {
    let dq = a.query_end.checked_sub(b.query_end)?;
    let dr = a.target_end.checked_sub(b.target_end)?;
    if dq == 0 || dq > 1000 {
        return None;
    }
    let delta = dr.abs_diff(dq);
    let distance = dr.min(dq);
    let mut score = a.span.min(distance) as i32;
    if delta > 0 || distance > a.span {
        if dr == 0 {
            score += 1;
        } else {
            let linear = 0.01 * k as f64 * delta as f64;
            let logarithmic = ((delta + 1) as f64).log2();
            score -= linear.min(logarithmic) as i32;
        }
    }
    Some(score)
}

/// Minimap-style sparse chaining with a 1000 bp predecessor horizon, 5000
/// predecessor cap, and the original 25-skip/50-nonbest heuristics. Local chains
/// require at least three seeds and score 100, then a second DP joins compatible
/// local chains across larger gaps.
pub fn chain(anchors: &[Anchor], k: usize) -> Vec<Anchor> {
    if anchors.is_empty() {
        return Vec::new();
    }
    let n = anchors.len();
    let mut scores = vec![0i32; n];
    let mut parent = vec![None; n];
    let mut touched = vec![usize::MAX; n];
    let mut start = 0;
    for i in 0..n {
        while start < i
            && anchors[i]
                .target_end
                .saturating_sub(anchors[start].target_end)
                > 1000
        {
            start += 1;
        }
        start = start.max(i.saturating_sub(5000));
        let mut best = anchors[i].span as i32;
        let mut skip = 0usize;
        let mut iterations = 0;
        for j in (start..i).rev() {
            let Some(step) = chain_score(anchors[i], anchors[j], k) else {
                continue;
            };
            let candidate = step + scores[j];
            if candidate > best {
                best = candidate;
                parent[i] = Some(j);
                skip = skip.saturating_sub(1);
            } else {
                if candidate != best {
                    iterations += 1;
                    if iterations > 50 {
                        break;
                    }
                }
                if touched[j] == i {
                    skip += 1;
                    if skip > 25 {
                        break;
                    }
                }
            }
            if let Some(p) = parent[j] {
                touched[p] = i;
            }
        }
        scores[i] = best;
    }
    let mut has_child = vec![false; n];
    for &p in &parent {
        if let Some(p) = p {
            has_child[p] = true;
        }
    }
    let mut terminals: Vec<usize> = (0..n)
        .filter(|&i| !has_child[i] && scores[i] >= 100)
        .collect();
    terminals.sort_unstable_by_key(|&i| (scores[i], i));
    let mut used = vec![false; n];
    let mut locals: Vec<Vec<usize>> = Vec::new();
    for &end in terminals.iter().rev() {
        if used[end] {
            continue;
        }
        // Backtrack until an already assigned chain or a >1000 score drop.
        let mut current = Some(end);
        let mut indexes = Vec::new();
        let mut best_gain = 0;
        let mut best_cut = None;
        while let Some(i) = current {
            if used[i] {
                break;
            }
            indexes.push(i);
            current = parent[i];
            let gain = scores[end] - current.map_or(0, |p| scores[p]);
            if gain > best_gain {
                best_gain = gain;
                best_cut = current;
            } else if best_gain - gain > 1000 {
                current = best_cut;
                break;
            }
        }
        if let Some(cut) = current {
            while indexes.last().is_some_and(|&i| i <= cut) {
                indexes.pop();
            }
        }
        for &i in &indexes {
            used[i] = true;
        }
        let local_score = scores[end] - current.map_or(0, |i| scores[i]);
        if indexes.len() >= 3 && local_score >= 100 {
            indexes.reverse();
            locals.push(indexes);
        }
    }
    if locals.is_empty() {
        return Vec::new();
    }
    locals.sort_unstable_by_key(|v| {
        let a = anchors[*v.last().unwrap()];
        (a.target_end, a.query_end)
    });
    // DP over local chains. Overlap is trimmed to the first seed beyond both
    // predecessor endpoints, matching the original chain stitching contract.
    let mut totals = vec![0i32; locals.len()];
    let mut previous = vec![None; locals.len()];
    let mut trim = vec![0usize; locals.len()];
    for i in 0..locals.len() {
        let local = &locals[i];
        let mut prefix = vec![anchors[local[0]].span as i32; local.len()];
        for x in 1..local.len() {
            prefix[x] = prefix[x - 1]
                + chain_score(anchors[local[x]], anchors[local[x - 1]], k).unwrap_or(0);
        }
        totals[i] = *prefix.last().unwrap();
        let first = anchors[local[0]];
        for j in (0..i).rev() {
            if first.target_end < anchors[locals[j][0]].target_end {
                continue;
            }
            let last = anchors[*locals[j].last().unwrap()];
            let Some(offset) = local.iter().position(|&v| {
                anchors[v].target_end > last.target_end && anchors[v].query_end > last.query_end
            }) else {
                continue;
            };
            let remaining =
                *prefix.last().unwrap() - prefix[offset] + anchors[local[offset]].span as i32;
            let candidate = totals[j] + remaining;
            if candidate > totals[i] {
                totals[i] = candidate;
                previous[i] = Some(j);
                trim[i] = offset;
            }
        }
    }
    let best = (0..locals.len())
        .max_by_key(|&i| (totals[i], std::cmp::Reverse(i)))
        .unwrap();
    let mut chain_ids = Vec::new();
    let mut next = Some(best);
    while let Some(i) = next {
        chain_ids.push(i);
        next = previous[i];
    }
    chain_ids.reverse();
    let mut result = Vec::new();
    for i in chain_ids {
        result.extend(locals[i][trim[i]..].iter().map(|&i| anchors[i]));
    }
    // A usable global chain is strictly increasing on both sequences. Equal
    // target positions can occur in the sparse DP but cannot be anchor windows.
    let mut monotone: Vec<Anchor> = Vec::new();
    for a in result {
        if monotone
            .last()
            .is_none_or(|p| a.target_end > p.target_end && a.query_end > p.query_end)
        {
            monotone.push(a);
        }
    }
    monotone
}

/// Partition a chain by cumulative diagonal drift, consecutive seed spacing,
/// and query-scaled maximum window length. Returns boundaries and tail band.
pub fn partition_windows(
    chain: &[Anchor],
    tlen: usize,
    qlen: usize,
    max_drift: usize,
    max_win: usize,
) -> (Vec<Anchor>, bool) {
    let n = chain.len();
    let te = |i: isize| {
        if i < 0 {
            -1
        } else {
            chain[i as usize].target_end as i64
        }
    };
    let qe = |i: isize| {
        if i < 0 {
            -1
        } else {
            chain[i as usize].query_end as i64
        }
    };
    let ts = |i: usize| {
        if i == n {
            tlen as i64
        } else {
            chain[i].target_end as i64 - chain[i].span as i64 + 1
        }
    };
    let qs = |i: usize| {
        if i == n {
            qlen as i64
        } else {
            chain[i].query_end as i64 - chain[i].span as i64 + 1
        }
    };
    let diag = |i: isize| {
        if i < 0 {
            0
        } else if i as usize == n {
            tlen as i64 - qlen as i64
        } else {
            te(i) - qe(i)
        }
    };
    let mut selected = Vec::new();
    let mut tail = false;
    let mut s = -1isize;
    while s < n as isize {
        let mut boundary = None;
        for i in (s + 1) as usize..=n {
            let gt = ts(i) - te(s) - 1;
            let gq = qs(i) - qe(s) - 1;
            let drift = (diag(i as isize) - diag(s)).unsigned_abs();
            let gap = (ts(i) - te(i as isize - 1) - 1).max(qs(i) - qe(i as isize - 1) - 1);
            if drift > max_drift as u64 || gap > max_drift as i64 || gt.max(gq) > max_win as i64 {
                break;
            }
            if gt >= 0 && gq >= 0 {
                boundary = Some(i);
            }
        }
        let adaptive;
        let e = if let Some(e) = boundary {
            adaptive = false;
            e
        } else {
            let mut e = (s + 1) as usize;
            while e < n && (ts(e) - te(s) - 1 < 0 || qs(e) - qe(s) - 1 < 0) {
                e += 1;
            }
            let drift = (diag(e as isize) - diag(s)).unsigned_abs();
            let gap = (ts(e) - te(s) - 1).max(qs(e) - qe(s) - 1);
            adaptive = drift > max_drift as u64 || gap > max_drift as i64;
            e
        };
        if e < n {
            let mut anchor = chain[e];
            anchor.adaptive_before = adaptive;
            selected.push(anchor);
        } else {
            tail = adaptive;
        }
        s = e as isize;
    }
    (selected, tail)
}

/// Seed-chain window boundaries; no chain means the aligner should use the full
/// graph with an adaptive band. Protein input always takes that fallback.
pub fn anchors(params: &Params, query: &[u8], consensus: &[u8]) -> Vec<Anchor> {
    anchor_windows(params, query, consensus).0
}

/// As `anchors`, additionally returning the tail window's adaptive-band flag.
pub fn anchor_windows(params: &Params, query: &[u8], consensus: &[u8]) -> (Vec<Anchor>, bool) {
    let matches = collect_anchors(params, query, consensus);
    let chain = chain(&matches, params.k);
    let max_win = params.win_max.min(500.max(query.len() / 25));
    partition_windows(
        &chain,
        consensus.len(),
        query.len(),
        params.win_band,
        max_win,
    )
}

/// Compute the last window's adaptive flag from a selected final boundary.
/// Prefer `anchor_windows` when its tail flag is already available.
pub fn tail_adaptive(
    params: &Params,
    query_len: usize,
    target_len: usize,
    last: Option<&Anchor>,
) -> bool {
    let (te, qe) = last.map_or((-1, -1), |a| (a.target_end as i64, a.query_end as i64));
    let gap = (target_len as i64 - te - 1).max(query_len as i64 - qe - 1);
    let drift = (target_len as i64 - query_len as i64 - (te - qe)).unsigned_abs();
    gap > params.win_band as i64 || drift > params.win_band as u64
}

/// Progressive order from minimizer multiset Jaccard similarity, weighted toward
/// longer reads, followed by maximum summed similarity to the growing group.
/// This preserves input order for proteins or when no minimizers are available.
///
/// This follows upstream's greedy group-similarity ordering, rather than building
/// a minimum spanning tree: start with the most similar ordered pair and append
/// the read having maximum summed similarity to all reads already selected.
/// The 0.51/0.49 length weighting makes pair similarity directional. Frequent
/// minimizers are discarded according to `mm_filter_ratio`; unlike upstream's
/// `initPara`, which resets that CLI option to 0.25, the supplied ratio is honored.
/// SIMD ntHash sketches differ from upstream Wang-hash sketches, so the resulting
/// guide order need not be identical when informative seeds are present.
pub fn guide_order(params: &Params, seqs: &[Sequence]) -> Vec<usize> {
    let n = seqs.len();
    let input: Vec<_> = (0..n).collect();
    if !params.progressive || n < 2 || params.alphabet.len() > 5 {
        return input;
    }
    let mut groups: BTreeMap<u64, BTreeMap<usize, usize>> = BTreeMap::new();
    for (rid, seq) in seqs.iter().enumerate() {
        for mm in sketch(&seq.seq, params.k, params.w) {
            *groups.entry(mm.value).or_default().entry(rid).or_default() += 1;
        }
    }
    if groups.is_empty() {
        return input;
    }
    let mut total = vec![0usize; n];
    let mut shared = vec![0usize; n * n];
    for counts in groups.values() {
        if counts.len() as f64 > n as f64 * params.mm_filter_ratio {
            continue;
        }
        for (&a, &ca) in counts {
            total[a] += ca;
            for (&b, &cb) in counts.range(..a) {
                shared[a * n + b] += ca.min(cb);
                shared[b * n + a] += ca.min(cb);
            }
        }
    }
    let max_len = seqs.iter().map(|s| s.seq.len()).max().unwrap_or(1).max(1) as f64;
    let mut similarity = vec![0f64; n * n];
    let mut best = (-1f64, 0, 1);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let common = shared[i * n + j];
            let union = total[i] + total[j] - common;
            let jac = if union == 0 {
                0.
            } else {
                common as f64 / union as f64
                    * (0.51 * seqs[i].seq.len() as f64 / max_len
                        + 0.49 * seqs[j].seq.len() as f64 / max_len)
            };
            similarity[i * n + j] = jac;
            if jac > best.0 {
                best = (jac, i, j);
            }
        }
    }
    let mut order = vec![best.1, best.2];
    let mut used = vec![false; n];
    used[best.1] = true;
    used[best.2] = true;
    let mut sums: Vec<f64> = (0..n)
        .map(|v| similarity[best.1 * n + v] + similarity[best.2 * n + v])
        .collect();
    while order.len() < n {
        let next = (0..n)
            .filter(|&v| !used[v])
            .max_by(|&a, &b| sums[a].total_cmp(&sums[b]).then_with(|| b.cmp(&a)))
            .unwrap();
        order.push(next);
        used[next] = true;
        for v in 0..n {
            if !used[v] {
                sums[v] += similarity[next * n + v];
            }
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    fn random_dna(n: usize) -> Vec<u8> {
        let mut x = 42u64;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                b"ACGT"[(x & 3) as usize]
            })
            .collect()
    }
    #[test]
    fn simd_matches_dependency_scalar_over_lengths_and_windows() {
        for n in [1, 5, 19, 31, 128, 1024, 4097] {
            let dna = random_dna(n);
            let packed = PackedSeqVec::from_ascii(&dna);
            for k in [1, 3, 5, 19, 28, 31].into_iter().filter(|&k| k <= n) {
                for w in [1, 3, 10, 255, 32767].into_iter().map(|w| w.min(n - k + 1)) {
                    let scalar =
                        simd_minimizers::minimizers(k, w).run_scalar_once(packed.as_slice());
                    let actual: Vec<_> = sketch(&dna, k, w)
                        .into_iter()
                        .map(|s| (s.end + 1 - k) as u32)
                        .collect();
                    assert_eq!(actual, scalar, "n={n},k={k},w={w}");
                }
            }
        }
    }
    #[test]
    fn ambiguous_runs_do_not_create_spanning_seeds() {
        let sequence = b"acgtACGTNNNNacguACGU";
        let seeds = sketch(sequence, 4, 3);
        assert!(!seeds.is_empty());
        for seed in &seeds {
            assert!(seed.end < 8 || seed.end >= 15);
        }
        assert_eq!(sketch(b"acgu", 4, 10), sketch(b"ACGT", 4, 10));
        assert!(sketch(b"NNNN", 3, 3).is_empty());
        assert!(sketch(b"AC", 3, 3).is_empty());
        assert!(sketch(b"ACGT", 0, 2).is_empty());
        assert!(sketch(b"ACGT", 3, 32768).is_empty());
        assert!(sketch(b"ACGT", 3, usize::MAX).is_empty());
    }
    #[test]
    fn largest_supported_window_matches_scalar() {
        let dna = random_dna(50_000);
        let packed = PackedSeqVec::from_ascii(&dna);
        let scalar = simd_minimizers::minimizers(31, 32767).run_scalar_once(packed.as_slice());
        let actual: Vec<_> = sketch(&dna, 31, 32767)
            .into_iter()
            .map(|m| (m.end + 1 - m.span) as u32)
            .collect();
        assert_eq!(actual, scalar);
    }
    #[test]
    fn exact_kmer_values_match() {
        let mut dna = random_dna(1024);
        dna.extend_from_within(..);
        for k in [1, 3, 19, 31] {
            let seeds = sketch(&dna, k, 1);
            assert_eq!(seeds.len(), dna.len() - k + 1);
            let mut seen = HashMap::new();
            for seed in seeds {
                let kmer = &dna[seed.end + 1 - seed.span..=seed.end];
                let exact = kmer.iter().enumerate().fold(0u64, |value, (i, base)| {
                    let bits = match base {
                        b'A' => 0,
                        b'C' => 1,
                        b'T' => 2,
                        b'G' => 3,
                        _ => unreachable!(),
                    };
                    value | (bits << (2 * i))
                });
                assert_eq!(seed.value, exact, "k={k},position={}", seed.end);
                if let Some(previous) = seen.insert(seed.value, kmer) {
                    assert_eq!(previous, kmer);
                }
            }
        }
    }
    #[test]
    fn repeat_filter_removes_homopolymer_matches() {
        let p = Params {
            k: 5,
            w: 3,
            tr_dist: 100,
            ..Params::default()
        };
        let dna = vec![b'A'; 100];
        assert!(collect_anchors(&p, &dna, &dna).is_empty());
        let unfiltered = Params {
            tr_dist: 0,
            max_occ: 1,
            ..p
        };
        assert!(collect_anchors(&unfiltered, &dna, &dna).is_empty());
    }
    #[test]
    fn repeat_intervals_extend_clip_and_merge() {
        let mm = vec![
            Minimizer {
                value: 1,
                end: 9,
                span: 5,
            },
            Minimizer {
                value: 1,
                end: 19,
                span: 5,
            },
            Minimizer {
                value: 2,
                end: 80,
                span: 5,
            },
            Minimizer {
                value: 2,
                end: 90,
                span: 5,
            },
        ];
        assert_eq!(repeat_intervals(&mm, 100, 5), vec![]);
        // The 50 bp merge gap joins these intervals.
        assert_eq!(repeat_intervals(&mm, 100, 100), vec![(0, 100)]);
    }
    #[test]
    fn chain_and_windows_are_monotone() {
        let dna = random_dna(5000);
        let p = Params {
            k: 19,
            w: 10,
            ..Params::default()
        };
        let (anchors, tail) = anchor_windows(&p, &dna, &dna);
        assert!(!anchors.is_empty());
        assert!(!tail);
        assert!(anchors
            .iter()
            .all(|a| a.target_end == a.query_end && !a.adaptive_before));
        for pair in anchors.windows(2) {
            assert!(pair[0].target_end + pair[1].span <= pair[1].target_end);
        }
    }
    #[test]
    fn windows_split_cumulative_drift() {
        let chain: Vec<_> = [(100, 100), (200, 200), (300, 220), (400, 230), (600, 600)]
            .into_iter()
            .map(|(t, q)| Anchor {
                target_end: t,
                query_end: q,
                span: 19,
                adaptive_before: false,
            })
            .collect();
        let (windows, _) = partition_windows(&chain, 800, 800, 100, 500);
        assert!(windows.iter().any(|a| a.adaptive_before));
        for pair in windows.windows(2) {
            assert!(pair[1].target_end >= pair[0].target_end + pair[1].span);
            assert!(pair[1].query_end >= pair[0].query_end + pair[1].span);
        }
    }
    #[test]
    fn guide_order_is_permutation_and_deterministic() {
        let seqs: Vec<_> = (0..12)
            .map(|i| Sequence {
                name: i.to_string(),
                seq: random_dna(600 + i),
                ..Sequence::default()
            })
            .collect();
        let p = Params {
            progressive: true,
            ..Params::default()
        };
        let order = guide_order(&p, &seqs);
        assert_eq!(order, guide_order(&p, &seqs));
        let mut sorted = order;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..seqs.len()).collect::<Vec<_>>());
        let short = vec![
            Sequence {
                seq: b"ACGT".to_vec(),
                ..Sequence::default()
            };
            3
        ];
        assert_eq!(guide_order(&p, &short), vec![0, 1, 2]);
    }
    #[test]
    fn no_local_chain_without_minimum_score() {
        let anchors: Vec<_> = (0..3)
            .map(|i| Anchor {
                target_end: 100 + i * 20,
                query_end: 100 + i * 20,
                span: 19,
                adaptive_before: false,
            })
            .collect();
        assert!(chain(&anchors, 19).is_empty());
    }
}
