//! Affine-gap global sequence-to-DAG alignment.
//!
//! The main DP kernel processes eight i32 columns per step using AVX2 or NEON,
//! including affine insertion-prefix scans. Worker-local contiguous arenas are
//! reused across reads and seed windows. A conservative score-range guard sends
//! extreme workloads to the original i64 scalar kernel. Active-band rounding and
//! weighted traceback preserve upstream AVX2 output ordering.
//! With `avx512`, full sixteen-column groups use AVX-512F while eight-column
//! tails keep the same logical band boundaries and traceback behavior.
use crate::{graph::Graph, AlignedBase, Params, Result};
use std::cell::RefCell;
#[cfg(feature = "avx512")]
use wide::i32x16;
use wide::i32x8;

const NEG: i64 = -(1_i64 << 50);
const WIDTH: usize = 8;
#[cfg(feature = "avx512")]
const WIDE_WIDTH: usize = 16;
const MATCH: u8 = 1;
const DELETE: u8 = 2;
const INSERT: u8 = 4;
const ALL: u8 = MATCH | DELETE | INSERT;

#[derive(Default)]
struct Row {
    start: usize,
    m: Vec<i64>,
    d: Vec<i64>,
    ins: Vec<i64>,
}

impl Row {
    fn new(start: usize, end: usize) -> Result<Self> {
        let len = end.checked_sub(start).ok_or("invalid alignment band")?;
        len.checked_mul(3).ok_or("alignment matrix is too large")?;
        // Allocate each state separately so no references depend on padding or
        // unchecked pointer arithmetic.
        let alloc = || -> Result<Vec<i64>> {
            let mut v = Vec::new();
            v.try_reserve_exact(len)
                .map_err(|e| format!("cannot allocate alignment band: {e}"))?;
            v.resize(len, NEG);
            Ok(v)
        };
        Ok(Self {
            start,
            m: alloc()?,
            d: alloc()?,
            ins: alloc()?,
        })
    }
    fn contains(&self, j: usize) -> bool {
        j >= self.start && j - self.start < self.m.len()
    }
    fn m(&self, j: usize) -> i64 {
        if self.contains(j) {
            self.m[j - self.start]
        } else {
            NEG
        }
    }
    fn d(&self, j: usize) -> i64 {
        if self.contains(j) {
            self.d[j - self.start]
        } else {
            NEG
        }
    }
    fn ins(&self, j: usize) -> i64 {
        if self.contains(j) {
            self.ins[j - self.start]
        } else {
            NEG
        }
    }
}

/// Align a sequence to a topologically ordered graph. The source node is part
/// of the returned path; the sink is excluded, as in upstream minipoa.
pub fn align(params: &Params, graph: &Graph, query: &[u8]) -> Result<Vec<AlignedBase>> {
    if graph.nodes.len() <= 2 {
        let mut result = Vec::with_capacity(query.len() + 1);
        result.push(AlignedBase {
            node: Some(0),
            base: params.encode(b'N'),
            aligned: None,
        });
        result.extend(query.iter().map(|&c| AlignedBase {
            node: None,
            base: params.encode(c),
            aligned: None,
        }));
        return Ok(result);
    }
    if !params.seeding {
        return poa(params, graph, 0, 1, query, params.adaptive, params.b);
    }
    let (consensus, ids, _) = graph.consensus(params);
    let (anchors, tail_adaptive) = crate::minimizer::anchor_windows(params, query, &consensus);
    if anchors.is_empty() {
        return poa(params, graph, 0, 1, query, true, params.b);
    }
    struct Window {
        beg: usize,
        end: usize,
        qbeg: usize,
        qend: usize,
        adaptive: bool,
        anchor: Vec<usize>,
    }
    let mut windows = Vec::with_capacity(anchors.len() + 1);
    let mut beg = 0;
    let mut qbeg = 0;
    let mut previous_target_end = None;
    for anchor in &anchors {
        let Some(target_start) = anchor
            .target_end
            .checked_add(1)
            .and_then(|x| x.checked_sub(anchor.span))
        else {
            return Err("invalid minimizer target span".into());
        };
        let Some(query_start) = anchor
            .query_end
            .checked_add(1)
            .and_then(|x| x.checked_sub(anchor.span))
        else {
            return Err("invalid minimizer query span".into());
        };
        let valid = anchor.span > 0
            && anchor.target_end < ids.len()
            && anchor.query_end < query.len()
            && query_start >= qbeg
            && previous_target_end.is_none_or(|end| target_start > end)
            && (0..anchor.span).all(|offset| {
                params.encode(consensus[target_start + offset])
                    == params.encode(query[query_start + offset])
            });
        if !valid {
            // Hash collisions or overlapping anchors cannot be used as hard
            // constraints. The exact graph aligner remains the fallback.
            return poa(params, graph, 0, 1, query, true, params.b);
        }
        windows.push(Window {
            beg,
            end: ids[target_start],
            qbeg,
            qend: query_start,
            adaptive: anchor.adaptive_before,
            anchor: ids[target_start..anchor.target_end].to_vec(),
        });
        beg = ids[anchor.target_end];
        qbeg = anchor.query_end + 1;
        previous_target_end = Some(anchor.target_end);
    }
    windows.push(Window {
        beg,
        end: 1,
        qbeg,
        qend: query.len(),
        adaptive: tail_adaptive,
        anchor: Vec::new(),
    });
    let process = |window: &Window| -> Result<Vec<AlignedBase>> {
        let mut result = poa(
            params,
            graph,
            window.beg,
            window.end,
            &query[window.qbeg..window.qend],
            window.adaptive,
            if window.adaptive {
                params.b
            } else {
                params.win_band
            },
        )?;
        result.extend(window.anchor.iter().map(|&id| AlignedBase {
            node: Some(id),
            base: graph.nodes[id].base,
            aligned: None,
        }));
        Ok(result)
    };
    let parts = crate::parallel::map_ordered(&windows, params.threads, process)?;
    Ok(parts.into_iter().flatten().collect())
}

/// Align the query between two graph anchors. Both endpoints are treated as
/// sentinel columns: `beg_id` is included and `end_id` is excluded.
pub fn poa(
    params: &Params,
    graph: &Graph,
    beg_id: usize,
    end_id: usize,
    query: &[u8],
    adaptive: bool,
    band_b: usize,
) -> Result<Vec<AlignedBase>> {
    let result = poa_inner(params, graph, beg_id, end_id, query, adaptive, band_b);
    if result.is_err() && params.f != 0 {
        // A user-selected narrow band must not produce an incomplete path. Retry
        // the exact matrix only when the band cannot reach a valid traceback.
        let mut full = params.clone();
        full.f = 0;
        poa_inner(&full, graph, beg_id, end_id, query, false, band_b)
    } else {
        result
    }
}

// The fast kernel uses lossless i32 states only when every reachable score is
// conservatively bounded. Extreme-score or coordinate workloads use the original
// i64 kernel, which is also retained as a differential-test oracle.
const NEG32: i32 = -(1 << 29);
const SCORE_LIMIT: u64 = 1 << 26;
// Prefix shift padding is below every guarded reachable or unreachable state.
// Using NEG32 itself could invent an unreachable candidate when carry<NEG32.
const SHIFT_FILL: i32 = NEG32 - 2 * SCORE_LIMIT as i32;

thread_local! {
    // One arena per executing worker, retained across reads and seed windows.
    // No row owns a heap allocation, and disjoint workers never share buffers.
    static DP_ARENA: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy, Default)]
struct FastRow {
    start: usize,
    len: usize,
    m_offset: usize,
    d_offset: usize,
    i_offset: usize,
}

struct FastMatrix<'a> {
    data: &'a mut Vec<i32>,
    rows: Vec<FastRow>,
    cursor: usize,
}

impl FastMatrix<'_> {
    fn allocate(&mut self, start: usize, stop: usize) -> Result<FastRow> {
        let len = stop.checked_sub(start).ok_or("invalid alignment band")?;
        let size = len
            .checked_mul(3)
            .and_then(|v| v.checked_add(WIDTH * 2))
            .ok_or("alignment matrix is too large")?;
        let stop = self
            .cursor
            .checked_add(size)
            .ok_or("alignment matrix is too large")?;
        if stop > self.data.len() {
            self.data
                .try_reserve(stop - self.data.len())
                .map_err(|e| format!("cannot allocate SIMD alignment arena: {e}"))?;
            self.data.resize(stop, NEG32);
        }
        let row = FastRow {
            start,
            len,
            m_offset: self.cursor + WIDTH,
            d_offset: self.cursor + WIDTH * 2 + len,
            i_offset: self.cursor + WIDTH * 2 + len * 2,
        };
        // Leading/trailing M padding permits shifted predecessor vector loads.
        // M and D are accumulated from all predecessors; I is overwritten by
        // the prefix scan, so it does not need an additional initialization pass.
        self.data[self.cursor..row.d_offset + len].fill(NEG32);
        self.cursor = stop;
        Ok(row)
    }
}

trait TraceMatrix {
    fn contains_at(&self, row: usize, column: usize) -> bool;
    fn m_at(&self, row: usize, column: usize) -> i64;
    fn d_at(&self, row: usize, column: usize) -> i64;
    fn i_at(&self, row: usize, column: usize) -> i64;
    fn negative(&self) -> i64;
}

impl TraceMatrix for [Row] {
    fn contains_at(&self, row: usize, column: usize) -> bool {
        self[row].contains(column)
    }
    fn m_at(&self, row: usize, column: usize) -> i64 {
        self[row].m(column)
    }
    fn d_at(&self, row: usize, column: usize) -> i64 {
        self[row].d(column)
    }
    fn i_at(&self, row: usize, column: usize) -> i64 {
        self[row].ins(column)
    }
    fn negative(&self) -> i64 {
        NEG
    }
}

impl TraceMatrix for FastMatrix<'_> {
    fn contains_at(&self, row: usize, column: usize) -> bool {
        let row = self.rows[row];
        column >= row.start && column - row.start < row.len
    }
    fn m_at(&self, row: usize, column: usize) -> i64 {
        let r = self.rows[row];
        if self.contains_at(row, column) {
            self.data[r.m_offset + column - r.start] as i64
        } else {
            NEG32 as i64
        }
    }
    fn d_at(&self, row: usize, column: usize) -> i64 {
        let r = self.rows[row];
        if self.contains_at(row, column) {
            self.data[r.d_offset + column - r.start] as i64
        } else {
            NEG32 as i64
        }
    }
    fn i_at(&self, row: usize, column: usize) -> i64 {
        let r = self.rows[row];
        if self.contains_at(row, column) {
            self.data[r.i_offset + column - r.start] as i64
        } else {
            NEG32 as i64
        }
    }
    fn negative(&self) -> i64 {
        NEG32 as i64
    }
}

struct MatrixRow<'a, M: TraceMatrix + ?Sized> {
    matrix: &'a M,
    index: usize,
}
impl<M: TraceMatrix + ?Sized> MatrixRow<'_, M> {
    fn contains(&self, j: usize) -> bool {
        self.matrix.contains_at(self.index, j)
    }
    fn m(&self, j: usize) -> i64 {
        self.matrix.m_at(self.index, j)
    }
    fn d(&self, j: usize) -> i64 {
        self.matrix.d_at(self.index, j)
    }
    fn ins(&self, j: usize) -> i64 {
        self.matrix.i_at(self.index, j)
    }
}

#[inline]
fn load8(data: &[i32], offset: usize) -> i32x8 {
    i32x8::new(data[offset..offset + WIDTH].try_into().unwrap())
}

#[inline]
fn store8(data: &mut [i32], offset: usize, value: i32x8) {
    data[offset..offset + WIDTH].copy_from_slice(value.as_array());
}

#[cfg(feature = "avx512")]
#[inline]
fn load16(data: &[i32], offset: usize) -> i32x16 {
    i32x16::new(data[offset..offset + WIDE_WIDTH].try_into().unwrap())
}

#[cfg(feature = "avx512")]
#[inline]
fn store16(data: &mut [i32], offset: usize, value: i32x16) {
    data[offset..offset + WIDE_WIDTH].copy_from_slice(value.as_array());
}

/// Sixteen affine insertion transitions with carry from the preceding block.
/// Constant lane shifts map to AVX-512F permutations rather than two unrelated
/// eight-lane scans: the shift-eight stage propagates gaps across both halves.
#[cfg(feature = "avx512")]
#[inline]
fn insertion_prefix16(md: i32x16, carry: i32, open: i32, extension: i32) -> i32x16 {
    let mut v = md + i32x16::splat(open);
    let a = v.to_array();
    v = v.max(
        i32x16::new([
            SHIFT_FILL, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11],
            a[12], a[13], a[14],
        ]) + i32x16::splat(extension),
    );
    let a = v.to_array();
    v = v.max(
        i32x16::new([
            SHIFT_FILL, SHIFT_FILL, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],
            a[10], a[11], a[12], a[13],
        ]) + i32x16::splat(extension * 2),
    );
    let a = v.to_array();
    v = v.max(
        i32x16::new([
            SHIFT_FILL, SHIFT_FILL, SHIFT_FILL, SHIFT_FILL, a[0], a[1], a[2], a[3], a[4], a[5],
            a[6], a[7], a[8], a[9], a[10], a[11],
        ]) + i32x16::splat(extension * 4),
    );
    let a = v.to_array();
    v = v.max(
        i32x16::new([
            SHIFT_FILL, SHIFT_FILL, SHIFT_FILL, SHIFT_FILL, SHIFT_FILL, SHIFT_FILL, SHIFT_FILL,
            SHIFT_FILL, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
        ]) + i32x16::splat(extension * 8),
    );
    let carry = i32x16::splat(carry)
        + i32x16::new([
            extension,
            extension * 2,
            extension * 3,
            extension * 4,
            extension * 5,
            extension * 6,
            extension * 7,
            extension * 8,
            extension * 9,
            extension * 10,
            extension * 11,
            extension * 12,
            extension * 13,
            extension * 14,
            extension * 15,
            extension * 16,
        ]);
    v.max(carry)
}

#[cfg(feature = "avx512")]
#[inline]
fn row_maximum16(scores: &[i32]) -> i32 {
    let (chunks, tail) = scores.as_chunks::<WIDE_WIDTH>();
    let maximum = chunks
        .iter()
        .fold(i32x16::splat(NEG32), |best, chunk| {
            best.max(load16(chunk, 0))
        })
        .reduce_max();
    if tail.is_empty() {
        maximum
    } else {
        // Logical rows remain multiples of eight. Never include a padded or
        // neighboring matrix state in the reduction.
        debug_assert_eq!(tail.len(), WIDTH);
        maximum.max(load8(tail, 0).reduce_max())
    }
}

/// Eight affine insertion transitions as a parallel maximum-prefix scan.
/// `carry` is I[j]; the returned lanes are I[j+1]..I[j+8]. Three lane-shift
/// stages replace the eight serial extension dependencies. All SIMD operations
/// are handled by wide's AVX2 or NEON backend, with its portable fallback.
#[inline]
fn insertion_prefix(md: i32x8, carry: i32, open: i32, extension: i32) -> i32x8 {
    let mut v = md + i32x8::splat(open);
    // Fixed lane selections expose constant LLVM shuffles. On NEON these
    // lower to EXT moves, avoiding the generic byte-table shuffle path.
    let a = v.to_array();
    v = v.max(
        i32x8::new([SHIFT_FILL, a[0], a[1], a[2], a[3], a[4], a[5], a[6]])
            + i32x8::splat(extension),
    );
    let a = v.to_array();
    v = v.max(
        i32x8::new([SHIFT_FILL, SHIFT_FILL, a[0], a[1], a[2], a[3], a[4], a[5]])
            + i32x8::splat(extension * 2),
    );
    let a = v.to_array();
    v = v.max(
        i32x8::new([
            SHIFT_FILL, SHIFT_FILL, SHIFT_FILL, SHIFT_FILL, a[0], a[1], a[2], a[3],
        ]) + i32x8::splat(extension * 4),
    );
    let carry = i32x8::splat(carry)
        + i32x8::new([
            extension,
            extension * 2,
            extension * 3,
            extension * 4,
            extension * 5,
            extension * 6,
            extension * 7,
            extension * 8,
        ]);
    v.max(carry)
}

fn i32_scores_safe(
    params: &Params,
    graph: &Graph,
    beg_id: usize,
    end_id: usize,
    query_len: usize,
) -> bool {
    if cfg!(feature = "scalar-dp") || beg_id >= graph.nodes.len() || end_id >= graph.nodes.len() {
        return false;
    }
    let Some(rows) = graph.nodes[end_id]
        .rank
        .checked_sub(graph.nodes[beg_id].rank)
    else {
        return false;
    };
    let Some(steps) = rows
        .checked_add(query_len)
        .and_then(|v| v.checked_add(WIDTH * 2 + 4))
    else {
        return false;
    };
    let opening = i64::from(params.gap_open) + i64::from(params.gap_ext);
    let largest = params
        .matrix
        .iter()
        .map(|&x| i64::from(x).unsigned_abs())
        .chain([
            opening.unsigned_abs(),
            i64::from(params.gap_ext).unsigned_abs(),
        ])
        .max()
        .unwrap_or(0);
    largest
        .checked_mul(steps as u64)
        .is_some_and(|bound| bound <= SCORE_LIMIT)
}

fn poa_inner(
    params: &Params,
    graph: &Graph,
    beg_id: usize,
    end_id: usize,
    query: &[u8],
    adaptive: bool,
    band_b: usize,
) -> Result<Vec<AlignedBase>> {
    if i32_scores_safe(params, graph, beg_id, end_id, query.len()) {
        poa_simd_i32(params, graph, beg_id, end_id, query, adaptive, band_b)
    } else {
        poa_scalar_i64(params, graph, beg_id, end_id, query, adaptive, band_b)
    }
}

fn poa_simd_i32(
    params: &Params,
    graph: &Graph,
    beg_id: usize,
    end_id: usize,
    query: &[u8],
    adaptive: bool,
    band_b: usize,
) -> Result<Vec<AlignedBase>> {
    let mut arena = ArenaLease::take();
    poa_simd_arena(
        params,
        graph,
        beg_id,
        end_id,
        query,
        adaptive,
        band_b,
        &mut arena.0,
    )
}

/// Return the worker's largest allocation even if an alignment unwinds.
struct ArenaLease(Vec<i32>);
impl ArenaLease {
    fn take() -> Self {
        Self(DP_ARENA.with(|pool| std::mem::take(&mut *pool.borrow_mut())))
    }
}
impl Drop for ArenaLease {
    fn drop(&mut self) {
        DP_ARENA.with(|pool| {
            let mut pool = pool.borrow_mut();
            if self.0.capacity() > pool.capacity() {
                std::mem::swap(&mut *pool, &mut self.0);
            }
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn poa_simd_arena(
    params: &Params,
    graph: &Graph,
    beg_id: usize,
    end_id: usize,
    query: &[u8],
    adaptive: bool,
    band_b: usize,
    arena: &mut Vec<i32>,
) -> Result<Vec<AlignedBase>> {
    if beg_id >= graph.nodes.len() || end_id >= graph.nodes.len() {
        return Err("alignment anchor is not in graph".into());
    }
    let beg = graph.nodes[beg_id].rank;
    let end = graph.nodes[end_id].rank;
    if end <= beg || end >= graph.rank.len() {
        return Err("alignment anchors are not topologically ordered".into());
    }
    let n = end - beg + 1;
    let m = query.len().checked_add(2).ok_or("sequence is too long")?;
    if n > i32::MAX as usize || m > i32::MAX as usize {
        return Err("alignment exceeds coordinate limits".into());
    }
    let unknown = params.encode(b'N');
    let padded = (m / WIDTH + 1) * WIDTH;
    let mut seq = Vec::with_capacity(padded);
    seq.push(unknown);
    seq.extend(query.iter().map(|&c| params.encode(c)));
    seq.resize(padded, unknown);
    let adaptive = adaptive || params.adaptive;
    let band = band_b
        .saturating_add(m.checked_div(params.f).unwrap_or(0))
        .min(i32::MAX as usize) as i64;
    let gap_ext = params.gap_ext;
    let gap_open = params.gap_open + gap_ext;
    let mut head = vec![NEG; n];
    let mut tail = vec![NEG; n];
    let mut support = vec![0_i64; n];
    head[0] = 0;
    // Most-supported-path distances determine a static band even when there
    // are branches of very different lengths in the DAG.
    for i in 1..n {
        let node = &graph.nodes[graph.rank[beg + i]];
        let mut best: Option<(usize, usize)> = None;
        for (k, &pred) in node.incoming.iter().enumerate() {
            let rank = graph.nodes[pred].rank;
            if rank < beg || rank >= beg + i {
                continue;
            }
            let p = rank - beg;
            if head[p] == NEG {
                continue;
            }
            if best.is_none_or(|(prev, weight)| {
                node.in_weights[k] > weight
                    || (node.in_weights[k] == weight && support[p] > support[prev])
            }) {
                best = Some((p, node.in_weights[k]));
            }
        }
        if let Some((p, w)) = best {
            head[i] = head[p] + 1;
            support[i] = support[p] + w as i64;
        }
    }
    tail[n - 1] = 1;
    for i in (0..n - 1).rev() {
        let node = &graph.nodes[graph.rank[beg + i]];
        let mut best: Option<(usize, usize)> = None;
        for (k, &succ) in node.outgoing.iter().enumerate() {
            let rank = graph.nodes[succ].rank;
            if rank <= beg + i || rank > end {
                continue;
            }
            let s = rank - beg;
            if tail[s] == NEG {
                continue;
            }
            if best.is_none_or(|(prev, weight)| {
                node.out_weights[k] > weight
                    || (node.out_weights[k] == weight && support[s] > support[prev])
            }) {
                best = Some((s, node.out_weights[k]));
            }
        }
        if let Some((s, w)) = best {
            tail[i] = tail[s] + 1;
            support[i] = support[s] + w as i64;
        }
    }
    if head[n - 1] == NEG {
        return Err("no graph path connects alignment anchors".into());
    }
    let mut peak = vec![0_i64; n];
    let mut streak = vec![0_i64; n];
    let mut peak_left = vec![m as i64; n];
    let mut peak_right = vec![0_i64; n];
    let mut offset_left = vec![0_i64; n];
    let mut offset_right = vec![0_i64; n];
    let mut extra = vec![0_i64; n];

    let alphabet = params.alphabet.len();
    let mut profiles = vec![0i32; alphabet * padded];
    for base in 0..alphabet {
        for (column, &query_base) in seq.iter().enumerate() {
            profiles[base * padded + column] = params.score(base, query_base);
        }
    }
    let ext = i32x8::splat(gap_ext);
    let open = i32x8::splat(gap_open);
    #[cfg(feature = "avx512")]
    let ext16 = i32x16::splat(gap_ext);
    #[cfg(feature = "avx512")]
    let open16 = i32x16::splat(gap_open);
    let mut matrix = FastMatrix {
        data: arena,
        rows: vec![FastRow::default(); n],
        cursor: 0,
    };
    for i in 0..n {
        if head[i] == NEG || tail[i] == NEG {
            continue;
        }
        let (left, right) = if params.f == 0 {
            (0, m as i64)
        } else if adaptive {
            let a = peak_left[i]
                .min(head[i] + offset_left[i])
                .min(m as i64 - tail[i]);
            let b = peak_right[i]
                .max(head[i] + offset_right[i])
                .max(m as i64 - tail[i]);
            (
                (a - band - extra[i]).max(0),
                (b + band + extra[i]).min(m as i64),
            )
        } else {
            (
                (head[i].min(m as i64 - tail[i]) - band).max(0),
                (head[i].max(m as i64 - tail[i]) + band).min(m as i64),
            )
        };
        if left > right || right < 0 {
            return Err("alignment band does not intersect graph path".into());
        }
        let start = left as usize / WIDTH * WIDTH;
        let stop = (right as usize / WIDTH + 1) * WIDTH;
        let row = matrix.allocate(start, stop)?;
        matrix.rows[i] = row;
        if i == 0 {
            if start != 0 {
                return Err("alignment band excludes source".into());
            }
            matrix.data[row.m_offset] = params.score(unknown, unknown);
            matrix.data[row.i_offset] = NEG32;
            for j in 1..row.len {
                let ins = (matrix.data[row.i_offset + j - 1] + gap_ext)
                    .max(matrix.data[row.m_offset + j - 1] + gap_open);
                matrix.data[row.i_offset + j] = ins;
                matrix.data[row.m_offset + j] = matrix.data[row.m_offset + j].max(ins);
            }
            continue;
        }
        let node = &graph.nodes[graph.rank[beg + i]];
        // The optional AVX-512 body consumes sixteen columns when both rows
        // contain them. Logical bands still round to eight; any final block
        // uses the existing AVX2/NEON kernel, with no scores outside the band.
        for &pred in &node.incoming {
            let rank = graph.nodes[pred].rank;
            if rank < beg || rank >= beg + i {
                continue;
            }
            let p = rank - beg;
            if head[p] == NEG {
                continue;
            }
            let previous = matrix.rows[p];
            let overlap_start = start.max(previous.start);
            let overlap_stop = stop.min(previous.start + previous.len);
            #[allow(unused_mut)]
            let mut body_end = overlap_start;
            #[cfg(feature = "avx512")]
            while body_end + WIDE_WIDTH <= overlap_stop {
                let local = body_end - start;
                let prior = body_end - previous.start;
                let diagonal = load16(matrix.data, previous.m_offset + prior - 1);
                let previous_m = load16(matrix.data, previous.m_offset + prior);
                let previous_d = load16(matrix.data, previous.d_offset + prior);
                let m = load16(matrix.data, row.m_offset + local).max(diagonal);
                let d = load16(matrix.data, row.d_offset + local)
                    .max(previous_d + ext16)
                    .max(previous_m + open16);
                store16(matrix.data, row.m_offset + local, m);
                store16(matrix.data, row.d_offset + local, d);
                body_end += WIDE_WIDTH;
            }
            for j in (body_end..overlap_stop).step_by(WIDTH) {
                let local = j - start;
                let prior = j - previous.start;
                let diagonal = load8(matrix.data, previous.m_offset + prior - 1);
                let previous_m = load8(matrix.data, previous.m_offset + prior);
                let previous_d = load8(matrix.data, previous.d_offset + prior);
                let m = load8(matrix.data, row.m_offset + local).max(diagonal);
                let d = load8(matrix.data, row.d_offset + local)
                    .max(previous_d + ext)
                    .max(previous_m + open);
                store8(matrix.data, row.m_offset + local, m);
                store8(matrix.data, row.d_offset + local, d);
            }
        }
        let base = if i == n - 1 { unknown } else { node.base };
        let profile = &profiles[base * padded..(base + 1) * padded];
        let mut carry = NEG32;
        #[allow(unused_mut)]
        let mut body_end = 0;
        #[cfg(feature = "avx512")]
        while body_end + WIDE_WIDTH <= row.len {
            let diagonal =
                load16(matrix.data, row.m_offset + body_end) + load16(profile, start + body_end);
            let md = diagonal.max(load16(matrix.data, row.d_offset + body_end));
            let next = insertion_prefix16(md, carry, gap_open, gap_ext);
            let a = next.to_array();
            let current = i32x16::new([
                carry, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11],
                a[12], a[13], a[14],
            ]);
            store16(matrix.data, row.i_offset + body_end, current);
            store16(matrix.data, row.m_offset + body_end, md.max(current));
            carry = next.as_array()[WIDE_WIDTH - 1];
            body_end += WIDE_WIDTH;
        }
        for local in (body_end..row.len).step_by(WIDTH) {
            let diagonal = load8(matrix.data, row.m_offset + local) + load8(profile, start + local);
            let md = diagonal.max(load8(matrix.data, row.d_offset + local));
            let next = insertion_prefix(md, carry, gap_open, gap_ext);
            let a = next.to_array();
            let current = i32x8::new([carry, a[0], a[1], a[2], a[3], a[4], a[5], a[6]]);
            store8(matrix.data, row.i_offset + local, current);
            store8(matrix.data, row.m_offset + local, md.max(current));
            carry = next.as_array()[WIDTH - 1];
        }
        if params.f != 0 && adaptive {
            let scores = &matrix.data[row.m_offset..row.m_offset + row.len];
            #[cfg(feature = "avx512")]
            let maximum_score = row_maximum16(scores);
            #[cfg(not(feature = "avx512"))]
            let maximum_score = scores
                .as_chunks::<WIDTH>()
                .0
                .iter()
                .fold(i32x8::splat(NEG32), |best, chunk| best.max(load8(chunk, 0)))
                .reduce_max();
            let argmax = scores
                .iter()
                .position(|&score| score == maximum_score)
                .unwrap();
            let maximum = (start + argmax) as i64;
            peak[i] = maximum;
            let mut max_streak = 0;
            let mut is_match = false;
            if node.base == seq[maximum as usize] {
                for &pred in &node.incoming {
                    let rank = graph.nodes[pred].rank;
                    if rank < beg || rank >= beg + i {
                        continue;
                    }
                    let p = rank - beg;
                    if head[p] == NEG {
                        continue;
                    }
                    max_streak = max_streak.max(streak[p]);
                    if peak[p] + 1 == maximum {
                        streak[i] = streak[i].max(streak[p] + 1);
                        is_match = true;
                    }
                }
            }
            if !is_match && max_streak < 30 {
                extra[i] = (extra[i] + max_streak + 1).min(5 * band);
            }
            if streak[i] >= 30 {
                offset_left[i] = maximum - head[i];
                offset_right[i] = maximum - head[i];
                extra[i] = 0;
                for &succ in &node.outgoing {
                    let rank = graph.nodes[succ].rank;
                    if rank > beg + i && rank <= end {
                        let s = rank - beg;
                        offset_left[s] = offset_left[i];
                        offset_right[s] = offset_right[i];
                    }
                }
            }
            for &succ in &node.outgoing {
                let rank = graph.nodes[succ].rank;
                if rank <= beg + i || rank > end {
                    continue;
                }
                let s = rank - beg;
                offset_left[s] = offset_left[s].min(offset_left[i]);
                offset_right[s] = offset_right[s].max(offset_right[i]);
                peak_left[s] = peak_left[s].min(maximum);
                peak_right[s] = peak_right[s].max(maximum + 1);
                extra[s] = if extra[i] != 0 {
                    extra[s].max(extra[i])
                } else {
                    0
                };
            }
        }
    }
    traceback(
        params,
        graph,
        beg,
        n,
        &seq,
        m,
        &head,
        &matrix,
        i64::from(gap_open),
        i64::from(gap_ext),
    )
}

fn poa_scalar_i64(
    params: &Params,
    graph: &Graph,
    beg_id: usize,
    end_id: usize,
    query: &[u8],
    adaptive: bool,
    band_b: usize,
) -> Result<Vec<AlignedBase>> {
    if beg_id >= graph.nodes.len() || end_id >= graph.nodes.len() {
        return Err("alignment anchor is not in graph".into());
    }
    let beg = graph.nodes[beg_id].rank;
    let end = graph.nodes[end_id].rank;
    if end <= beg || end >= graph.rank.len() {
        return Err("alignment anchors are not topologically ordered".into());
    }
    let n = end - beg + 1;
    let m = query.len().checked_add(2).ok_or("sequence is too long")?;
    if n > i32::MAX as usize || m > i32::MAX as usize {
        return Err("alignment exceeds coordinate limits".into());
    }
    let unknown = params.encode(b'N');
    let padded = (m / WIDTH + 1) * WIDTH;
    let mut seq = Vec::with_capacity(padded);
    seq.push(unknown);
    seq.extend(query.iter().map(|&c| params.encode(c)));
    seq.resize(padded, unknown);
    let adaptive = adaptive || params.adaptive;
    let band = band_b
        .saturating_add(m.checked_div(params.f).unwrap_or(0))
        .min(i32::MAX as usize) as i64;
    let gap_ext = i64::from(params.gap_ext);
    let gap_open = i64::from(params.gap_open) + gap_ext;
    let mut head = vec![NEG; n];
    let mut tail = vec![NEG; n];
    let mut support = vec![0_i64; n];
    head[0] = 0;
    // Most-supported-path distances determine a static band even when there
    // are branches of very different lengths in the DAG.
    for i in 1..n {
        let node = &graph.nodes[graph.rank[beg + i]];
        let mut best: Option<(usize, usize)> = None;
        for (k, &pred) in node.incoming.iter().enumerate() {
            let rank = graph.nodes[pred].rank;
            if rank < beg || rank >= beg + i {
                continue;
            }
            let p = rank - beg;
            if head[p] == NEG {
                continue;
            }
            if best.is_none_or(|(prev, weight)| {
                node.in_weights[k] > weight
                    || (node.in_weights[k] == weight && support[p] > support[prev])
            }) {
                best = Some((p, node.in_weights[k]));
            }
        }
        if let Some((p, w)) = best {
            head[i] = head[p] + 1;
            support[i] = support[p] + w as i64;
        }
    }
    tail[n - 1] = 1;
    for i in (0..n - 1).rev() {
        let node = &graph.nodes[graph.rank[beg + i]];
        let mut best: Option<(usize, usize)> = None;
        for (k, &succ) in node.outgoing.iter().enumerate() {
            let rank = graph.nodes[succ].rank;
            if rank <= beg + i || rank > end {
                continue;
            }
            let s = rank - beg;
            if tail[s] == NEG {
                continue;
            }
            if best.is_none_or(|(prev, weight)| {
                node.out_weights[k] > weight
                    || (node.out_weights[k] == weight && support[s] > support[prev])
            }) {
                best = Some((s, node.out_weights[k]));
            }
        }
        if let Some((s, w)) = best {
            tail[i] = tail[s] + 1;
            support[i] = support[s] + w as i64;
        }
    }
    if head[n - 1] == NEG {
        return Err("no graph path connects alignment anchors".into());
    }
    let mut peak = vec![0_i64; n];
    let mut streak = vec![0_i64; n];
    let mut peak_left = vec![m as i64; n];
    let mut peak_right = vec![0_i64; n];
    let mut offset_left = vec![0_i64; n];
    let mut offset_right = vec![0_i64; n];
    let mut extra = vec![0_i64; n];
    let mut rows: Vec<Row> = (0..n).map(|_| Row::default()).collect();
    for i in 0..n {
        if head[i] == NEG || tail[i] == NEG {
            continue;
        }
        let (left, right) = if params.f == 0 {
            (0, m as i64)
        } else if adaptive {
            let a = peak_left[i]
                .min(head[i] + offset_left[i])
                .min(m as i64 - tail[i]);
            let b = peak_right[i]
                .max(head[i] + offset_right[i])
                .max(m as i64 - tail[i]);
            (
                (a - band - extra[i]).max(0),
                (b + band + extra[i]).min(m as i64),
            )
        } else {
            (
                (head[i].min(m as i64 - tail[i]) - band).max(0),
                (head[i].max(m as i64 - tail[i]) + band).min(m as i64),
            )
        };
        if left > right || right < 0 {
            return Err("alignment band does not intersect graph path".into());
        }
        let start = left as usize / WIDTH * WIDTH;
        let stop = (right as usize / WIDTH + 1) * WIDTH;
        let mut row = Row::new(start, stop)?;
        if i == 0 {
            if !row.contains(0) {
                return Err("alignment band excludes source".into());
            }
            row.m[0] = i64::from(params.score(unknown, unknown));
            for j in 1..row.m.len() {
                row.ins[j] = (row.ins[j - 1] + gap_ext).max(row.m[j - 1] + gap_open);
                row.m[j] = row.m[j].max(row.ins[j]);
            }
            rows[i] = row;
            continue;
        }
        let node = &graph.nodes[graph.rank[beg + i]];
        let base = if i == n - 1 { unknown } else { node.base };
        for (local, j) in (start..stop).enumerate() {
            let block = j / WIDTH * WIDTH;
            let mut diagonal = NEG;
            let mut deletion = NEG;
            for &pred in &node.incoming {
                let rank = graph.nodes[pred].rank;
                if rank < beg || rank >= beg + i {
                    continue;
                }
                let p = rank - beg;
                if head[p] == NEG {
                    continue;
                }
                let prev = &rows[p];
                // Upstream gates transitions a SIMD block at a time. Matching
                // this avoids architecture-dependent changes at band edges.
                if !prev.contains(block) {
                    continue;
                }
                if j > 0 {
                    diagonal = diagonal.max(prev.m(j - 1));
                }
                deletion = deletion.max(prev.d(j) + gap_ext).max(prev.m(j) + gap_open);
            }
            row.d[local] = deletion;
            let md = (diagonal + i64::from(params.score(base, seq[j]))).max(deletion);
            row.m[local] = md.max(row.ins[local]);
            if local + 1 < row.m.len() {
                row.ins[local + 1] = (row.ins[local] + gap_ext).max(md + gap_open);
            }
        }
        if params.f != 0 && adaptive {
            let mut argmax = 0;
            for j in 1..row.m.len() {
                if row.m[j] > row.m[argmax] {
                    argmax = j;
                }
            }
            let maximum = (start + argmax) as i64;
            peak[i] = maximum;
            let mut max_streak = 0;
            let mut is_match = false;
            if node.base == seq[maximum as usize] {
                for &pred in &node.incoming {
                    let rank = graph.nodes[pred].rank;
                    if rank < beg || rank >= beg + i {
                        continue;
                    }
                    let p = rank - beg;
                    if head[p] == NEG {
                        continue;
                    }
                    max_streak = max_streak.max(streak[p]);
                    if peak[p] + 1 == maximum {
                        streak[i] = streak[i].max(streak[p] + 1);
                        is_match = true;
                    }
                }
            }
            if !is_match && max_streak < 30 {
                extra[i] = (extra[i] + max_streak + 1).min(5 * band);
            }
            if streak[i] >= 30 {
                offset_left[i] = maximum - head[i];
                offset_right[i] = maximum - head[i];
                extra[i] = 0;
                for &succ in &node.outgoing {
                    let rank = graph.nodes[succ].rank;
                    if rank > beg + i && rank <= end {
                        let s = rank - beg;
                        offset_left[s] = offset_left[i];
                        offset_right[s] = offset_right[i];
                    }
                }
            }
            for &succ in &node.outgoing {
                let rank = graph.nodes[succ].rank;
                if rank <= beg + i || rank > end {
                    continue;
                }
                let s = rank - beg;
                offset_left[s] = offset_left[s].min(offset_left[i]);
                offset_right[s] = offset_right[s].max(offset_right[i]);
                peak_left[s] = peak_left[s].min(maximum);
                peak_right[s] = peak_right[s].max(maximum + 1);
                extra[s] = if extra[i] != 0 {
                    extra[s].max(extra[i])
                } else {
                    0
                };
            }
        }
        rows[i] = row;
    }
    traceback(
        params,
        graph,
        beg,
        n,
        &seq,
        m,
        &head,
        rows.as_slice(),
        gap_open,
        gap_ext,
    )
}

#[allow(clippy::too_many_arguments)]
fn traceback<M: TraceMatrix + ?Sized>(
    params: &Params,
    graph: &Graph,
    beg: usize,
    n: usize,
    seq: &[usize],
    m: usize,
    head: &[i64],
    rows: &M,
    gap_open: i64,
    gap_ext: i64,
) -> Result<Vec<AlignedBase>> {
    let mut i = n - 1;
    let mut j = m - 1;
    let mut op = ALL;
    let mut result = Vec::with_capacity(m);
    let mut steps = 0;
    while i > 0 || j > 0 {
        steps += 1;
        if steps > n + m {
            return Err("alignment traceback failed to advance".into());
        }
        let row = MatrixRow {
            matrix: rows,
            index: i,
        };
        if !row.contains(j) || row.m(j) <= rows.negative() / 2 {
            return Err("alignment traceback left active band".into());
        }
        let node = &graph.nodes[graph.rank[beg + i]];
        let base = if i == n - 1 {
            params.encode(b'N')
        } else {
            node.base
        };
        let score = i64::from(params.score(base, seq[j]));
        let choose = |state: u8, penalty: i64, previous_j: usize| -> Option<usize> {
            let mut best = None;
            let mut weight = 0;
            for (k, &pred) in node.incoming.iter().enumerate() {
                let rank = graph.nodes[pred].rank;
                if rank < beg || rank >= beg + i {
                    continue;
                }
                let p = rank - beg;
                if head[p] == NEG || !rows.contains_at(p, previous_j) {
                    continue;
                }
                let source = if state == DELETE {
                    rows.d_at(p, previous_j)
                } else {
                    rows.m_at(p, previous_j)
                };
                let target = if state == MATCH { row.m(j) } else { row.d(j) };
                if target == source + penalty && (best.is_none() || node.in_weights[k] > weight) {
                    best = Some(k);
                    weight = node.in_weights[k];
                }
            }
            best
        };
        let preferred = if op & MATCH != 0 && j > 0 && base == seq[j] {
            choose(MATCH, score, j - 1)
        } else {
            None
        };
        let preferred =
            preferred.filter(|&k| node.in_weights[k] >= node.in_weights.iter().sum::<usize>() / 10);
        if let Some(k) = preferred {
            push_diagonal(&mut result, graph, i, n, j, m, seq[j], node.id, base);
            i = graph.nodes[node.incoming[k]].rank - beg;
            j -= 1;
            op = ALL;
            continue;
        }
        if op & DELETE != 0 && (op == DELETE || row.m(j) == row.d(j)) {
            // Prefer opening a gap over extending it when both attain the same
            // score, then prefer the incoming edge with the greatest support.
            if let Some(k) = choose(INSERT, gap_open, j) {
                i = graph.nodes[node.incoming[k]].rank - beg;
                // With zero opening cost, opening and extending are tied;
                // excluding DELETE here can strand traceback at column zero.
                op = if params.gap_open == 0 {
                    ALL
                } else {
                    MATCH | INSERT
                };
                continue;
            }
            if let Some(k) = choose(DELETE, gap_ext, j) {
                i = graph.nodes[node.incoming[k]].rank - beg;
                op = DELETE;
                continue;
            }
        }
        if op & INSERT != 0
            && j > 0
            && row.contains(j - 1)
            && (op == INSERT || row.m(j) == row.ins(j))
        {
            if row.ins(j) == row.m(j - 1) + gap_open {
                result.push(AlignedBase {
                    node: None,
                    base: seq[j],
                    aligned: None,
                });
                j -= 1;
                // The same tied-state case applies to an insertion run.
                op = if params.gap_open == 0 {
                    ALL
                } else {
                    MATCH | DELETE
                };
                continue;
            }
            if row.ins(j) == row.ins(j - 1) + gap_ext {
                result.push(AlignedBase {
                    node: None,
                    base: seq[j],
                    aligned: None,
                });
                j -= 1;
                op = INSERT;
                continue;
            }
        }
        if op & MATCH != 0 && j > 0 {
            if let Some(k) = choose(MATCH, score, j - 1) {
                push_diagonal(&mut result, graph, i, n, j, m, seq[j], node.id, base);
                i = graph.nodes[node.incoming[k]].rank - beg;
                j -= 1;
                op = ALL;
                continue;
            }
        }
        return Err(format!(
            "alignment traceback has no predecessor at graph row {i}, query column {j}"
        ));
    }
    let start = graph.rank[beg];
    result.push(AlignedBase {
        node: Some(start),
        base: graph.nodes[start].base,
        aligned: None,
    });
    result.reverse();
    result.pop();
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn push_diagonal(
    result: &mut Vec<AlignedBase>,
    graph: &Graph,
    i: usize,
    n: usize,
    j: usize,
    m: usize,
    query_base: usize,
    node_id: usize,
    node_base: usize,
) {
    if i == n - 1 && j != m - 1 {
        result.push(AlignedBase {
            node: None,
            base: query_base,
            aligned: None,
        });
    } else if node_base == query_base {
        result.push(AlignedBase {
            node: Some(node_id),
            base: graph.nodes[node_id].base,
            aligned: None,
        });
    } else {
        let parent = graph.nodes[node_id].parent;
        result.push(AlignedBase {
            node: graph.nodes[parent].aligned[query_base],
            base: query_base,
            aligned: Some(parent),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn linear(params: &Params, reference: &[u8]) -> Graph {
        let mut graph = Graph::new(params);
        let path = align(params, &graph, reference).unwrap();
        graph.add_path(0, &path).unwrap();
        graph.toposort(false).unwrap();
        graph
    }
    fn pairwise_optimum(params: &Params, reference: &[u8], query: &[u8]) -> i64 {
        let width = query.len() + 1;
        let mut best = vec![NEG; (reference.len() + 1) * width];
        let mut deletion = best.clone();
        let mut insertion = best.clone();
        best[0] = 0;
        let e = i64::from(params.gap_ext);
        let o = i64::from(params.gap_open) + e;
        for i in 0..=reference.len() {
            for j in 0..=query.len() {
                let pos = i * width + j;
                if i > 0 {
                    deletion[pos] = (best[pos - width] + o).max(deletion[pos - width] + e);
                }
                if j > 0 {
                    insertion[pos] = (best[pos - 1] + o).max(insertion[pos - 1] + e);
                }
                let diagonal =
                    if i > 0 && j > 0 {
                        best[pos - width - 1]
                            + i64::from(params.score(
                                params.encode(reference[i - 1]),
                                params.encode(query[j - 1]),
                            ))
                    } else {
                        best[pos]
                    };
                best[pos] = diagonal.max(deletion[pos]).max(insertion[pos]);
            }
        }
        *best.last().unwrap()
    }
    fn alignment_score(
        params: &Params,
        graph: &Graph,
        path: &[AlignedBase],
        reference_len: usize,
    ) -> i64 {
        let mut score = 0;
        let mut previous = 0;
        let mut last_op = 0;
        let add_gap = |op, score: &mut i64, last: &mut u8| {
            *score += i64::from(params.gap_ext);
            if *last != op {
                *score += i64::from(params.gap_open);
            }
            *last = op;
        };
        for item in path.iter().skip(1) {
            let target = item.node.or(item.aligned);
            if let Some(id) = target {
                let rank = graph.nodes[id].rank;
                for _ in previous + 1..rank {
                    add_gap(DELETE, &mut score, &mut last_op);
                }
                score += i64::from(params.score(graph.nodes[id].base, item.base));
                last_op = MATCH;
                previous = rank;
            } else {
                add_gap(INSERT, &mut score, &mut last_op);
            }
        }
        for _ in previous..reference_len {
            add_gap(DELETE, &mut score, &mut last_op);
        }
        score
    }
    #[test]
    fn affine_alignment_matches_independent_pairwise_optimum() {
        let mut rng = 1923827_u64;
        let mut random = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 32) as usize
        };
        let params = Params {
            f: 0,
            ..Params::default()
        };
        for _ in 0..500 {
            let reference: Vec<_> = (0..1 + random() % 22)
                .map(|_| b"ACGT"[random() % 4])
                .collect();
            let query: Vec<_> = (0..random() % 23).map(|_| b"ACGT"[random() % 4]).collect();
            let graph = linear(&params, &reference);
            let path = align(&params, &graph, &query).unwrap();
            assert_eq!(path.len(), query.len() + 1);
            assert_eq!(
                alignment_score(&params, &graph, &path, reference.len()),
                pairwise_optimum(&params, &reference, &query),
                "reference={:?} query={:?}",
                String::from_utf8_lossy(&reference),
                String::from_utf8_lossy(&query)
            );
        }
    }
    #[test]
    fn affine_alignment_handles_zero_gaps_and_ambiguous_bases() {
        let mut rng = 912873_u64;
        let mut random = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 32) as usize
        };
        for (matching, mismatch, opening, extension) in [
            (2, -4, -4, -2),
            (5, -1, 0, -1),
            (0, -2, -2, 0),
            (3, 0, 0, 0),
        ] {
            let mut params = Params {
                f: 0,
                gap_open: opening,
                gap_ext: extension,
                ..Params::default()
            };
            for a in 0..4 {
                for b in 0..4 {
                    params.matrix[a * 5 + b] = if a == b { matching } else { mismatch };
                }
            }
            params.matrix[24] = matching;
            for _ in 0..150 {
                let reference: Vec<_> = (0..1 + random() % 20)
                    .map(|_| b"ACGTN"[random() % 5])
                    .collect();
                let query: Vec<_> = (0..random() % 21).map(|_| b"ACGTN"[random() % 5]).collect();
                let graph = linear(&params, &reference);
                let path = align(&params, &graph, &query).unwrap_or_else(|e|panic!("scoring={matching},{mismatch},{opening},{extension}; reference={:?} query={:?}: {e}",String::from_utf8_lossy(&reference),String::from_utf8_lossy(&query)));
                assert_eq!(path.len(), query.len() + 1);
                assert_eq!(alignment_score(&params,&graph,&path,reference.len()),pairwise_optimum(&params,&reference,&query),"scoring={matching},{mismatch},{opening},{extension}; reference={:?} query={:?}",String::from_utf8_lossy(&reference),String::from_utf8_lossy(&query));
            }
        }
    }
    #[test]
    fn static_and_adaptive_bands_preserve_simple_indels() {
        let mut params = Params {
            b: 2,
            f: 40,
            ..Params::default()
        };
        let reference = b"ACGTACGATAGCTACGATCGATGCTAGCTAGCTAGCATCGATCG";
        let query = b"ACGTACGATAGCTACGATCGATTTTGCTAGCTAGCTAGCATCGATCG";
        let graph = linear(&params, reference);
        let mut full = params.clone();
        full.f = 0;
        let optimum = pairwise_optimum(&params, reference, query);
        for adaptive in [false, true] {
            params.adaptive = adaptive;
            let path = align(&params, &graph, query).unwrap();
            assert_eq!(path.len(), query.len() + 1);
            assert_eq!(
                alignment_score(&params, &graph, &path, reference.len()),
                optimum
            );
        }
        assert_eq!(
            alignment_score(
                &params,
                &graph,
                &align(&full, &graph, query).unwrap(),
                reference.len()
            ),
            optimum
        );
    }
    #[test]
    fn seeded_windows_are_identical_across_worker_counts() {
        let mut params = Params {
            seeding: true,
            k: 7,
            w: 4,
            tr_dist: 0,
            win_max: 80,
            win_band: 20,
            ..Params::default()
        };
        let mut rng = 123981_u64;
        let reference: Vec<_> = (0..2400)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                b"ACGT"[(rng >> 32) as usize % 4]
            })
            .collect();
        let mut query = reference.clone();
        query.splice(800..800, b"TGCATGC".iter().copied());
        query.drain(1600..1605);
        query[2000] = if query[2000] == b'A' { b'C' } else { b'A' };
        let graph = linear(&params, &reference);
        let windows = crate::minimizer::anchor_windows(&params, &query, &reference).0;
        assert!(
            windows.len() > 2,
            "test must exercise multiple alignment windows"
        );
        let sequential = align(&params, &graph, &query).unwrap();
        assert_eq!(sequential.len(), query.len() + 1);
        for threads in [2, 4, 16] {
            params.threads = threads;
            assert_eq!(align(&params, &graph, &query).unwrap(), sequential);
        }
    }
    #[test]
    fn row_outside_band_is_unreachable() {
        let mut row = Row::new(8, 24).unwrap();
        row.m[3] = 17;
        assert_eq!(row.m(11), 17);
        assert_eq!(row.m(7), NEG);
        assert_eq!(row.m(24), NEG);
    }
    #[test]
    fn simd_prefix_scan_matches_serial_affine_transitions() {
        let mut state = 8917321u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as i32 & 0x7fffffff
        };
        for (opening, extension) in [(0, 0), (0, -2), (-4, 0), (-6, -2), (-117, -12)] {
            for case in 0..500 {
                let md = std::array::from_fn(|_| {
                    if random() % 11 == 0 {
                        NEG32
                    } else {
                        random() % 201 - 100
                    }
                });
                let carry = if case % 9 == 0 {
                    NEG32
                } else {
                    random() % 201 - 100
                };
                let actual = insertion_prefix(i32x8::new(md), carry, opening, extension).to_array();
                let mut value = carry;
                let expected = md.map(|md| {
                    value = (value + extension).max(md + opening);
                    value
                });
                assert_eq!(
                    actual, expected,
                    "open={opening},ext={extension},carry={carry}"
                );
            }
        }
    }

    #[test]
    fn simd_matches_i64_traceback_on_branched_graphs_and_bands() {
        let mut state = 7512129u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as usize
        };
        for case in 0..35 {
            let mut params = Params::default();
            match case % 4 {
                1 => {
                    params.gap_open = 0;
                    params.gap_ext = -1;
                }
                2 => {
                    params.gap_ext = 0;
                }
                3 => {
                    params.gap_open = 0;
                    params.gap_ext = 0;
                }
                _ => {}
            }
            if case % 5 == 0 {
                params.matrix[24] = 0;
            }
            let template: Vec<_> = (0..40 + random() % 80)
                .map(|_| b"ACGTN"[random() % 5])
                .collect();
            let mut graph = linear(&params, &template);
            for rid in 1..5 {
                let mut query = Vec::new();
                for &base in &template {
                    if random() % 20 == 0 {
                        continue;
                    }
                    query.push(if random() % 12 == 0 {
                        b"ACGTN"[random() % 5]
                    } else {
                        base
                    });
                    if random() % 20 == 0 {
                        query.push(b"ACGT"[random() % 4]);
                    }
                }
                for (f, adaptive, band) in [
                    (0, false, 0),
                    (40, false, 2),
                    (40, true, 3),
                    (40, false, 100),
                ] {
                    params.f = f;
                    params.adaptive = adaptive;
                    let fast = poa_simd_i32(&params, &graph, 0, 1, &query, adaptive, band);
                    let scalar = poa_scalar_i64(&params, &graph, 0, 1, &query, adaptive, band);
                    assert_eq!(
                        fast, scalar,
                        "case={case},read={rid},f={f},adaptive={adaptive},band={band}"
                    );
                }
                params.f = 0;
                let path = poa_scalar_i64(&params, &graph, 0, 1, &query, false, 100).unwrap();
                graph.add_path(rid, &path).unwrap();
                graph.toposort(false).unwrap();
            }
        }
    }

    #[test]
    fn score_guard_dispatches_normal_work_to_simd_and_extreme_work_to_i64() {
        let params = Params::default();
        let graph = linear(&params, b"ACGTACGTACGT");
        DP_ARENA.with(|arena| *arena.borrow_mut() = Vec::new());
        assert_eq!(
            i32_scores_safe(&params, &graph, 0, 1, 12),
            !cfg!(feature = "scalar-dp")
        );
        poa_inner(&params, &graph, 0, 1, b"ACGTACCTACGT", false, 100).unwrap();
        DP_ARENA
            .with(|arena| assert_eq!(arena.borrow().capacity() > 0, !cfg!(feature = "scalar-dp")));
        let mut extreme = params.clone();
        extreme.matrix.iter_mut().for_each(|v| *v = i32::MAX);
        assert!(!i32_scores_safe(&extreme, &graph, 0, 1, 12));
        DP_ARENA.with(|arena| *arena.borrow_mut() = Vec::new());
        assert_eq!(
            poa_inner(&extreme, &graph, 0, 1, b"ACGTACCTACGT", false, 100),
            poa_scalar_i64(&extreme, &graph, 0, 1, b"ACGTACCTACGT", false, 100)
        );
        DP_ARENA.with(|arena| assert_eq!(arena.borrow().capacity(), 0));
    }

    #[test]
    fn simd_arena_reuses_storage_and_restores_it_after_error() {
        let params = Params::default();
        let graph = linear(&params, b"ACGTACGTACGTACGT");
        poa_simd_i32(&params, &graph, 0, 1, b"ACGTACCTACGTACGT", false, 100).unwrap();
        let before = DP_ARENA.with(|arena| (arena.borrow().as_ptr(), arena.borrow().capacity()));
        poa_simd_i32(&params, &graph, 0, 1, b"ACGT", false, 100).unwrap();
        let after = DP_ARENA.with(|arena| (arena.borrow().as_ptr(), arena.borrow().capacity()));
        assert_eq!(before, after);
        assert!(poa_simd_i32(&params, &graph, 1, 0, b"A", false, 100).is_err());
        DP_ARENA
            .with(|arena| assert_eq!((arena.borrow().as_ptr(), arena.borrow().capacity()), before));
    }
    #[cfg(feature = "avx512")]
    #[test]
    fn avx512_prefix_matches_serial_including_unreachable_and_cross_block_carries() {
        let mut state = 9716271u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as i32 & 0x7fffffff
        };
        for (open, extension) in [
            (0, 0),
            (0, -2),
            (-4, 0),
            (-6, -2),
            (-117, -12),
            (-3_000_000, -3_000_000),
        ] {
            for case in 0..400 {
                let md: [i32; 24] = std::array::from_fn(|_| {
                    if case % 13 == 0 || random() % 7 == 0 {
                        NEG32
                    } else {
                        random() % 2001 - 1000
                    }
                });
                let carry = if case % 11 == 0 {
                    NEG32 - 1000
                } else {
                    random() % 2001 - 1000
                };
                let body = insertion_prefix16(
                    i32x16::new(md[..16].try_into().unwrap()),
                    carry,
                    open,
                    extension,
                )
                .to_array();
                let tail = insertion_prefix(
                    i32x8::new(md[16..].try_into().unwrap()),
                    body[15],
                    open,
                    extension,
                )
                .to_array();
                let actual: Vec<_> = body.into_iter().chain(tail).collect();
                let mut score = carry;
                let expected: Vec<_> = md
                    .into_iter()
                    .map(|md| {
                        score = (score + extension).max(md + open);
                        score
                    })
                    .collect();
                assert_eq!(
                    actual, expected,
                    "case={case},open={open},extension={extension},carry={carry}"
                );
            }
        }
        let actual = insertion_prefix16(i32x16::splat(NEG32), NEG32 - 1000, -6, -2).to_array();
        assert_eq!(actual[0], NEG32 - 6);
        assert_eq!(actual[15], NEG32 - 6);
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn avx512_maximum_respects_eight_lane_tails_and_boundaries() {
        for len in [8, 16, 24, 32, 40, 48, 56, 72] {
            for maximum in 0..len {
                let mut scores = vec![-100; len];
                scores[maximum] = 123;
                assert_eq!(row_maximum16(&scores), 123);
                scores.fill(NEG32);
                assert_eq!(row_maximum16(&scores), NEG32);
            }
        }
        // Each 8-column logical band can start halfway into a 16-column group.
        // The dedicated body may use unaligned offsets but must not widen it.
        let mut arena = Vec::new();
        let mut matrix = FastMatrix {
            data: &mut arena,
            rows: Vec::new(),
            cursor: 0,
        };
        let row = matrix.allocate(8, 32).unwrap();
        assert_eq!((row.start, row.len), (8, 24));
        store16(matrix.data, row.m_offset, i32x16::splat(7));
        store8(matrix.data, row.m_offset + 16, i32x8::splat(9));
        assert_eq!(
            row_maximum16(&matrix.data[row.m_offset..row.m_offset + row.len]),
            9
        );
        assert_eq!(matrix.data[row.m_offset - 1], NEG32);
        assert_eq!(matrix.data[row.m_offset + row.len], NEG32);
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn avx512_tracebacks_match_scalar_across_vector_and_band_edges() {
        let mut state = 1923801u64;
        let template: Vec<_> = (0..95)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                b"ACGTN"[(state >> 32) as usize % 5]
            })
            .collect();
        let mut cases = 0;
        for len in [
            1, 6, 7, 8, 9, 14, 15, 16, 17, 22, 23, 24, 25, 30, 31, 32, 33, 47, 48, 49, 63, 64, 65,
            79, 80, 81, 95,
        ] {
            let mut params = Params::default();
            if len % 3 == 0 {
                params.gap_open = 0;
            }
            if len % 5 == 0 {
                params.gap_ext = 0;
            }
            let reference = &template[..len];
            let mut graph = linear(&params, reference);
            let mut alternate = reference.to_vec();
            if len > 8 {
                alternate[7] = if alternate[7] == b'A' { b'C' } else { b'A' };
            }
            let alternate_path =
                poa_scalar_i64(&params, &graph, 0, 1, &alternate, false, 100).unwrap();
            graph.add_path(1, &alternate_path).unwrap();
            graph.toposort(false).unwrap();
            let mut insertion = reference.to_vec();
            insertion.splice((len / 2)..(len / 2), b"ACGTACGTA".iter().copied());
            let deletion = reference
                .iter()
                .enumerate()
                .filter_map(|(i, &b)| (i % 13 != 7).then_some(b))
                .collect::<Vec<_>>();
            for query in [
                Vec::new(),
                reference.to_vec(),
                alternate,
                insertion,
                deletion,
            ] {
                for (f, adaptive, band) in [
                    (0, false, 0),
                    (40, false, 0),
                    (40, false, 1),
                    (40, false, 7),
                    (40, false, 8),
                    (40, false, 15),
                    (40, true, 1),
                    (40, true, 8),
                ] {
                    params.f = f;
                    params.adaptive = adaptive;
                    assert_eq!(
                        poa_simd_i32(&params, &graph, 0, 1, &query, adaptive, band),
                        poa_scalar_i64(&params, &graph, 0, 1, &query, adaptive, band),
                        "len={len},query={},f={f},adaptive={adaptive},band={band}",
                        query.len()
                    );
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 1080);
    }
}
