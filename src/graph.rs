//! Weighted POA DAG with stable node IDs and substitution-column groups.
use crate::{AlignedBase, Params, Result, Sequence};
use std::collections::VecDeque;
use std::fmt::Write;

#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub rank: usize,
    pub parent: usize,
    pub base: usize,
    pub incoming: Vec<usize>,
    pub in_weights: Vec<usize>,
    pub outgoing: Vec<usize>,
    pub out_weights: Vec<usize>,
    pub aligned: Vec<Option<usize>>,
    pub coverage: usize,
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub rank: Vec<usize>,
    /// Sequence IDs and their paths, in alignment order.
    pub paths: Vec<(usize, Vec<usize>)>,
    alphabet_size: usize,
}

impl Graph {
    pub fn new(params: &Params) -> Self {
        let mut graph = Self {
            nodes: Vec::new(),
            rank: Vec::new(),
            paths: Vec::new(),
            alphabet_size: params.alphabet.len(),
        };
        graph.add_node(params.encode(b'N'));
        graph.add_node(params.encode(b'N'));
        graph
    }

    pub fn add_node(&mut self, base: usize) -> usize {
        let id = self.nodes.len();
        let mut aligned = vec![None; self.alphabet_size];
        aligned[base] = Some(id);
        self.nodes.push(Node {
            id,
            rank: 0,
            parent: id,
            base,
            incoming: Vec::new(),
            in_weights: Vec::new(),
            outgoing: Vec::new(),
            out_weights: Vec::new(),
            aligned,
            coverage: 0,
        });
        id
    }

    /// Add one observation of an edge. Self edges are ignored as in upstream.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        if let Some(i) = self.nodes[from].outgoing.iter().position(|&n| n == to) {
            self.nodes[from].out_weights[i] += 1;
        } else {
            self.nodes[from].outgoing.push(to);
            self.nodes[from].out_weights.push(1);
        }
        if let Some(i) = self.nodes[to].incoming.iter().position(|&n| n == from) {
            self.nodes[to].in_weights[i] += 1;
        } else {
            self.nodes[to].incoming.push(from);
            self.nodes[to].in_weights.push(1);
        }
        self.nodes[to].coverage += 1;
    }

    /// The alignment starts with source node 0 and excludes sink node 1.
    pub fn add_path(&mut self, seq_id: usize, alignment: &[AlignedBase]) -> Result<()> {
        for item in alignment {
            if item.base >= self.alphabet_size {
                return Err("alignment base outside alphabet".into());
            }
            if item.node.is_some_and(|n| n >= self.nodes.len())
                || item.aligned.is_some_and(|n| n >= self.nodes.len())
            {
                return Err("alignment references an unknown node".into());
            }
        }
        let mut previous = None;
        let mut path = Vec::with_capacity(alignment.len());
        for item in alignment {
            let id = if let Some(id) = item.node {
                id
            } else if let Some(parent) = item.aligned {
                let parent = self.nodes[parent].parent;
                if let Some(id) = self.nodes[parent].aligned[item.base] {
                    id
                } else {
                    let id = self.add_node(item.base);
                    self.nodes[id].parent = parent;
                    self.nodes[parent].aligned[item.base] = Some(id);
                    id
                }
            } else {
                self.add_node(item.base)
            };
            if let Some(prev) = previous {
                self.add_edge(prev, id);
            }
            previous = Some(id);
            if id >= 2 {
                path.push(id);
            }
        }
        self.add_edge(previous.unwrap_or(0), 1);
        self.paths.push((seq_id, path));
        Ok(())
    }

    /// Stable LIFO Kahn ordering matches upstream's traversal and tie order.
    pub fn toposort(&mut self, grouped: bool) -> Result<()> {
        let mut degree = vec![0; self.nodes.len()];
        let mut stack = Vec::new();
        for n in &self.nodes {
            if grouped && n.parent != n.id {
                continue;
            }
            degree[n.id] = if grouped {
                n.aligned
                    .iter()
                    .flatten()
                    .map(|&i| self.nodes[i].incoming.len())
                    .sum()
            } else {
                n.incoming.len()
            };
            if degree[n.id] == 0 {
                stack.push(n.id);
            }
        }
        self.rank.clear();
        let mut visited = 0;
        while let Some(id) = stack.pop() {
            let rank = self.rank.len();
            self.rank.push(id);
            let members = if grouped {
                self.nodes[id].aligned.iter().flatten().copied().collect()
            } else {
                vec![id]
            };
            for member in members {
                self.nodes[member].rank = rank;
                visited += 1;
                for &next in &self.nodes[member].outgoing {
                    let next = if grouped {
                        self.nodes[next].parent
                    } else {
                        next
                    };
                    if degree[next] == 0 {
                        return Err("invalid graph column dependencies".into());
                    }
                    degree[next] -= 1;
                    if degree[next] == 0 {
                        stack.push(next);
                    }
                }
            }
        }
        if visited != self.nodes.len() {
            return Err("graph contains a cycle (possibly between alignment columns)".into());
        }
        Ok(())
    }

    /// Heaviest incoming edge, breaking ties by accumulated path support.
    pub fn consensus(&self, params: &Params) -> (Vec<u8>, Vec<usize>, Vec<usize>) {
        let mut degree: Vec<_> = self.nodes.iter().map(|n| n.incoming.len()).collect();
        let mut queue = VecDeque::from([0]);
        let mut previous = vec![None; self.nodes.len()];
        let mut score = vec![0usize; self.nodes.len()];
        while let Some(id) = queue.pop_front() {
            let node = &self.nodes[id];
            if id != 0 {
                let best = node
                    .incoming
                    .iter()
                    .enumerate()
                    .max_by(|&(ai, &a), &(bi, &b)| {
                        node.in_weights[ai]
                            .cmp(&node.in_weights[bi])
                            .then(score[a].cmp(&score[b]))
                            .then(bi.cmp(&ai))
                    });
                if let Some((i, &p)) = best {
                    previous[id] = Some(p);
                    score[id] = score[p] + node.in_weights[i];
                }
            }
            if id == 1 {
                break;
            }
            for &next in &node.outgoing {
                degree[next] -= 1;
                if degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }
        let mut ids = Vec::new();
        let mut id = previous[1];
        while let Some(current) = id {
            if current == 0 {
                break;
            }
            ids.push(current);
            id = previous[current];
        }
        ids.reverse();
        let sequence = ids
            .iter()
            .map(|&i| params.decode(self.nodes[i].base))
            .collect();
        let coverage = ids
            .iter()
            .map(|&i| {
                self.nodes[i]
                    .coverage
                    .max(self.nodes[i].out_weights.iter().sum())
            })
            .collect();
        (sequence, ids, coverage)
    }

    pub fn msa(&mut self, params: &Params, sequences: &[Sequence]) -> Result<String> {
        if self.paths.is_empty() {
            return Ok(String::new());
        }
        let rank = self.rank.clone();
        let node_ranks: Vec<_> = self.nodes.iter().map(|node| node.rank).collect();
        let result = self
            .toposort(true)
            .and_then(|()| self.msa_grouped(params, sequences));
        self.rank = rank;
        for (node, rank) in self.nodes.iter_mut().zip(node_ranks) {
            node.rank = rank;
        }
        result
    }

    fn msa_grouped(&self, params: &Params, sequences: &[Sequence]) -> Result<String> {
        let mut output = String::new();
        let width = self.rank.len().saturating_sub(2);
        for (sid, path) in &self.paths {
            let seq = sequences
                .get(*sid)
                .ok_or("unknown sequence ID in graph path")?;
            let mut row = vec![b'-'; width];
            for &id in path {
                let rank = self.nodes[self.nodes[id].parent].rank;
                if rank == 0 || rank > width {
                    return Err("graph path is outside MSA columns".into());
                }
                row[rank - 1] = params.decode(self.nodes[id].base);
            }
            writeln!(output, ">{} {}", seq.name, seq.comment).unwrap();
            writeln!(
                output,
                "{}",
                String::from_utf8(row).map_err(|e| e.to_string())?
            )
            .unwrap();
        }
        Ok(output)
    }

    pub fn gfa(&self, params: &Params, sequences: &[Sequence]) -> Result<String> {
        if self.nodes.len() <= 2 && self.paths.is_empty() {
            return Ok(String::new());
        }
        let mut output = String::from("H\tVN:Z:1.0\n");
        let mut degree: Vec<_> = self.nodes.iter().map(|n| n.incoming.len()).collect();
        let mut queue = VecDeque::from([0]);
        while let Some(id) = queue.pop_front() {
            if id == 1 {
                break;
            }
            let node = &self.nodes[id];
            if id >= 2 {
                writeln!(
                    output,
                    "S\t{}\t{}",
                    id - 1,
                    params.decode(node.base) as char
                )
                .unwrap();
                for &prev in &node.incoming {
                    if prev != 0 {
                        writeln!(output, "L\t{}\t+\t{}\t+\t0M", prev - 1, id - 1).unwrap();
                    }
                }
            }
            for &next in &node.outgoing {
                degree[next] -= 1;
                if degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }
        for (sid, path) in &self.paths {
            let seq = sequences
                .get(*sid)
                .ok_or("unknown sequence ID in graph path")?;
            write!(output, "P\t{}\t", seq.name).unwrap();
            if path.is_empty() {
                output.push('*');
            }
            for (i, &id) in path.iter().enumerate() {
                if i > 0 {
                    output.push(',');
                }
                write!(output, "{}+", id - 1).unwrap();
            }
            output.push_str("\t*\n");
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn base(node: Option<usize>, base: usize, aligned: Option<usize>) -> AlignedBase {
        AlignedBase {
            node,
            base,
            aligned,
        }
    }
    #[test]
    fn weights_and_consensus_ties() {
        let p = Params::default();
        let mut g = Graph::new(&p);
        g.add_path(
            0,
            &[
                base(Some(0), 4, None),
                base(None, 0, None),
                base(None, 1, None),
            ],
        )
        .unwrap();
        g.add_path(
            1,
            &[
                base(Some(0), 4, None),
                base(Some(2), 0, None),
                base(None, 2, Some(3)),
            ],
        )
        .unwrap();
        g.add_path(
            2,
            &[
                base(Some(0), 4, None),
                base(Some(2), 0, None),
                base(Some(4), 2, None),
            ],
        )
        .unwrap();
        g.toposort(false).unwrap();
        assert_eq!(g.consensus(&p).0, b"AG");
        assert_eq!(g.consensus(&p).2, vec![3, 2]);
        assert_eq!(g.nodes[2].coverage, 3);
        assert_eq!(g.nodes[3].aligned[2], Some(4));
    }
    #[test]
    fn rejects_cycles() {
        let p = Params::default();
        let mut g = Graph::new(&p);
        let a = g.add_node(0);
        let b = g.add_node(1);
        g.add_edge(0, a);
        g.add_edge(a, b);
        g.add_edge(b, a);
        g.add_edge(b, 1);
        assert!(g.toposort(false).unwrap_err().contains("cycle"));
    }
}
