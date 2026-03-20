use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;

use crate::{DType, GraphNode, Op};

/// Compute a topological ordering of the computation graph nodes.
///
/// # Panics
/// Panics if the graph contains a cycle.
pub fn topo_order<T: DType>(graph: &[GraphNode<T>]) -> Vec<usize> {
    // Build dependency graph
    let mut dep_graph = DiGraphMap::<usize, ()>::new();
    for node in graph.iter() {
        let idx = node.id.get();
        dep_graph.add_node(idx);
    }
    for node in graph.iter() {
        let dst = node.id.get();
        match &node.op {
            Op::BinaryOp { l_id, r_id, .. } => {
                dep_graph.add_edge(l_id.get(), dst, ());
                dep_graph.add_edge(r_id.get(), dst, ());
            }
            Op::UnaryOp { v_id, .. } => {
                dep_graph.add_edge(v_id.get(), dst, ());
            }
            Op::FusedMulAdd { a_id, b_id, c_id } => {
                dep_graph.add_edge(a_id.get(), dst, ());
                dep_graph.add_edge(b_id.get(), dst, ());
                dep_graph.add_edge(c_id.get(), dst, ());
            }
            Op::MatMul {
                l_id, r_id, o_id, ..
            } => {
                dep_graph.add_edge(l_id.get(), dst, ());
                dep_graph.add_edge(r_id.get(), dst, ());
                if let Some(o) = o_id {
                    dep_graph.add_edge(o.get(), dst, ());
                }
            }
            Op::Permute { v_id } => {
                dep_graph.add_edge(v_id.get(), dst, ());
            }
            // No incoming edges for other ops
            _ => {}
        }
    }
    // Compute topological order
    toposort(&dep_graph, None).expect("Cycle detected in graph!")
}
