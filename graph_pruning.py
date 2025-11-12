import os

import argparse
import time
import numpy as np
import networkx as nx
from scipy.sparse import load_npz, save_npz
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Filter edges from a pre-built DIRECTED graph based on global edge weight statistics."
    )
    parser.add_argument(
        "--input_graph_path", type=str, required=True,
        help="Path to the input sparse graph adjacency matrix (.npz)."
    )
    parser.add_argument(
        "--input_ids_path", type=str, required=True,
        help="Path to the input node IDs array (.npy)."
    )
    parser.add_argument(
        "--alpha", type=float, default=2.5,
        help="The number of standard deviations above the mean to set the pruning threshold (Threshold = mean + alpha * std)."
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=0.5,
        help="The ratio of edges to sample for calculating statistics (e.g., 0.5 for 50%)."
    )

    parser.add_argument(
        "--output_graph_path", type=str, default='./graph/graph_pruned.npz',
        help="Path to save the new, pruned sparse graph."
    )
    args = parser.parse_args()

    start_time = time.time()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading graph data...")
    try:
        adjacency_matrix = load_npz(args.input_graph_path)
        graph_node_ids = np.load(args.input_ids_path, allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error: A required input file was not found. Details: {e}")
        return

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Creating NetworkX DiGraph object...")
    G_directed = nx.from_scipy_sparse_array(adjacency_matrix, create_using=nx.DiGraph)
    G_undirected = nx.from_scipy_sparse_array(adjacency_matrix, create_using=nx.Graph)
    print(f"Original directed graph loaded with {G_directed.number_of_nodes()} nodes and {G_directed.number_of_edges()} edges.")

    print(f"\n--- [Phase 1: Sampling edge weights from the existing directed graph] ---")

    all_edge_weights = [data['weight'] for u, v, data in tqdm(G_undirected.edges(data=True), desc="Extracting weights")]
    all_edge_weights = np.array(all_edge_weights)

    if len(all_edge_weights) == 0:
        print("Graph has no edges. Nothing to filter. Exiting.")
        return

    sample_size = int(len(all_edge_weights) * args.sample_ratio)
    edge_weight_sample = np.random.choice(all_edge_weights, size=sample_size, replace=False)

    mean_weight = np.mean(edge_weight_sample)
    std_weight = np.std(edge_weight_sample)

    prune_threshold = mean_weight + args.alpha * std_weight

    print(f"\nSampled {sample_size} edges ({(args.sample_ratio * 100):.1f}%) to calculate statistics.")
    print(f"Sample statistics: Mean={mean_weight:.4f}, Std={std_weight:.4f}")
    print(f"Calculated pruning threshold (mean + {args.alpha}*std): {prune_threshold:.4f}")

    print(f"\n--- [Phase 2: Constructing final pruned graph] ---")

    edges_to_keep = []
    for u, v, data in tqdm(G_directed.edges(data=True), desc="Filtering edges"):
        weight = data.get('weight', 0)

        if weight > prune_threshold:
            edges_to_keep.append((u, v, weight))

    G_pruned = nx.DiGraph()
    G_pruned.add_nodes_from(G_directed.nodes())
    G_pruned.add_weighted_edges_from(edges_to_keep)

    filtered_adjacency_matrix = nx.to_scipy_sparse_array(G_pruned, nodelist=sorted(G_directed.nodes()), format='csr')

    output_dir = os.path.dirname(args.output_graph_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    save_npz(args.output_graph_path, filtered_adjacency_matrix)
    filtered_ids_path = args.output_graph_path.replace('.npz', '_ids.npy')
    np.save(filtered_ids_path, graph_node_ids)

    end_time = time.time()

    print("\n--- [Process Complete] ---")
    print(f"Original directed graph edges: {G_directed.number_of_edges()}")
    print(f"Final pruned directed graph edges: {G_pruned.number_of_edges()}")
    print(f"Filtered graph saved to: {args.output_graph_path}")
    print(f"Corresponding IDs saved to: {filtered_ids_path}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()