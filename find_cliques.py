import os
import argparse
import time
import numpy as np
import networkx as nx
from scipy.sparse import load_npz
from tqdm import tqdm
import json

from utils import *


def main():
    parser = argparse.ArgumentParser(
        description="Find, analyze, and evaluate a clique-based defense strategy, and save all clique details."
    )

    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--ids_path", type=str, required=True)
    parser.add_argument("--adv_npz_path", type=str, required=True)

    parser.add_argument("--min_clique_size", type=int, default=3)
    parser.add_argument("--max_clique_size", type=int, default=11)

    parser.add_argument("--prefilter_density_threshold", type=float, default=1.00, help="Threshold to pre-filter trusted cliques.")

    parser.add_argument("--output_json", type=str, default='./results/cliques.json')
    parser.add_argument("--max_cliques_to_save", type=int, default=-1,
                        help="Maximum number of analyzed cliques to save in the output file. Set to -1 for unlimited.")

    args = parser.parse_args()
    start_time = time.time()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading data...")
    try:
        adjacency_matrix = load_npz(args.graph_path)
        graph_node_ids = np.load(args.ids_path, allow_pickle=True)
        adv_ids_full_list, = load_npz_data_from_keys(args.adv_npz_path, keys=('ids',))
        if adv_ids_full_list is None: raise FileNotFoundError
    except FileNotFoundError:
        print(f"Error: A required input file was not found. Please check paths.")
        return

    adv_id_set = set(adv_ids_full_list)
    ground_truth_is_adv = np.array([node_id in adv_id_set for node_id in graph_node_ids])
    id_to_node_index_map = {id_str: i for i, id_str in enumerate(graph_node_ids)}

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Creating graph object...")
    G = nx.from_scipy_sparse_array(adjacency_matrix, create_using=nx.Graph)
    num_nodes = G.number_of_nodes()

    print(f"\n--- [Phase 1: Analyzing all cliques of size [{args.min_clique_size}, {args.max_clique_size}]] ---")
    all_analyzed_cliques = []
    cliques_iterator = nx.find_cliques(G)

    for clique_indices in tqdm(cliques_iterator, desc="Analyzing Cliques"):
        clique_size = len(clique_indices)
        if not (args.min_clique_size <= clique_size <= args.max_clique_size):
            continue

        clique_subgraph = G.subgraph(clique_indices)
        num_edges = clique_subgraph.number_of_edges()
        if num_edges == 0: continue

        edge_weights = [data['weight'] for _, _, data in clique_subgraph.edges(data=True)]
        avg_weight = np.mean(edge_weights) if edge_weights else 0
        std_weight = np.std(edge_weights) if edge_weights else 0
        num_adv_nodes = np.sum(ground_truth_is_adv[clique_indices])
        adv_ratio = num_adv_nodes / clique_size

        all_analyzed_cliques.append({
            "size": clique_size,
            "avg_edge_weight": round(avg_weight, 4),
            "std_edge_weight": round(std_weight, 4),
            "adversarial_ratio": round(adv_ratio, 4),
            "member_indices": clique_indices
        })
    print(f"Analysis complete. Found and analyzed {len(all_analyzed_cliques)} cliques matching the size criteria.")

    print(f"\n--- [Phase 2: Applying detection rule] ---")

    print(f"Rule: Pre-filter out cliques with avg_edge_weight >= {args.prefilter_density_threshold}")
    detection_candidates = [c for c in all_analyzed_cliques if c['avg_edge_weight'] < args.prefilter_density_threshold]
    print(
        f"Pre-filtered {len(all_analyzed_cliques) - len(detection_candidates)} cliques. {len(detection_candidates)} cliques remain.")

    predicted_as_malicious = np.zeros(num_nodes, dtype=bool)

    for clique_info in detection_candidates:
        predicted_as_malicious[clique_info['member_indices']] = True

    print(f"Flagged all nodes belonging to the {len(detection_candidates)} remaining cliques.")

    print(f"\n--- [Phase 3: Performance Evaluation] ---")
    tp = np.sum(np.logical_and(predicted_as_malicious, ground_truth_is_adv))
    fp = np.sum(np.logical_and(predicted_as_malicious, ~ground_truth_is_adv))
    tn = np.sum(np.logical_and(~predicted_as_malicious, ~ground_truth_is_adv))
    fn = np.sum(np.logical_and(~predicted_as_malicious, ground_truth_is_adv))

    total_adv = np.sum(ground_truth_is_adv)
    total_corpus = num_nodes - total_adv
    accuracy = (tp + tn) / num_nodes if num_nodes > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(precision, recall)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    end_time = time.time()

    print("\n" + "=" * 80)
    print("Clique-based Defense (Prefilter-Only): Full Evaluation & Analysis Report")
    print("=" * 80)

    print(f"Detection Rule (Prefilter Only):")
    print(f"  - Find all cliques with size in [{args.min_clique_size}, {args.max_clique_size}].")
    print(f"  - Filter out cliques with avg_edge_weight >= {args.prefilter_density_threshold}.")
    print(f"  - Flag ALL nodes in remaining cliques as suspicious.")
    print("-" * 80)

    print(f"Total Nodes: {num_nodes} (Corpus: {total_corpus}, Adversarial: {total_adv})")
    print(f"Total nodes predicted as malicious: {tp + fp}")

    print("\n--- Confusion Matrix ---")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")

    print("\n--- Performance Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (TPR):    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print("=" * 80)

    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds.")

    for clique in all_analyzed_cliques:
        clique["member_ids"] = [graph_node_ids[i] for i in clique.pop("member_indices")]

    full_report = {
        "detection_rule": {
            "type": "clique_with_prefilter",
            "clique_size_range": f"[{args.min_clique_size}, {args.max_clique_size}]",
            "prefilter_density_threshold": args.prefilter_density_threshold
        },
        "metrics": {
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_positive_rate": float(fpr)
        },
        "analyzed_cliques_count": len(all_analyzed_cliques),
        "analyzed_cliques_details": []
    }

    cliques_to_save = all_analyzed_cliques
    if args.max_cliques_to_save != -1:
        all_analyzed_cliques.sort(key=lambda x: x['avg_edge_weight'])
        cliques_to_save = all_analyzed_cliques[:args.max_cliques_to_save]

    full_report["analyzed_cliques_details"] = cliques_to_save

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(
            f"\nFull analysis report, including details of {len(cliques_to_save)} cliques, saved to {args.output_json}")


if __name__ == "__main__":
    main()