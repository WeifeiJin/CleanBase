import os

import argparse
import time
import numpy as np
import json
from tqdm import tqdm
from utils import *

def main():
    parser = argparse.ArgumentParser(
        description="Filter a database by removing all nodes that appear in any clique from a report file."
    )
    parser.add_argument(
        "--database_npz", type=str, required=True,
        help="Path to the full database NPZ file to be filtered."
    )
    parser.add_argument(
        "--cliques_json_path", type=str, required=True,
        help="Path to the JSON file containing the list of cliques."
    )
    parser.add_argument(
        "--output_cleaned_npz", type=str, default="./embeds/database_cleaned.npz",
        help="Path to save the new, cleaned database .npz file."
    )

    args = parser.parse_args()
    start_time = time.time()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading all data sources...")

    combined_ids, combined_texts, combined_embs = load_npz_data(args.database_npz)
    if combined_ids is None:
        return

    try:
        with open(args.cliques_json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)

        if 'metrics' in analysis_data:
            print("\n--- [Metrics from Input Report File] ---")
            metrics = analysis_data['metrics']
            print(f"  - True Positives (TP): {metrics.get('tp', 'N/A')}")
            print(f"  - False Positives (FP): {metrics.get('fp', 'N/A')}")
            print(f"  - True Negatives (TN): {metrics.get('tn', 'N/A')}")
            print(f"  - False Negatives (FN): {metrics.get('fn', 'N/A')}")
            print(f"  - Recall (TPR): {metrics.get('recall', 'N/A'):.4f}")
            print(f"  - False Positive Rate (FPR): {metrics.get('false_positive_rate', 'N/A'):.4f}")
            print("----------------------------------------\n")

        possible_keys = ['analyzed_cliques_details', 'analyzed_cliques_examples']
        cliques_list = None
        for key in possible_keys:
            if key in analysis_data:
                cliques_list = analysis_data[key]
                break

        if not cliques_list:
            print(f"Error: No valid clique list found in {args.cliques_json_path}")
            return

        nodes_to_remove_set = set()
        for clique in tqdm(cliques_list, desc="Collecting all member IDs from cliques"):
            member_ids = clique.get("member_ids", [])
            nodes_to_remove_set.update(member_ids)

        print(f"Collected {len(nodes_to_remove_set)} unique node IDs from {len(cliques_list)} cliques to be filtered.")
    except FileNotFoundError:
        print(f"Error: Cliques JSON file not found at {args.cliques_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {args.cliques_json_path}")
        return

    print(f"\n--- [Filtering Database] ---")

    clean_mask = np.array(
        [node_id not in nodes_to_remove_set for node_id in tqdm(combined_ids, desc="Creating filter mask")])

    cleaned_ids = combined_ids[clean_mask]
    cleaned_texts = combined_texts[clean_mask]
    cleaned_embs = combined_embs[clean_mask]

    original_size = len(combined_ids)
    cleaned_size = len(cleaned_ids)
    num_removed = original_size - cleaned_size

    print("\n--- [Filtering Complete] ---")
    print(f"Original database size: {original_size}")
    print(f"Nodes identified for removal: {len(nodes_to_remove_set)}")
    print(f"Nodes actually removed: {num_removed}")
    print(f"Final cleaned database size: {cleaned_size}")

    output_dir = os.path.dirname(args.output_cleaned_npz)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving cleaned database to {args.output_cleaned_npz}...")

    np.savez_compressed(
        args.output_cleaned_npz,
        ids=cleaned_ids,
        texts=cleaned_texts,
        embeddings=cleaned_embs
    )

    end_time = time.time()
    print("\nProcess complete.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()