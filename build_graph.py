import os

import argparse
import time
import numpy as np
import faiss
from scipy.sparse import csr_matrix, save_npz
from utils import *


def main():
    parser = argparse.ArgumentParser(
        description="Build a k-NN similarity graph for the entire or a subset of the database."
    )

    parser.add_argument(
        "--corpus_npz", type=str, required=True,
        help="Path to corpus embeddings .npz."
    )
    parser.add_argument(
        "--adv_npz", type=str, required=True,
        help="Path to adv_texts embeddings .npz."
    )
    parser.add_argument(
        "--num_corpus", type=int, default=None,
        help="Number of corpus documents to use (default: all)."
    )
    parser.add_argument(
        "--num_adv", type=int, default=None,
        help="Number of adversarial documents to use (default: all)."
    )
    parser.add_argument(
        "--k_neighbors", type=int, default=10,
        help="Number of nearest neighbors to connect for each node in the graph."
    )
    parser.add_argument(
        "--output_path", type=str, default='./graph/knn_graph.npz',
        help="Path to save the output sparse graph and IDs."
    )
    args = parser.parse_args()

    start_total = time.time()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading data...")

    full_corpus_ids, full_corpus_embs = load_ids_embs(args.corpus_npz)
    if full_corpus_embs is None: return

    full_adv_ids, full_adv_embs = load_ids_embs(args.adv_npz)
    if full_adv_embs is None: return

    num_corpus_to_use = len(full_corpus_ids) if args.num_corpus is None else min(args.num_corpus, len(full_corpus_ids))
    num_adv_to_use = len(full_adv_ids) if args.num_adv is None else min(args.num_adv, len(full_adv_ids))

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Selecting {num_corpus_to_use} corpus samples and {num_adv_to_use} adversarial samples.")

    corpus_ids = full_corpus_ids[:num_corpus_to_use]
    corpus_embs = full_corpus_embs[:num_corpus_to_use]

    adv_ids = full_adv_ids[:num_adv_to_use]
    adv_embs = full_adv_embs[:num_adv_to_use]

    combined_ids = np.concatenate([corpus_ids, adv_ids])
    combined_embs = np.vstack([corpus_embs, adv_embs]).astype('float32')
    num_nodes = len(combined_ids)
    dim = combined_embs.shape[1]
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total nodes in graph: {num_nodes}")


    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Normalizing embeddings...")
    norms = np.linalg.norm(combined_embs, axis=1, keepdims=True)
    combined_embs_normed = combined_embs / (norms + 1e-8)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Building FAISS index...")
    nlist = int(np.sqrt(num_nodes))
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(combined_embs_normed)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAISS index is ready.")

    k_search = args.k_neighbors + 1
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Searching for {k_search} nearest neighbors for all {num_nodes} nodes...")
    t_search_start = time.time()

    distances, indices = index.search(combined_embs_normed, k_search)
    t_search_end = time.time()
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAISS search completed in {t_search_end - t_search_start:.2f} seconds.")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Constructing sparse adjacency matrix...")
    row_ind = []
    col_ind = []
    data = []

    for i in range(num_nodes):
        for j_local, neighbor_idx in enumerate(indices[i][1:]):
            if neighbor_idx == -1:
                continue
            row_ind.append(i)
            col_ind.append(neighbor_idx)
            dist_sq = distances[i][j_local + 1]
            similarity = 1 - (dist_sq / 2)
            data.append(similarity)

    adjacency_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(num_nodes, num_nodes))

    if args.output_path is not None:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving graph to {args.output_path}...")
        save_npz(args.output_path, adjacency_matrix)

        ids_path = args.output_path.replace('.npz', '_ids.npy')
        np.save(ids_path, combined_ids)
        print(f"  - Graph (Adjacency Matrix): {args.output_path}")
        print(f"  - Node IDs: {ids_path}")

    end_total = time.time()
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Graph construction complete. Total time: {end_total - start_total:.2f} seconds.")
    print(f"Output files:")


if __name__ == "__main__":
    main()