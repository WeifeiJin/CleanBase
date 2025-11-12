import os
import argparse
import time
import numpy as np
from utils import *


def main():
    parser = argparse.ArgumentParser(
        description="Combine a corpus dataset and an adversarial dataset into a single 'attacked' database .npz file."
    )
    parser.add_argument(
        "--corpus_npz", type=str, required=True,
        help="Path to the input corpus NPZ file (must contain ids, texts, embeddings)."
    )
    parser.add_argument(
        "--adv_npz", type=str, required=True,
        help="Path to the input adversarial NPZ file (must contain ids, texts, embeddings)."
    )
    parser.add_argument(
        "--output_npz", type=str, default='./embeds/database_attacked.npz',
        help="Path for the output combined (attacked) NPZ file."
    )
    args = parser.parse_args()

    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting dataset merge process...")

    corpus_ids, corpus_texts, corpus_embs = load_npz_data(args.corpus_npz)
    adv_ids, adv_texts, adv_embs = load_npz_data(args.adv_npz)

    if corpus_embs is None or adv_embs is None:
        print("Aborting due to file loading errors.")
        return

    print("\n" + "-" * 50)
    print("Merging datasets...")
    print(f"Corpus documents: {len(corpus_ids)}")
    print(f"Adversarial documents: {len(adv_ids)}")

    combined_ids = np.concatenate([corpus_ids, adv_ids])
    combined_texts = np.concatenate([corpus_texts, adv_texts])

    combined_embs = np.vstack([corpus_embs, adv_embs])

    total_docs = len(combined_ids)
    print(f"Total documents in the new combined database: {total_docs}")
    print(f"Shape of combined embeddings: {combined_embs.shape}")
    print("-" * 50 + "\n")

    output_dir = os.path.dirname(args.output_npz)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving combined data to {args.output_npz}...")

    np.savez_compressed(
        args.output_npz,
        ids=combined_ids,
        texts=combined_texts,
        embeddings=combined_embs
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Save complete.")

    print("\nVerifying saved file...")
    try:
        saved_data = np.load(args.output_npz, allow_pickle=True)
        if len(saved_data['ids']) == total_docs:
            print("Verification successful: Number of items in saved file matches the combined total.")
        else:
            print("Verification FAILED: Number of items mismatch.")
    except Exception as e:
        print(f"Verification failed with an error: {e}")

    end_time = time.time()
    print(f"\nTotal elapsed time: {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()