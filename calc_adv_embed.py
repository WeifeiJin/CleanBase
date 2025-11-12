import os
import argparse
import json
from tqdm import tqdm
import torch
import numpy as np
from src.utils import load_models

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i: i + batch_size]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--model_code", type=str, default="contriever-msmarco")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_path", type=str, default='./embeds/adv.npz',
                        help="Path to save texts, UNIQUE ids, and embeddings.")
    args = parser.parse_args()

    _, c_model, tokenizer, get_emb = load_models(args.model_code)
    c_model.eval().to(args.device)

    print(f"Loading data from {args.input_json} and generating unique IDs...")
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    ids = []

    for entry in data.values():
        base_tid = entry.get("id")
        for i, txt in enumerate(entry.get("injected_adv", [])):
            texts.append(txt)
            unique_id = f"{base_tid}-adv-{i}"
            ids.append(unique_id)

    print(f"Processed {len(data)} questions, generated {len(texts)} adversarial texts with unique IDs.")

    all_embs = []
    print("Generating embeddings for all adversarial texts...")
    for batch_texts in tqdm(batchify(texts, args.batch_size), desc="Embedding",
                            total=-(-len(texts) // args.batch_size)):
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(args.device)

        with torch.no_grad():
            embs = get_emb(c_model, inputs)
        all_embs.append(embs.cpu().numpy())

    all_embs = np.vstack(all_embs)

    if args.out_path is not None:
        output_dir = os.path.dirname(args.out_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        np.savez_compressed(
            args.out_path,
            ids=np.array(ids, dtype=object),
            texts=np.array(texts, dtype=object),
            embeddings=all_embs
        )
        print(f"\nSaved: {len(texts)} texts' embeddings with unique IDs to {args.out_path}")


if __name__ == "__main__":
    main()