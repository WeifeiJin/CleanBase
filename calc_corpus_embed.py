import json
import torch
import numpy as np
from tqdm import tqdm
from src.utils import load_models
import argparse

def batchify(iterator, batch_size):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--model_code", type=str, default="contriever-msmarco",
                        help="load_models")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_path", type=str, default='./embeds/corpus.npz',
                        help="ids, titles, texts, embeddings")
    args = parser.parse_args()

    with open(args.corpus_path, "r", encoding="utf-8") as f:
        total_docs = sum(1 for _ in f)

    _, c_model, tokenizer, get_emb = load_models(args.model_code)
    c_model.eval().to(args.device)

    ids, titles, texts, embs = [], [], [], []

    pbar = tqdm(total=total_docs, desc="Embedding docs", unit="doc")
    with open(args.corpus_path, "r", encoding="utf-8") as fin:
        for batch in batchify(fin, args.batch_size):
            docs = [json.loads(line) for line in batch]
            titles_batch = [d.get("title", "") for d in docs]
            texts_batch = [d.get("text", "") for d in docs]

            inputs = tokenizer(
                texts_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(args.device)

            with torch.no_grad():
                batch_emb = get_emb(c_model, inputs)
            batch_emb = batch_emb.cpu().numpy()

            for d, title, text, e in zip(docs, titles_batch, texts_batch, batch_emb):
                ids.append(d["_id"])
                titles.append(title)
                texts.append(text)
                embs.append(e)

            pbar.update(len(batch))

    pbar.close()

    embs = np.stack(embs, axis=0)
    if args.out_path is not None:
        np.savez(
            args.out_path,
            ids=np.array(ids, dtype=object),
            titles=np.array(titles, dtype=object),
            texts=np.array(texts, dtype=object),
            embeddings=embs
        )

        print(f"\nSaved {len(ids)} docs (ids, titles, texts, embeddings) to {args.out_path}")

if __name__ == "__main__":
    main()
