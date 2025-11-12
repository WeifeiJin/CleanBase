import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import torch
from collections import defaultdict

from src.models import create_model
from src.utils import load_beir_datasets, load_models, save_results, load_json, setup_seeds, clean_str
from src.prompts import wrap_prompt, wrap_judge_prompt
from utils import *


def get_topk_docs(query_emb, corpus_embs, k):
    corpus_embs_tensor = torch.tensor(corpus_embs, device=query_emb.device)
    query_emb_normed = torch.nn.functional.normalize(query_emb, p=2, dim=1)
    corpus_embs_normed = torch.nn.functional.normalize(corpus_embs_tensor, p=2, dim=1)
    cos_sims = torch.mm(query_emb_normed, corpus_embs_normed.T).squeeze(0)
    topk_scores, topk_indices = torch.topk(cos_sims, k=min(k, len(cos_sims)))
    return topk_indices.cpu().numpy(), topk_scores.cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RAG performance on a static, pre-filtered database')

    parser.add_argument("--database_path", type=str, required=True)
    parser.add_argument("--adv_ids_path", type=str, required=True)

    parser.add_argument("--eval_model_code", type=str, default="contriever-msmarco")
    parser.add_argument('--eval_dataset', type=str, default="nq")

    parser.add_argument('--split', type=str, default='test', help="Dataset split, e.g., 'test' or 'dev'.")
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='llama7b')
    parser.add_argument('--judge_model_name', type=str, default='gpt4')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--repeat_times', type=int, default=1)
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument("--output_json", type=str, default='./results/eval_pipeline_report.json')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    setup_seeds(args.seed)
    if args.model_config_path is None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'
        args.judge_model_config_path = f'model_configs/{args.judge_model_name}_config.json'

    print("Loading datasets and models...")
    _, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

    incorrect_answers = load_json(f'results/PoisonedRAG/{args.eval_dataset}.json')
    incorrect_answers = list(incorrect_answers.values())

    print(f"Loading final database from: {args.database_path}")
    final_db_data = np.load(args.database_path, allow_pickle=True)
    db_ids, db_texts, db_embs = final_db_data["ids"], final_db_data["texts"], final_db_data["embeddings"]

    print(f"Loading adversarial IDs for evaluation from: {args.adv_ids_path}")
    adv_ids_data = np.load(args.adv_ids_path, allow_pickle=True)
    adv_id_set = set(adv_ids_data["ids"])
    adv_counts_per_query = defaultdict(int)
    for adv_id in adv_ids_data["ids"]:
        adv_counts_per_query[adv_id] += 1

    model, _, tokenizer, get_emb = load_models(args.eval_model_code)
    model.eval().to(device)
    llm = create_model(args.model_config_path)
    judger = create_model(args.judge_model_config_path)

    all_results, asr_list, ret_list = [], [], []

    for iter_num in range(args.repeat_times):
        print(f'\n######################## Iteration: {iter_num + 1}/{args.repeat_times} #######################')
        target_queries_idx = range(iter_num * args.M, (iter_num + 1) * args.M)
        asr_cnt, ret_sublist = 0, []
        iter_results = []
        for i in tqdm(target_queries_idx, desc=f"Evaluating Iter {iter_num + 1}"):
            if i >= len(incorrect_answers): continue

            query_info = incorrect_answers[i]
            question, qid, incco_ans = query_info['question'], query_info['id'], query_info['incorrect answer']

            if qid not in queries: continue

            try:
                tokens = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
                if tokens["input_ids"].max() >= tokenizer.vocab_size or tokens["input_ids"].min() < 0:
                    print(f"[SKIP] invalid tokens in: {question[:120]}")
                    continue
                query_input = tokens.to(device)
            except Exception as e:
                print(f"[ERROR] tokenizer failed for: {question[:120]} \n{e}")
                continue

            with torch.no_grad():
                query_emb = get_emb(model, query_input)

            topk_indices, _ = get_topk_docs(query_emb, db_embs, k=args.top_k)

            topk_ids = db_ids[topk_indices]
            topk_contents_array = db_texts[topk_indices]
            topk_contents = topk_contents_array.tolist()

            cnt_from_adv = sum(1 for doc_id in topk_ids if doc_id in adv_id_set)
            ret_sublist.append(cnt_from_adv)

            query_prompt = wrap_prompt(question, topk_contents)
            response = llm.query(query_prompt)
            print(f'Response: {response}')
            judge_prompt = wrap_judge_prompt(question, incco_ans, response)
            judge_res = judger.query(judge_prompt)
            print(f'Judge Response: {judge_res}')
            is_attack_successful = 'no'
            if 'yes' in extract_boolean_prefix(judge_res).lower():
                asr_cnt += 1
                is_attack_successful = 'yes'

            topk_ids_list = topk_ids.tolist()

            iter_results.append({
                "id": qid,
                "question": question,
                "retrieved_topk_ids": topk_ids_list,
                "llm_response": response,
                "incorrect_answer_target": incco_ans,
                "judge_model_response": judge_res.strip(),
                "is_attack_successful": is_attack_successful
            })

        asr_list.append(asr_cnt)
        ret_list.append(ret_sublist)

        if args.output_json:
            output_dir = os.path.dirname(args.output_json)
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(iter_results, f, indent=2, ensure_ascii=False)
            print(f"saved to {args.output_json}")

    print("\n" + "=" * 80)
    print("Final RAG Performance Evaluation on Pre-Filtered Database")
    print("=" * 80)

    asr = np.array(asr_list) / args.M
    asr_mean = round(np.mean(asr), 4)

    flat_ret_list = np.array([item for sublist in ret_list for item in sublist])

    total_adv_for_tested_queries = 500
    ret_precision_mean = round(np.mean(flat_ret_list / args.top_k), 4)

    ret_recall_mean = round(
        np.sum(flat_ret_list) / total_adv_for_tested_queries if total_adv_for_tested_queries > 0 else 0, 4)
    ret_f1_mean = round(f1_score(ret_precision_mean, ret_recall_mean), 4)

    print(f"ASR Mean: {asr_mean:.4f} (Lower is better)\n")
    print(f"--- Adversarial Document Retrieval Metrics ---")
    print(f"Retrieval Precision Mean: {ret_precision_mean}")
    print(f"Retrieval Recall Mean: {ret_recall_mean}")
    print(f"Retrieval F1 Mean: {ret_f1_mean}\n")


if __name__ == '__main__':
    main()