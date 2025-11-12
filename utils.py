import numpy as np
import re

def load_ids_embs(npz_path: str):
    try:
        data = np.load(npz_path, allow_pickle=True)
        return data["ids"], data["embeddings"]
    except FileNotFoundError:
        print(f"Error: File not found at {npz_path}")
        return None, None

def load_npz_data(npz_path: str):
    print(f"Loading data from: {npz_path}")
    try:
        data = np.load(npz_path, allow_pickle=True)
        required_keys = ["ids", "texts", "embeddings"]
        if not all(key in data for key in required_keys):
            if "embedding" in data:
                data = {"ids": data["ids"], "texts": data["texts"], "embeddings": data["embedding"]}
            else:
                raise KeyError(f"One or more required keys not found in {npz_path}")
        return data["ids"], data["texts"], data["embeddings"]
    except FileNotFoundError:
        print(f"    [ERROR] File not found at: {npz_path}")
        return None, None, None
    except Exception as e:
        print(f"    [ERROR] An error occurred while loading {npz_path}: {e}")
        return None, None, None

def load_npz_data_from_keys(npz_path: str, keys=('ids', 'embeddings')):
    try:
        data = np.load(npz_path, allow_pickle=True)
        return [np.array(data[key]) if key in data else None for key in keys]
    except FileNotFoundError:
        print(f"Error: File not found at {npz_path}")
        return [None] * len(keys)

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-8)

def extract_boolean_prefix(response: str) -> str:
    if not response:
        return ""
    match = re.match(r"^(.*?)(?:\.|\n|$)", response.strip())
    return match.group(1).strip() if match else response.strip()