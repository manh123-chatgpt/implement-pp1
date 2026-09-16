import os
import json
import glob
from tqdm import tqdm

# === TỰ ĐỘNG NHẬN DIỆN MÔI TRƯỜNG ===
if os.path.exists("/kaggle/input"):
    # KAGGLE - Tự dò tìm thư mục data chứa train.json
    WORK_DIR = "/kaggle/working"
    _candidates = [
        "/kaggle/input/datasets/thurdayafternoon/uit-legal-ir-data/uit-legal-ir-data",
        "/kaggle/input/uit-legal-ir-data/uit-legal-ir-data",
        "/kaggle/input/uit-legal-ir-data",
    ]
    DATA_DIR = next((p for p in _candidates if os.path.exists(os.path.join(p, "train.json"))), _candidates[0])
    print(f"📂 Kaggle DATA_DIR: {DATA_DIR}")
else:
    # LOCAL
    _local_candidates = [
        "uit-legal-ir-data",
        "LegalIR - Public Test-20260824T120008Z-1-001/LegalIR - Public Test",
        "."
    ]
    DATA_DIR = next((p for p in _local_candidates if os.path.exists(os.path.join(p, "train.json"))), _local_candidates[0])
    WORK_DIR = "."

CORPUS_DIR = os.path.join(DATA_DIR, "selected-contexts")


def load_corpus(corpus_dir=CORPUS_DIR, force_original=False):
    resolved_path = os.path.join(WORK_DIR, "legal_corpus_resolved.json")
    if not force_original and os.path.exists(resolved_path):
        print(f"♻️ Đang nạp Corpus SIÊU CẤP (Graph Resolved) từ '{resolved_path}'...")
        with open(resolved_path, "r", encoding="utf-8") as f:
            return json.load(f)

    corpus = {}

    json_files = glob.glob(os.path.join(corpus_dir, "**", "*.json"), recursive=True)

    print(f"Đang đọc {len(json_files)} tài liệu...")
    for file_path in tqdm(json_files):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            doc_id = str(data["id"])
            title = data.get("name", "")
            passage = data.get("passage", "")
            
            # Title Boosting 3x: Lặp lại tiêu đề 3 lần để từ khóa số hiệu/tên luật có trọng số vượt trội
            title_boost = f"{title} {title} {title}".strip()
            full_text = f"{title_boost}\n{passage}".strip()
            corpus[doc_id] = full_text

    print(f"Đã nạp thành công {len(corpus)} văn bản vào Corpus (với Title 3x Boosting).")
    return corpus

def load_train_data(train_path=os.path.join(DATA_DIR, "train.json")):
    with open(train_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_test_data(test_path=os.path.join(DATA_DIR, "public-official.json")):
    with open(test_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    corpus = load_corpus()
    train_data = load_train_data()
    test_data = load_test_data()

    print(f"Số lượng câu hỏi Train: {len(train_data)}")
    print(f"Số lượng câu hỏi Test: {len(test_data)}")
    
    sample_id = next(iter(corpus))
    print(f"\nMẫu tài liệu [ID: {sample_id}]:")
    print(corpus[sample_id][:300], "...")