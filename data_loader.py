import os
import json
import glob
import re
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


def clean_legal_boilerplate(text):
    """
    Loại bỏ nhiễu hành chính thuần túy không mang giá trị ngữ nghĩa:
    1. Quốc hiệu & Tiêu ngữ: 'CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM ... Độc lập - Tự do - Hạnh phúc'
    2. Khối 'Nơi nhận: ...' ở cuối văn bản (bảo toàn 'Phụ lục' nếu có)
    """
    if not text:
        return ""
    # 1. Xóa Quốc hiệu & Tiêu ngữ
    text = re.sub(
        r'cộng\s*hòa\s*xã\s*hội\s*chủ\s*nghĩa\s*việt\s*nam[\s\r\n\-–—_]*độc\s*lập\s*[\-–—]\s*tự\s*do\s*[\-–—]\s*hạnh\s*phúc[\s\r\n\-–—_]*',
        '',
        text,
        flags=re.IGNORECASE
    )
    # 2. Xóa phần Nơi nhận ở đuôi văn bản (chỉ dừng trước Phụ lục nếu có)
    text = re.sub(
        r'\bnơi\s*nhận\s*:[\s\S]*?(?=(?:\n\s*phụ\s*lục\b|\Z))',
        '',
        text,
        flags=re.IGNORECASE
    )
    # 3. Chuẩn hóa khoảng trống thừa
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()


def load_corpus(corpus_dir=CORPUS_DIR, force_original=False, clean_noise=True):
    resolved_clean = os.path.join(WORK_DIR, "legal_corpus_resolved_clean.json")
    resolved_path = os.path.join(WORK_DIR, "legal_corpus_resolved.json")

    if clean_noise and os.path.exists(resolved_clean):
        print(f"♻️ Đang nạp Corpus SIÊU CẤP ĐÃ LÀM SẠCH NHIỄU từ '{resolved_clean}'...")
        with open(resolved_clean, "r", encoding="utf-8") as f:
            return json.load(f)

    if not force_original and os.path.exists(resolved_path):
        print(f"♻️ Đang nạp Corpus SIÊU CẤP (Graph Resolved) từ '{resolved_path}'...")
        with open(resolved_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        if clean_noise:
            print("🧹 Đang tự động loại bỏ Quốc hiệu & 'Nơi nhận' trên toàn bộ Corpus...")
            corpus = {k: clean_legal_boilerplate(v) for k, v in corpus.items()}
            try:
                with open(resolved_clean, "w", encoding="utf-8") as f:
                    json.dump(corpus, f, ensure_ascii=False, indent=1)
                print(f"💾 Đã lưu cache sạch vào '{resolved_clean}'.")
            except Exception:
                pass
        return corpus

    corpus = {}

    json_files = glob.glob(os.path.join(corpus_dir, "**", "*.json"), recursive=True)

    print(f"Đang đọc {len(json_files)} tài liệu...")
    for file_path in tqdm(json_files):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            doc_id = str(data["id"])
            title = data.get("name", "")
            passage = data.get("passage", "")
            if clean_noise:
                passage = clean_legal_boilerplate(passage)
            
            # Title Boosting 3x: Lặp lại tiêu đề 3 lần để từ khóa số hiệu/tên luật có trọng số vượt trội
            title_boost = f"{title} {title} {title}".strip()
            full_text = f"{title_boost}\n{passage}".strip()
            corpus[doc_id] = full_text

    print(f"Đã nạp thành công {len(corpus)} văn bản vào Corpus (với Title 3x Boosting & Cleaned).")
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