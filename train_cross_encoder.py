import os
import json
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from tqdm import tqdm
from data_loader import load_corpus, load_train_data, WORK_DIR
from dense_retriever import legal_chunk_document
import string
from rank_bm25 import BM25Okapi
import numpy as np

# Các siêu tham số chuẩn theo nghiên cứu của BTC SoICT / UIT (Bảng 2, Mục 7.2)
BASE_MODEL = "itdainb/PhoRanker"
OUTPUT_DIR = os.path.join(WORK_DIR, "fine_tuned_vietnamese_cross_encoder")
EPOCHS = 2
LR = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 256
RANDOM_SEED = 28

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def get_best_passage(question, full_text, max_chars=600):
    """Trích xuất đoạn điều luật liên quan nhất từ văn bản pháp lý"""
    chunks = legal_chunk_document(full_text, max_chunk_chars=max_chars, overlap=100, max_chunks=100)
    if len(chunks) <= 1:
        return full_text[:max_chars]
    
    tokenized_chunks = [ch.lower().translate(str.maketrans('', '', string.punctuation)).split() for ch in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_q = question.lower().translate(str.maketrans('', '', string.punctuation)).split()
    scores = bm25.get_scores(tokenized_q)
    best_idx = int(np.argmax(scores))
    return chunks[best_idx]

def train_cross_encoder():
    print("=" * 70)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN CROSS-ENCODER '{BASE_MODEL}' CHUẨN BTC UIT")
    print(f"⚙️ Epochs: {EPOCHS} | LR: {LR} | Batch Size: {BATCH_SIZE} | Max Length: {MAX_LENGTH}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Thiết bị tính toán: {device.upper()}")

    # 1. Nạp dữ liệu
    corpus = load_corpus()
    train_data = load_train_data()

    # Nạp Semi-Hard Negatives đã khai thác
    neg_file = os.path.join(WORK_DIR, "semi_hard_negatives.pkl")
    if not os.path.exists(neg_file):
        neg_file = "semi_hard_negatives.pkl"

    semi_hard_negatives = {}
    if os.path.exists(neg_file):
        with open(neg_file, "rb") as f:
            semi_hard_negatives = pickle.load(f)
        print(f"✅ Đã nạp Semi-Hard Negatives cho {len(semi_hard_negatives)} câu hỏi từ '{neg_file}'.")
    else:
        print(f"⚠️ Chưa tìm thấy '{neg_file}'. Hãy chạy 'python mine_semi_hard_negatives.py' trước!")
        return

    # Hỗ trợ tùy chọn tách từ Pyvi nếu có
    has_pyvi = False
    try:
        from pyvi import ViTokenizer
        has_pyvi = True
        print("✅ Đã kích hoạt thư viện tách từ tiếng Việt Pyvi.")
    except ImportError:
        pass

    # 2. Phân chia Train/Validation theo tỷ lệ 90/10 (Mục 5.4 trong bài báo)
    qids = list(train_data.keys())
    random.shuffle(qids)
    split_idx = int(len(qids) * 0.9)
    train_qids = set(qids[:split_idx])
    val_qids = set(qids[split_idx:])
    print(f"📊 Phân chia dữ liệu: {len(train_qids)} câu hỏi Train | {len(val_qids)} câu hỏi Validation")

    train_samples = []
    val_samples = []

    for qid, item in tqdm(train_data.items(), desc="Chuẩn bị mẫu huấn luyện"):
        raw_question = item.get("question", "").strip()
        answers = item.get("answer", [])
        if not raw_question or not answers:
            continue

        question = ViTokenizer.tokenize(raw_question) if has_pyvi else raw_question
        is_train = (qid in train_qids)

        # 2.1. Tách từng đáp án đúng thành các cặp độc lập (Mục 5.2 trong bài báo)
        for ans_id in answers:
            ans_str = str(ans_id)
            if ans_str in corpus:
                doc_text = get_best_passage(raw_question, corpus[ans_str])
                doc_tok = ViTokenizer.tokenize(doc_text) if has_pyvi else doc_text
                example = InputExample(texts=[question, doc_tok], label=1.0)
                if is_train:
                    train_samples.append(example)
                else:
                    val_samples.append(example)

        # 2.2. Ghép cặp với Semi-Hard Negatives (Label = 0.0)
        negs = semi_hard_negatives.get(str(qid), [])
        for neg_id in negs:
            neg_str = str(neg_id)
            if neg_str in corpus:
                doc_text = get_best_passage(raw_question, corpus[neg_str])
                doc_tok = ViTokenizer.tokenize(doc_text) if has_pyvi else doc_text
                example = InputExample(texts=[question, doc_tok], label=0.0)
                if is_train:
                    train_samples.append(example)
                else:
                    val_samples.append(example)

    print(f"✅ Tổng mẫu Train: {len(train_samples)} | Mẫu Validation: {len(val_samples)}")

    # 3. Tạo DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)

    # 4. Khởi tạo mô hình PhoRanker
    print(f"⚡ Đang tải Cross-Encoder '{BASE_MODEL}'...")
    model = CrossEncoder(
        BASE_MODEL,
        num_labels=1,
        max_length=MAX_LENGTH,
        device=device,
        default_activation_function=torch.nn.Identity()
    )

    # Đánh giá trên tập validation
    evaluator = None
    if val_samples:
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_samples, name="UIT-Val")

    warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)

    print(f"🔥 Đang huấn luyện trong {EPOCHS} Epochs với BCEWithLogitsLoss...")
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=EPOCHS,
        loss_fct=nn.BCEWithLogitsLoss(),
        evaluation_steps=len(train_dataloader) // 2 if len(train_dataloader) > 10 else 10,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LR},
        output_path=OUTPUT_DIR,
        use_amp=True if device == "cuda" else False,
        show_progress_bar=True
    )

    print("\n" + "=" * 70)
    print(f"🎉 HUẤN LUYỆN CROSS-ENCODER HOÀN TẤT!")
    print(f"💾 Mô hình đã được lưu tại: '{OUTPUT_DIR}'")
    print("=" * 70)

if __name__ == "__main__":
    train_cross_encoder()
