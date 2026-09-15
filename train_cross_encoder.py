import os
import json
import pickle
import random
import string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging as hf_logging
)
hf_logging.set_verbosity_error()
from data_loader import load_corpus, load_train_data, WORK_DIR
from dense_retriever import legal_chunk_document
from rank_bm25 import BM25Okapi

# Các siêu tham số chuẩn theo nghiên cứu của BTC SoICT / UIT
BASE_MODEL = "itdainb/PhoRanker"
OUTPUT_DIR = os.path.join(WORK_DIR, "fine_tuned_vietnamese_cross_encoder")
EPOCHS = 2
LR = 2e-5
BATCH_SIZE = 32
MAX_LENGTH = 256
RANDOM_SEED = 28

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

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

class TextPairDataset(Dataset):
    """Dataset chứa các cặp (câu hỏi, đoạn văn bản, nhãn 0/1)"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def make_collate_fn(tokenizer, max_length=256):
    def collate_fn(batch):
        queries = [item[0] for item in batch]
        passages = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch], dtype=torch.float)

        encoded = tokenizer(
            queries,
            passages,
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt"
        )
        return encoded, labels
    return collate_fn

def train_cross_encoder(epochs=None, lr=None, batch_size=None, output_dir=None):
    """
    Huấn luyện Cross-Encoder PhoRanker.
    Tất cả hyperparams đều configurable để hỗ trợ Grid Search.
    
    Args:
        epochs: Số epochs (mặc định: EPOCHS=2)
        lr: Learning rate (mặc định: LR=2e-5)
        batch_size: Batch size (mặc định: BATCH_SIZE=32)
        output_dir: Thư mục lưu mô hình (mặc định: OUTPUT_DIR)
    """
    _epochs = epochs or EPOCHS
    _lr = lr or LR
    _bs = batch_size or BATCH_SIZE
    _output = output_dir or OUTPUT_DIR
    
    print("=" * 70)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN CROSS-ENCODER '{BASE_MODEL}' CHUẨN BTC UIT")
    print(f"⚙️ Epochs: {_epochs} | LR: {_lr} | Batch Size: {_bs} | Max Length: {MAX_LENGTH}")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Thiết bị tính toán: {str(device).upper()}")

    # 1. Nạp dữ liệu
    corpus = load_corpus()
    train_data = load_train_data()

    # Nạp Semi-Hard Negatives đã khai thác
    _NEG_CANDIDATES = [
        os.path.join(WORK_DIR, "semi_hard_negatives.pkl"),
        "semi_hard_negatives.pkl",
        "/kaggle/input/notebooks/thurdayafternoon/legal-ir/semi_hard_negatives.pkl",
        "/kaggle/input/datasets/thurdayafternoon/pkl-cache/semi_hard_negatives.pkl",
    ]
    neg_file = next((p for p in _NEG_CANDIDATES if os.path.exists(p) and os.path.getsize(p) > 10000), None)

    semi_hard_negatives = {}
    if neg_file:
        try:
            with open(neg_file, "rb") as f:
                semi_hard_negatives = pickle.load(f)
            print(f"✅ Đã nạp Semi-Hard Negatives cho {len(semi_hard_negatives)} câu hỏi từ '{neg_file}'.")
        except Exception as e:
            print(f"⚠️ File Semi-Hard Negatives '{neg_file}' bị lỗi ({e}). Cần khai thác lại!")
            return
    else:
        print("⚠️ Chưa tìm thấy file 'semi_hard_negatives.pkl' hợp lệ. Hãy chạy Bước 1 trước!")
        return

    # Hỗ trợ tùy chọn tách từ Pyvi nếu có
    has_pyvi = False
    try:
        from pyvi import ViTokenizer
        has_pyvi = True
        print("✅ Đã kích hoạt thư viện tách từ tiếng Việt Pyvi.")
    except ImportError:
        pass

    # 2. Phân chia Train/Validation theo tỷ lệ 90/10
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

        # 2.1. Mẫu dương (Positive samples, label = 1.0)
        for ans_id in answers:
            ans_str = str(ans_id)
            if ans_str in corpus:
                doc_text = get_best_passage(raw_question, corpus[ans_str])
                doc_tok = ViTokenizer.tokenize(doc_text) if has_pyvi else doc_text
                sample = (question, doc_tok, 1.0)
                if is_train:
                    train_samples.append(sample)
                else:
                    val_samples.append(sample)

        # 2.2. Mẫu âm bán khó (Semi-Hard Negatives, label = 0.0)
        negs = semi_hard_negatives.get(str(qid), [])
        for neg_id in negs:
            neg_str = str(neg_id)
            if neg_str in corpus:
                doc_text = get_best_passage(raw_question, corpus[neg_str])
                doc_tok = ViTokenizer.tokenize(doc_text) if has_pyvi else doc_text
                sample = (question, doc_tok, 0.0)
                if is_train:
                    train_samples.append(sample)
                else:
                    val_samples.append(sample)

    print(f"✅ Tổng mẫu Train: {len(train_samples)} | Mẫu Validation: {len(val_samples)}")

    # 3. Tải Tokenizer & Model trực tiếp từ Transformers (Loại bỏ triệt để lỗi fit_mixin / DataParallel)
    print(f"⚡ Đang tải mô hình & Tokenizer '{BASE_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.to(device)

    collate_fn = make_collate_fn(tokenizer, max_length=MAX_LENGTH)
    train_loader = DataLoader(
        TextPairDataset(train_samples),
        batch_size=_bs,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if os.name != 'nt' else 0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        TextPairDataset(val_samples),
        batch_size=_bs * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if os.name != 'nt' else 0,
        pin_memory=(device.type == "cuda")
    ) if val_samples else None

    # 4. Optimizer & Scheduler & AMP Scaler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=_lr)
    total_steps = len(train_loader) * _epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    criterion = nn.BCEWithLogitsLoss()
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_loss = float('inf')
    os.makedirs(_output, exist_ok=True)

    print(f"🔥 Bắt đầu huấn luyện {_epochs} Epochs với PyTorch Native & AMP (Mixed Precision)...")

    for epoch in range(_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{_epochs}")
        for batch_encoded, batch_labels in pbar:
            batch_encoded = {k: v.to(device) for k, v in batch_encoded.items()}
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(**batch_encoded)
                logits = outputs.logits.squeeze(-1)
                loss = criterion(logits, batch_labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        print(f"📌 Epoch {epoch + 1} hoàn tất. Train Loss trung bình: {avg_train_loss:.4f}")

        # Đánh giá trên Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_encoded, v_labels in val_loader:
                    v_encoded = {k: v.to(device) for k, v in v_encoded.items()}
                    v_labels = v_labels.to(device)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        out = model(**v_encoded)
                        lgt = out.logits.squeeze(-1)
                        v_l = criterion(lgt, v_labels)
                    val_loss += v_l.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"📊 Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"💾 Lưu checkpoint tốt nhất vào: '{_output}'...")
                model.save_pretrained(_output)
                tokenizer.save_pretrained(_output)

    # Đảm bảo mô hình cuối cùng luôn được lưu
    if not os.path.exists(os.path.join(_output, "config.json")):
        print(f"💾 Lưu mô hình cuối cùng vào: '{_output}'...")
        model.save_pretrained(_output)
        tokenizer.save_pretrained(_output)

    print("\n" + "=" * 70)
    print(f"🎉 HUẤN LUYỆN CROSS-ENCODER HOÀN TẤT THÀNH CÔNG!")
    print(f"💾 Mô hình đã được lưu tại: '{_output}'")
    print("=" * 70)
    
    return best_val_loss  # Trả về val_loss để Grid Search so sánh

if __name__ == "__main__":
    train_cross_encoder()
