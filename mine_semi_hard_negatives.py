import os
import json
import pickle
import random
from tqdm import tqdm
from data_loader import load_corpus, load_train_data, WORK_DIR
from dense_retriever import FineTunedDenseSearcher, DenseSearcher

# Cố định seed theo thực nghiệm BTC (Seed 28 trong Bảng 2 đạt MRR@10 cao nhất 0.7911)
RANDOM_SEED = 28
random.seed(RANDOM_SEED)

def mine_semi_hard_negatives(
    output_path="semi_hard_negatives.pkl",
    top_candidates_k=90,
    num_negatives=10
):
    """
    Thuật toán Semi-Hard Negative Mining chuẩn theo nghiên cứu của BTC SoICT / UIT:
    1. Bi-Encoder dự đoán danh sách Top 90 ứng viên cho mỗi câu hỏi trong train.
    2. Loại bỏ các đáp án đúng (true positives).
    3. Chọn NGẪU NHIÊN n = 10 văn bản từ số còn lại trong Top 90 làm Semi-Hard Negatives.
    (Tránh Hard Negatives quá khó làm nổ gradient và suy giảm mô hình).
    """
    print("=" * 70)
    print("⛏️ KHAI THÁC MẪU ÂM BÁN KHÓ (SEMI-HARD NEGATIVE MINING) CHUẨN BTC UIT")
    print(f"🎯 Số ứng viên Bi-Encoder: Top {top_candidates_k} | Số mẫu âm mỗi câu hỏi: {num_negatives}")
    print("=" * 70)

    corpus = load_corpus()
    train_data = load_train_data()

    print("🚀 Đang khởi tạo Bi-Encoder Retriever...")
    # Tự động nạp fine-tuned model nếu có, hoặc base model
    retriever = FineTunedDenseSearcher(corpus, use_cache=True)

    semi_hard_negatives = {}
    stats = {"total_questions": 0, "total_negatives": 0, "min_negs": 999, "max_negs": 0}

    # Hỗ trợ tùy chọn tách từ Pyvi nếu có
    has_pyvi = False
    try:
        from pyvi import ViTokenizer
        has_pyvi = True
        print("✅ Đã kích hoạt thư viện tách từ tiếng Việt Pyvi.")
    except ImportError:
        print("ℹ️ Pyvi chưa được cài đặt, sử dụng văn bản gốc.")

    for qid, item in tqdm(train_data.items(), desc="Khai thác Semi-Hard Negatives"):
        question = item.get("question", "").strip()
        answers = set(str(a) for a in item.get("answer", []))

        if not question or not answers:
            continue

        query_text = ViTokenizer.tokenize(question) if has_pyvi else question

        # 1. Lấy Top 90 ứng viên từ Bi-Encoder
        candidates = retriever.search(query_text, top_k=top_candidates_k)

        # 2. Loại bỏ các đáp án đúng khỏi danh sách ứng viên
        non_positive_candidates = [str(doc_id) for doc_id in candidates if str(doc_id) not in answers]

        # 3. Lấy ngẫu nhiên n mẫu từ danh sách còn lại (Semi-Hard)
        if len(non_positive_candidates) >= num_negatives:
            sampled_negatives = random.sample(non_positive_candidates, num_negatives)
        else:
            sampled_negatives = non_positive_candidates

        if sampled_negatives:
            semi_hard_negatives[str(qid)] = sampled_negatives
            count = len(sampled_negatives)
            stats["total_questions"] += 1
            stats["total_negatives"] += count
            stats["min_negs"] = min(stats["min_negs"], count)
            stats["max_negs"] = max(stats["max_negs"], count)

    save_file = os.path.join(WORK_DIR, output_path)
    with open(save_file, "wb") as f:
        pickle.dump(semi_hard_negatives, f)

    print("\n" + "=" * 70)
    print(f"🎉 Khai thác thành công Semi-Hard Negatives cho {stats['total_questions']} câu hỏi!")
    print(f"📊 Tổng số mẫu âm bán khó: {stats['total_negatives']} (Trung bình: {stats['total_negatives'] / max(1, stats['total_questions']):.1f}/câu)")
    print(f"💾 File kết quả đã được lưu tại: '{save_file}'")
    print("=" * 70)

if __name__ == "__main__":
    mine_semi_hard_negatives()
