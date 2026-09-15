"""
⚡ FAST EVALUATION SCRIPT: Thử nghiệm & Tinh chỉnh siêu tốc trên tập con Validation.
Thay vì đợi hàng tiếng đồng hồ trên 8.533 văn bản, script này:
1. Lấy ngẫu nhiên N câu hỏi đại diện (mặc định: 50 câu) từ tập train/val.
2. Đánh giá tốc độ và độ chính xác (Recall@5, Precision@5) chỉ trong 15-30 giây.
3. Cho phép so sánh nhanh giữa các mô hình: BM25, BKAI, Jina-v3, Qwen-Legal, PhoRanker.
"""

import os
import sys
import time
import random
import argparse
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

from data_loader import load_corpus, load_train_data
from evaluator import compute_metrics

def run_fast_evaluation(
    num_samples=50,
    seed=42,
    mode="all",  # "baseline", "jina", "qwen", "ensemble", "reranker", hoặc "all"
    st1_weight=0.75,
    rerank_top_k=30
):
    print("=" * 75)
    print(f"⚡ FAST EVALUATION: TINH CHỈNH SIÊU TỐC ({num_samples} CÂU HỎI THẨM ĐỊNH)")
    print(f"🎯 Seed: {seed} | Mode: {mode.upper()} | St1 Weight: {st1_weight}")
    print("=" * 75)

    from hybrid_retriever import HybridSearcher

    corpus = load_corpus()
    train_data = load_train_data()

    # Lọc các câu hỏi có đáp án hợp lệ
    valid_items = {
        qid: item for qid, item in train_data.items()
        if item.get("question", "").strip() and item.get("answer", [])
    }

    # Chọn ngẫu nhiên N câu hỏi cố định theo seed
    random.seed(seed)
    selected_qids = random.sample(list(valid_items.keys()), min(num_samples, len(valid_items)))

    val_questions = {qid: valid_items[qid]["question"].strip() for qid in selected_qids}
    val_truth = {qid: [str(a) for a in valid_items[qid]["answer"]] for qid in selected_qids}

    print(f"📊 Đã chọn {len(val_questions)} câu hỏi đại diện (Corpus: {len(corpus)} văn bản)")

    # Định nghĩa các cấu hình cần thử nghiệm
    configs = []
    if mode in ["baseline", "all"]:
        configs.append({
            "name": "1. Baseline (BM25 + BKAI Bi-Encoder)",
            "params": {"light_mode": True, "use_reranker": False, "use_jina": False, "use_qwen": False}
        })
    if mode in ["jina", "all"]:
        configs.append({
            "name": "2. SOTA Jina-v3 (8k Context + BM25)",
            "params": {"light_mode": True, "use_reranker": False, "use_jina": True, "use_qwen": False}
        })
    if mode in ["qwen", "all"]:
        configs.append({
            "name": "3. Qwen3 Legal (Chuyên gia Luật VN + BM25)",
            "params": {"light_mode": True, "use_reranker": False, "use_jina": False, "use_qwen": True}
        })
    if mode in ["ensemble", "all"]:
        configs.append({
            "name": "4. Siêu Đội Hình (BM25 + Jina-v3 + Qwen-Legal)",
            "params": {"light_mode": True, "use_reranker": False, "use_jina": True, "use_qwen": True}
        })
    if mode in ["reranker", "all"]:
        configs.append({
            "name": "5. PhoRanker Re-ranking (3 Fix Cốt Lõi)",
            "params": {"light_mode": True, "use_reranker": True, "use_jina": False, "use_qwen": False}
        })

    summary_results = []

    for cfg in configs:
        print("\n" + "-" * 75)
        print(f"🚀 Đang chạy: {cfg['name']}...")
        try:
            searcher = HybridSearcher(corpus, **cfg["params"])
            
            start_time = time.time()
            preds = {}
            for qid, q_text in tqdm(val_questions.items(), desc=f"Eval {cfg['name'][:25]}"):
                preds[qid] = searcher.search(
                    q_text,
                    top_k=5,
                    rerank_top_k=rerank_top_k,
                    st1_weight=st1_weight
                )
            elapsed = time.time() - start_time
            ms_per_query = (elapsed / len(val_questions)) * 1000

            metrics = compute_metrics(preds, val_truth, k=5)
            recall = metrics["Recall"] * 100
            precision = metrics["Precision"] * 100

            summary_results.append({
                "Tên cấu hình": cfg["name"],
                "Recall@5": f"{recall:.2f}%",
                "Precision@5": f"{precision:.2f}%",
                "Thời gian": f"{elapsed:.1f}s",
                "Tốc độ": f"{ms_per_query:.0f} ms/câu"
            })

            print(f"👉 Kết quả: Recall@5: {recall:.2f}% | Precision@5: {precision:.2f}% | Tốc độ: {ms_per_query:.0f} ms/câu")

        except Exception as e:
            print(f"⚠️ Lỗi khi chạy cấu hình {cfg['name']}: {e}")
            summary_results.append({
                "Tên cấu hình": cfg["name"],
                "Recall@5": "Lỗi",
                "Precision@5": "Lỗi",
                "Thời gian": "-",
                "Tốc độ": "-"
            })

    # In bảng tổng kết so sánh
    print("\n" + "=" * 80)
    print("📊 BẢNG TỔNG HỢP SO SÁNH HIỆU QUẢ CÁC MÔ HÌNH")
    print("=" * 80)
    header = f"{'Cấu hình mô hình':<42} | {'Recall@5':<10} | {'Precision@5':<12} | {'Tốc độ':<12}"
    print(header)
    print("-" * len(header))
    for res in summary_results:
        print(f"{res['Tên cấu hình']:<42} | {res['Recall@5']:<10} | {res['Precision@5']:<12} | {res['Tốc độ']:<12}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Evaluation Script for UIT Legal IR")
    parser.add_argument("--samples", type=int, default=50, help="Số câu hỏi thẩm định nhanh (default: 50)")
    parser.add_argument("--mode", type=str, default="all", choices=["baseline", "jina", "qwen", "ensemble", "reranker", "all"], help="Chế độ kiểm tra")
    parser.add_argument("--weight", type=float, default=0.75, help="Trọng số Stage 1 trong score fusion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_fast_evaluation(
        num_samples=args.samples,
        seed=args.seed,
        mode=args.mode,
        st1_weight=args.weight
    )
