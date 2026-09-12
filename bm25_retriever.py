import os
import re
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
from data_loader import load_corpus, load_train_data
from evaluator import compute_metrics

_BM25_CACHE_CANDIDATES = [
    "bm25_tokenized_cache_pyvi_v2.pkl",
    "/kaggle/working/bm25_tokenized_cache_pyvi_v2.pkl",
    "/kaggle/input/notebooks/thurdayafternoon/legal-ir/bm25_tokenized_cache_pyvi_v2.pkl",
    "/kaggle/input/datasets/thurdayafternoon/pkl-cache/bm25_tokenized_cache_pyvi_v2.pkl",
]
CACHE_FILE = next((p for p in _BM25_CACHE_CANDIDATES if os.path.exists(p)), _BM25_CACHE_CANDIDATES[0])

# Các từ để hỏi phổ biến trong tiếng Việt cần lọc bớt khi truy vấn để tránh làm loãng từ khóa cốt lõi
QUESTION_STOPWORDS = {
    "là", "gì", "như", "thế", "nào", "bao", "nhiêu", "ở", "đâu", "khi", "ai",
    "có", "được", "không", "ra", "sao", "cho", "về", "của", "và", "các", "những",
    "theo", "quy", "định", "pháp", "luật", "thực", "hiện", "này", "đó"
}

def fast_preprocess(text, remove_stopwords=False):
    text = text.lower()
    
    # [QUAN TRỌNG] Tách văn bản thành từng dòng ngắn trước khi đưa vào Pyvi. 
    # Thuật toán CRF của Pyvi sẽ bị treo (freeze) nếu độ dài của 1 chuỗi string quá lớn (VD: một bộ luật dài hàng trăm trang).
    lines = text.split('\n')
    tokenized_lines = [ViTokenizer.tokenize(line) for line in lines if line.strip()]
    text = " ".join(tokenized_lines)
    
    tokens = re.findall(r'[\w_]+', text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in QUESTION_STOPWORDS]
    return tokens

class BM25Searcher:
    def __init__(self, corpus, use_cache=True):
        self.doc_ids = list(corpus.keys())
        
        if use_cache and os.path.exists(CACHE_FILE):
            print(f"⚡ Đang nạp chỉ mục BM25 v2 từ cache '{CACHE_FILE}'...")
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
                self.doc_ids = data["doc_ids"]
                self.tokenized_corpus = data["tokens"]
        else:
            print("⚡ Đang tiền xử lý nâng cao với Title Boosting cho 8.532 văn bản...")
            self.tokenized_corpus = []
            for doc_id in tqdm(self.doc_ids, desc="Indexing BM25"):
                full_text = corpus[doc_id]
                tokens = fast_preprocess(full_text)
                self.tokenized_corpus.append(tokens)
            
            with open(CACHE_FILE, "wb") as f:
                pickle.dump({"doc_ids": self.doc_ids, "tokens": self.tokenized_corpus}, f)
            print(f"✅ Đã lưu cache BM25 vào '{CACHE_FILE}'.")
            
        print("Đang khởi tạo thuật toán BM25 (k1=1.2, b=0.3 - Tối ưu cho Legal Long-Text)...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=1.2, b=0.3)
        print("✅ Đã sẵn sàng tìm kiếm BM25!")

    def search(self, query, top_k=5):
        # Lọc bớt từ hỏi để tập trung vào từ khóa quan trọng
        query_tokens = fast_preprocess(query, remove_stopwords=True)
        if not query_tokens:
            query_tokens = fast_preprocess(query, remove_stopwords=False)
            
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.doc_ids[idx] for idx in top_indices]

if __name__ == "__main__":
    # 1. Nạp dữ liệu
    corpus = load_corpus()
    train_data = load_train_data()
    
    # 2. Lấy thử nghiệm 500 câu đầu từ train.json để đo điểm
    val_items = list(train_data.items())[:500]
    val_questions = {k: v["question"] for k, v in val_items}
    val_truth = {k: v["answer"] for k, v in val_items}
    
    # 3. Khởi tạo BM25
    searcher = BM25Searcher(corpus, use_cache=True)
    
    # 4. Chạy tìm kiếm trên 500 câu
    print("\n🔍 Đang chạy tìm kiếm BM25 trên 500 câu hỏi...")
    predictions = {}
    for qid, question in tqdm(val_questions.items(), desc="Searching"):
        top_docs = searcher.search(question, top_k=5)
        predictions[qid] = top_docs
        
    # 5. Chấm điểm
    results = compute_metrics(predictions, val_truth, k=5)
    print("\n" + "="*45)
    print(f"📊 KẾT QUẢ BASELINE BM25 (Top 5):")
    print(f"👉 Recall@5   : {results['Recall'] * 100:.2f}%")
    print(f"👉 Precision@5: {results['Precision'] * 100:.2f}%")
    print("="*45)
