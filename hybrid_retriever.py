import os
import json
import zipfile
import torch
import numpy as np
from tqdm import tqdm
from data_loader import load_corpus, load_train_data, load_test_data, WORK_DIR
from bm25_retriever import BM25Searcher
from dense_retriever import DenseSearcher
from evaluator import compute_metrics
import string
from rank_bm25 import BM25Okapi
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import re

# Từ điển ánh xạ từ viết tắt và thuật ngữ pháp lý Việt Nam phổ biến
LEGAL_ACRONYMS = {
    r"\bbhxh\b": "bhxh bảo hiểm xã hội",
    r"\bbhyt\b": "bhyt bảo hiểm y tế",
    r"\bbhtn\b": "bhtn bảo hiểm thất nghiệp",
    r"\btand\b": "tand tòa án nhân dân",
    r"\bvksnd\b": "vksnd viện kiểm sát nhân dân",
    r"\bubnd\b": "ubnd ủy ban nhân dân",
    r"\bhđnd\b": "hđnd hội đồng nhân dân",
    r"\bhdnd\b": "hđnd hội đồng nhân dân",
    r"\bdnnvv\b": "dnnvv doanh nghiệp nhỏ và vừa",
    r"\bdn\b": "doanh nghiệp",
    r"\bnlđ\b": "nlđ người lao động",
    r"\bnld\b": "nlđ người lao động",
    r"\bnsnn\b": "nsnn ngân sách nhà nước",
    r"\bcsgt\b": "csgt cảnh sát giao thông",
    r"\bpccc\b": "pccc phòng cháy chữa cháy",
    r"\batlđ\b": "atlđ an toàn lao động",
    r"\batld\b": "atlđ an toàn lao động",
    r"\bttđb\b": "ttđb tiêu thụ đặc biệt",
    r"\bttdb\b": "ttđb tiêu thụ đặc biệt",
    r"\btndn\b": "tndn thu nhập doanh nghiệp",
    r"\btncn\b": "tncn thu nhập cá nhân",
    r"\bgtgt\b": "gtgt giá trị gia tăng",
    r"\bvat\b": "vat giá trị gia tăng",
    r"\bđkkd\b": "đkkd đăng ký kinh doanh",
    r"\bdkth\b": "dkth điều kiện thực hiện",
    r"\bqđ\b": "quyết định",
    r"\bnđ\b": "nghị định",
    r"\btt\b": "thông tư",
    r"\bhđlđ\b": "hợp đồng lao động",
    r"\bhdld\b": "hợp đồng lao động",
    
    # --- TỪ ĐIỂN ĐỒNG NGHĨA DÂN GIAN (Colloquial Synonyms) ---
    r"\bxe máy\b": "xe máy xe mô tô xe gắn máy",
    r"\bxe ôtô\b": "xe ôtô xe ô tô xe hơi",
    r"\bô tô\b": "ô tô xe ô tô xe hơi",
    r"\bsổ đỏ\b": "sổ đỏ giấy chứng nhận quyền sử dụng đất",
    r"\bsổ hồng\b": "sổ hồng giấy chứng nhận quyền sở hữu nhà ở",
    r"\bcmnd\b": "cmnd chứng minh nhân dân thẻ căn cước công dân",
    r"\bcccd\b": "cccd thẻ căn cước công dân chứng minh nhân dân",
    r"\bcăn cước\b": "căn cước thẻ căn cước công dân chứng minh nhân dân",
    r"\bnghỉ đẻ\b": "nghỉ đẻ thai sản sinh con",
    r"\bđuổi việc\b": "đuổi việc sa thải chấm dứt hợp đồng lao động",
    r"\bđền bù\b": "đền bù bồi thường thiệt hại",
    r"\bphạt nguội\b": "phạt nguội xử phạt vi phạm hành chính qua camera",
    r"\bbằng lái xe\b": "bằng lái xe giấy phép lái xe",
    r"\bgiấy phép lái xe\b": "giấy phép lái xe bằng lái xe",
    r"\bly dị\b": "ly dị ly hôn",
    r"\bcông an\b": "công an cảnh sát cơ quan điều tra",
    r"\bđất đai\b": "đất đai quyền sử dụng đất",
    r"\blàm luật\b": "làm luật hối lộ đưa hối lộ nhận hối lộ"
}

def expand_legal_query(query):
    """
    Tự động mở rộng câu hỏi pháp lý với các thuật ngữ viết tắt và đồng nghĩa
    """
    expanded = query.lower()
    for pattern, replacement in LEGAL_ACRONYMS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded

def extract_law_numbers(text):
    """
    Trích xuất số hiệu văn bản pháp luật, ví dụ:
    - 136/2020/nđ-cp
    - 58/2016/tt-btc
    - 45/2019/qh14
    - 5868/qđ-byt
    """
    text = text.lower()
    patterns = [
        r'\b\d+/\d+/[a-zđ\-_]+',  # Ví dụ: 136/2020/nđ-cp, 58/2016/tt-btc
        r'\b\d+/[a-zđ\-_]+'       # Ví dụ: 5868/qđ-byt
    ]
    matches = []
    for p in patterns:
        found = re.findall(p, text)
        matches.extend(found)
    return list(set(matches))

def extract_article_number(text):
    """
    Trích xuất "Điều X" (Article X) từ câu hỏi.
    Ví dụ: "Theo Điều 15..." -> "điều 15"
    """
    text = text.lower()
    matches = re.findall(r'\bđiều\s+\d+\b', text)
    return list(set(matches))

from dense_retriever import (
    DenseSearcher, FineTunedDenseSearcher, E5LargeSearcher, legal_chunk_document,
    JinaEmbeddingSearcher, QwenLegalEmbeddingSearcher, Qwen4BEmbeddingSearcher
)
from sentence_transformers import CrossEncoder

def get_best_chunk_bm25(query, text, max_chunk_chars=800):
    """Tìm Chunk tốt nhất trong tài liệu chứa câu trả lời cho câu hỏi bằng BM25"""
    chunks = legal_chunk_document(text, max_chunk_chars=max_chunk_chars, overlap=100, max_chunks=150)
    if len(chunks) == 1:
        return chunks[0]
    
    tokenized_chunks = [ch.lower().translate(str.maketrans('', '', string.punctuation)).split() for ch in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    tokenized_query = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    scores = bm25.get_scores(tokenized_query)
    
    best_idx = np.argmax(scores)
    return chunks[best_idx]

class CrossEncoderReranker:
    def __init__(self, model_path=None, corpus=None):
        self.corpus = corpus
        self.is_active = False
        self.use_hf = False

        # Đồng bộ Pyvi tokenization giữa train và inference
        self.has_pyvi = False
        try:
            from pyvi import ViTokenizer
            self.has_pyvi = True
            self._tokenize = ViTokenizer.tokenize
            print("✅ CrossEncoder Reranker: Đã kích hoạt Pyvi tách từ (đồng bộ với training).")
        except ImportError:
            self._tokenize = lambda x: x
            print("ℹ️ CrossEncoder Reranker: Pyvi không khả dụng, sử dụng văn bản gốc.")

        # Tự động dò đường dẫn model đã fine-tune, hoặc tải PhoRanker gốc / Qwen3-Reranker-4B
        candidates = [
            model_path,
            os.path.join(WORK_DIR, "fine_tuned_vietnamese_cross_encoder"),
            "fine_tuned_vietnamese_cross_encoder",
            "/kaggle/input/notebooks/thurdayafternoon/legal-ir/fine_tuned_vietnamese_cross_encoder",
            "/kaggle/input/datasets/thurdayafternoon/pkl-cache/fine_tuned_vietnamese_cross_encoder",
            "/kaggle/input/fine-tuned-vietnamese-cross-encoder/fine_tuned_vietnamese_cross_encoder",
            "itdainb/PhoRanker"
        ]
        chosen_path = None
        for p in candidates:
            if p and (os.path.exists(p) or "/" in p or p.startswith("Qwen") or p.startswith("bge") or p.startswith("infgrad")):
                chosen_path = p
                break

        if not chosen_path:
            chosen_path = "itdainb/PhoRanker"

        self.chosen_path = chosen_path
        self.is_finetuned = (chosen_path != "itdainb/PhoRanker" and not any(k in chosen_path for k in ["Qwen", "Prism", "bge", "jina", "gemma"]))
        self.is_llm_reranker = any(k in chosen_path.lower() for k in ["qwen", "prism", "bge", "jina", "gemma"])
        self.is_4b = "4b" in chosen_path.lower()
        
        # Thiết lập độ dài ngữ cảnh phù hợp: PhoRanker dùng 256, LLM Reranker dùng 1536
        self.max_length = 1536 if self.is_llm_reranker else 256
        self.max_passage_chars = 2500 if self.is_llm_reranker else 600
        
        print(f"🚀 KHỞI ĐỘNG RERANKER: '{chosen_path}' (LLM Mode: {self.is_llm_reranker} | 4B Model: {self.is_4b} | Context: {self.max_length})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = CrossEncoder(
                chosen_path,
                device=device,
                num_labels=1,
                max_length=self.max_length,
                trust_remote_code=True,
                automodel_args={"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
            )
            self.is_active = True
            print(f"✅ Re-ranker sẵn sàng trên thiết bị: {device.upper()} (FP16 Mode)")
        except Exception as e:
            print(f"⚠️ Thử nạp Cross-Encoder qua Transformers native: {e}")
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(chosen_path, trust_remote_code=True)
                self.hf_model = AutoModelForSequenceClassification.from_pretrained(
                    chosen_path,
                    num_labels=1,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                self.hf_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.hf_model.to(self.hf_device).eval()
                self.is_active = True
                self.use_hf = True
                print(f"✅ Re-ranker sẵn sàng với Transformers native trên thiết bị: {self.hf_device}")
            except Exception as e2:
                print(f"⚠️ Không thể nạp Cross-Encoder: {e2}. Hệ thống sẽ sử dụng điểm lọc Stage 1.")
                self.is_active = False

    def score_candidates(self, query, top_docs):
        """Tính điểm thô từ Cross-Encoder cho danh sách ứng viên (chỉ chạy 1 lần duy nhất)"""
        if not self.is_active or not self.corpus or not top_docs:
            return np.zeros(len(top_docs))

        query_text = (self._tokenize(query) if (self.has_pyvi and not self.is_llm_reranker) else query)
        pairs = []
        for doc_id in top_docs:
            doc_text = self.corpus.get(str(doc_id), "")
            passage = get_best_chunk_bm25(query, doc_text, max_chunk_chars=self.max_passage_chars)
            passage_text = (self._tokenize(passage) if (self.has_pyvi and not self.is_llm_reranker) else passage)
            pairs.append([query_text, passage_text])

        # ⚡ Tối ưu VRAM: Đối với mô hình 4B, dùng batch_size=2 để tránh OOM trên Kaggle T4 (15GB VRAM)
        batch_sz = (2 if self.is_4b else 16) if torch.cuda.is_available() else 4

        try:
            if self.use_hf:
                scores = []
                with torch.inference_mode():
                    for i in range(0, len(pairs), batch_sz):
                        b_pairs = pairs[i:i + batch_sz]
                        enc = self.tokenizer([p[0] for p in b_pairs], [p[1] for p in b_pairs],
                                             padding=True, truncation="longest_first", max_length=self.max_length, return_tensors="pt")
                        enc = {k: v.to(self.hf_device) for k, v in enc.items()}
                        out = self.hf_model(**enc)
                        lgt = out.logits.squeeze(-1).cpu().numpy()
                        if lgt.ndim == 0:
                            scores.append(float(lgt))
                        else:
                            scores.extend(lgt.tolist())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return np.array(scores)
            else:
                with torch.inference_mode():
                    scores = self.model.predict(pairs, batch_size=batch_sz, show_progress_bar=False)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return scores
        except Exception as e:
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                print(f"⚠️ CUDA OOM khi chấm điểm! Đang dọn VRAM và thử lại an toàn với batch_size=1...")
                torch.cuda.empty_cache()
                try:
                    with torch.inference_mode():
                        scores = self.model.predict(pairs, batch_size=1, show_progress_bar=False)
                    torch.cuda.empty_cache()
                    return scores
                except Exception as e2:
                    print(f"⚠️ Vẫn lỗi sau retry batch_size=1: {e2}")
            else:
                print(f"⚠️ Lỗi score_candidates: {e}")
            return np.zeros(len(top_docs))

    def fuse_and_rank(self, top_docs, rr_scores, candidate_scores=None, mega_boost_docs=None, exact_matched_docs=None, top_k=5, st1_weight=None, rerank_top_k=30):
        """Kết hợp điểm (Score Fusion) cực nhanh trong RAM bằng đại số ma trận"""
        if len(top_docs) == 0:
            return []
        
        # Chỉ re-rank trong phạm vi rerank_top_k
        eval_docs = top_docs[:rerank_top_k]
        eval_rr_scores = rr_scores[:rerank_top_k] if len(rr_scores) >= len(eval_docs) else np.zeros(len(eval_docs))
        
        mega_boost_docs = mega_boost_docs or set()
        exact_matched_docs = exact_matched_docs or set()

        if candidate_scores:
            c_vals = [candidate_scores.get(d, 0.0) for d in eval_docs]
            max_st1 = max(c_vals) if c_vals else 1.0
            min_st1 = min(c_vals) if c_vals else 0.0
            range_st1 = max(max_st1 - min_st1, 1e-6)
        else:
            range_st1 = 1.0
            min_st1 = 0.0

        max_rr = float(np.max(eval_rr_scores)) if len(eval_rr_scores) > 0 else 1.0
        min_rr = float(np.min(eval_rr_scores)) if len(eval_rr_scores) > 0 else 0.0
        range_rr = max(max_rr - min_rr, 1e-6)

        w_st1 = st1_weight if st1_weight is not None else (0.75 if self.is_finetuned else 0.85)

        final_scores = {}
        for i, doc_id in enumerate(eval_docs):
            rr_norm = (eval_rr_scores[i] - min_rr) / range_rr
            if candidate_scores and doc_id in candidate_scores:
                st1_norm = (candidate_scores[doc_id] - min_st1) / range_st1
            else:
                st1_norm = (len(eval_docs) - i) / len(eval_docs)

            fused = w_st1 * st1_norm + (1.0 - w_st1) * rr_norm

            if doc_id in mega_boost_docs:
                fused += 100.0
            elif doc_id in exact_matched_docs:
                fused += 10.0

            final_scores[doc_id] = fused

        # Thêm các ứng viên ngoài rerank_top_k với điểm thấp hơn
        for i, doc_id in enumerate(top_docs[rerank_top_k:]):
            final_scores[doc_id] = -100.0 - i

        ranked_docs = sorted(top_docs, key=lambda d: final_scores.get(d, -999.0), reverse=True)
        return ranked_docs[:top_k]

    def rerank(self, query, top_docs, candidate_scores=None, mega_boost_docs=None, exact_matched_docs=None, top_k=5, st1_weight=None, rerank_top_k=30):
        if not self.is_active or not self.corpus or not top_docs:
            return top_docs[:top_k]

        eval_docs = top_docs[:rerank_top_k]
        rr_scores = self.score_candidates(query, eval_docs)
        return self.fuse_and_rank(top_docs, rr_scores, candidate_scores, mega_boost_docs, exact_matched_docs, top_k, st1_weight, rerank_top_k)

class HybridSearcher:
    def __init__(self, corpus, use_reranker=False, light_mode=True, reranker_model_path=None,
                 bm25_type="okapi", bm25_k1=1.2, bm25_b=0.3,
                 use_jina=False, use_qwen=False, use_qwen4b=False):
        """
        use_reranker=False: Hệ thống Stage 1 (BM25 + Bi-Encoder + Rules).
        use_reranker=True: Kích hoạt Cross-Encoder (PhoRanker hoặc Qwen3-Reranker-4B).
        bm25_type: "okapi" hoặc "plus" (BM25Plus).
        bm25_k1/bm25_b: Tham số BM25 cho Grid Search.
        use_jina=True: Kích hoạt mô hình SOTA jinaai/jina-embeddings-v3 (8.192 context).
        use_qwen=True: Kích hoạt mô hình Qwen3 Vietnamese Legal Embedding.
        use_qwen4b=True: Kích hoạt siêu mô hình 4B Qwen/Qwen3-Embedding-4B.
        """
        print(f"🚀 Khởi tạo HỆ THỐNG RETRIEVAL CHUẨN BTC UIT (Light Mode: {light_mode} | Reranker: {use_reranker} | Jina-v3: {use_jina} | Qwen-Legal: {use_qwen} | Qwen3-4B: {use_qwen4b})...")
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.light_mode = light_mode
        self.use_jina = use_jina
        self.use_qwen = use_qwen
        self.use_qwen4b = use_qwen4b
        
        # Bản đồ số hiệu văn bản tra cứu O(1)
        self.doc_law_numbers = {}
        self.corpus_lower = {}
        for doc_id in self.doc_ids:
            self.corpus_lower[doc_id] = corpus[doc_id].lower()
            title_line = corpus[doc_id].split("\n")[0]
            laws = extract_law_numbers(title_line)
            if laws:
                self.doc_law_numbers[doc_id] = laws
        
        # Giai đoạn 1: Bộ thu thập ứng viên (Candidate Retrieval)
        self.bm25_searcher = BM25Searcher(corpus, use_cache=True, bm25_type=bm25_type, k1=bm25_k1, b=bm25_b)
        self.finetuned_searcher = FineTunedDenseSearcher(corpus, use_cache=True)
        
        if not light_mode:
            self.dense_searcher = DenseSearcher(corpus, use_cache=True)
            self.e5_searcher = E5LargeSearcher(corpus, use_cache=True)
        else:
            self.dense_searcher = None
            self.e5_searcher = None

        # Tích hợp Jina-v3, Qwen-Legal, Qwen3-4B (nếu được kích hoạt)
        self.jina_searcher = JinaEmbeddingSearcher(corpus, use_cache=True) if use_jina else None
        self.qwen_searcher = QwenLegalEmbeddingSearcher(corpus, use_cache=True) if use_qwen else None
        self.qwen4b_searcher = Qwen4BEmbeddingSearcher(corpus, use_cache=True) if use_qwen4b else None

        # Giai đoạn 2: Bộ tái xếp hạng (Cross-Encoder Re-ranker)
        if use_reranker:
            self.reranker = CrossEncoderReranker(model_path=reranker_model_path, corpus=corpus)
        else:
            self.reranker = None

    def search(self, query, top_k=5, candidate_k=90, rerank_top_k=30, rrf_k=10, st1_weight=None):
        """
        Quy trình 2 giai đoạn chuẩn BTC:
        - Giai đoạn 1: BM25 + Bi-Encoder + Luật boost lấy Top 90 ứng viên.
        - Giai đoạn 2: Cross-Encoder PhoRanker re-rank Top 30 tinh hoa nhất.
        
        Args:
            candidate_k: Số ứng viên Stage 1 lấy từ mỗi mô hình (90).
            rerank_top_k: Số ứng viên tinh hoa đưa vào Re-ranker (30).
            st1_weight: Trọng số Stage 1 trong fusion (0.0-1.0). None = dùng mặc định.
        """
        query_laws = extract_law_numbers(query)
        query_articles = extract_article_number(query)
        
        exact_matched_docs = set()
        mega_boost_docs = set()
        
        for doc_id, doc_text_lower in self.corpus_lower.items():
            has_law = False
            has_article = False
            
            if query_laws and doc_id in self.doc_law_numbers:
                for ql in query_laws:
                    if any(ql in dl or dl in ql for dl in self.doc_law_numbers[doc_id]):
                        has_law = True
                        break
            
            if has_law and query_articles:
                for qa in query_articles:
                    if qa in doc_text_lower:
                        has_article = True
                        break
                
            if has_law and has_article:
                mega_boost_docs.add(doc_id)
            elif has_law:
                exact_matched_docs.add(doc_id)

        expanded_query = expand_legal_query(query)

        # 1. Lọc ứng viên từ BM25 và Bi-Encoder
        bm25_top = self.bm25_searcher.search(expanded_query, top_k=candidate_k)
        finetuned_top = self.finetuned_searcher.search(expanded_query, top_k=candidate_k)

        scores = {}
        for rank, doc_id in enumerate(bm25_top):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 1.5

        for rank, doc_id in enumerate(finetuned_top):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 2.5

        # Nếu kích hoạt Jina-v3 (8k context), kết hợp với trọng số cao
        if self.jina_searcher:
            jina_top = self.jina_searcher.search(query, top_k=candidate_k)
            for rank, doc_id in enumerate(jina_top):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 3.0

        # Nếu kích hoạt Qwen Legal, kết hợp với trọng số cao
        if self.qwen_searcher:
            qwen_top = self.qwen_searcher.search(query, top_k=candidate_k)
            for rank, doc_id in enumerate(qwen_top):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 3.0

        # Nếu kích hoạt Qwen3-4B (Tier S+), kết hợp với trọng số cao nhất (x3.5)
        if self.qwen4b_searcher:
            qwen4b_top = self.qwen4b_searcher.search(query, top_k=candidate_k)
            for rank, doc_id in enumerate(qwen4b_top):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 3.5

        # Nếu không ở chế độ light mode, kết hợp thêm BGE-M3 và E5
        if not self.light_mode and self.dense_searcher and self.e5_searcher:
            dense_top = self.dense_searcher.search(expanded_query, top_k=candidate_k)
            e5_top = self.e5_searcher.search(expanded_query, top_k=candidate_k)
            for rank, doc_id in enumerate(dense_top):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 0.5
            for rank, doc_id in enumerate(e5_top):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 1.0

        for doc_id in exact_matched_docs:
            scores[doc_id] = scores.get(doc_id, 0.0) + 5.0
            
        for doc_id in mega_boost_docs:
            scores[doc_id] = scores.get(doc_id, 0.0) + 7.0

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [doc_id for doc_id, _ in sorted_docs[:candidate_k]]

        # 2. Giai đoạn Re-ranking với PhoRanker (kết hợp Score Fusion)
        # Chỉ đưa Top rerank_top_k (30) ứng viên tinh hoa vào Re-ranker
        # để tránh nhiễu từ các ứng viên hạng thấp (50-90)
        if self.reranker and self.reranker.is_active:
            rerank_candidates = top_candidates[:rerank_top_k]
            return self.reranker.rerank(
                query,
                rerank_candidates,
                candidate_scores=scores,
                mega_boost_docs=mega_boost_docs,
                exact_matched_docs=exact_matched_docs,
                top_k=top_k,
                st1_weight=st1_weight,
                rerank_top_k=rerank_top_k
            )
        
        return top_candidates[:top_k]

    def get_query_candidates_and_rerank_scores(self, query, candidate_k=90, max_rerank_k=50):
        """
        Dùng cho Grid Search siêu tốc:
        Thu thập ứng viên Stage 1 và chấm điểm Reranker đúng 1 LẦN DUY NHẤT.
        Trả về dictionary dữ liệu sẵn sàng cho in-memory vector fusion.
        """
        query_laws = extract_law_numbers(query)
        query_articles = extract_article_number(query)
        
        exact_matched_docs = set()
        mega_boost_docs = set()
        
        for doc_id, doc_text_lower in self.corpus_lower.items():
            laws_in_doc = self.doc_law_numbers.get(doc_id, [])
            has_law = any(ql in laws_in_doc for ql in query_laws)
            has_article = False
            if has_law and query_articles:
                for qa in query_articles:
                    if qa in doc_text_lower:
                        has_article = True
                        break
                
            if has_law and has_article:
                mega_boost_docs.add(doc_id)
            elif has_law:
                exact_matched_docs.add(doc_id)

        expanded_query = expand_legal_query(query)

        bm25_top = self.bm25_searcher.search(expanded_query, top_k=candidate_k)
        finetuned_top = self.finetuned_searcher.search(expanded_query, top_k=candidate_k)

        scores = {}
        rrf_k = 10
        for rank, doc_id in enumerate(bm25_top):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 1.5

        for rank, doc_id in enumerate(finetuned_top):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 2.5

        if self.jina_searcher:
            jina_top = self.jina_searcher.search(query, top_k=candidate_k)
            for rank, doc_id in enumerate(jina_top):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 3.0

        if self.qwen_searcher:
            qwen_top = self.qwen_searcher.search(query, top_k=candidate_k)
            for rank, doc_id in enumerate(qwen_top):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 3.0

        if self.qwen4b_searcher:
            qwen4b_top = self.qwen4b_searcher.search(query, top_k=candidate_k)
            for rank, doc_id in enumerate(qwen4b_top):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 3.5

        if not self.light_mode and self.dense_searcher and self.e5_searcher:
            dense_top = self.dense_searcher.search(expanded_query, top_k=candidate_k)
            e5_top = self.e5_searcher.search(expanded_query, top_k=candidate_k)
            for rank, doc_id in enumerate(dense_top):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 0.5
            for rank, doc_id in enumerate(e5_top):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * 1.0

        for doc_id in exact_matched_docs:
            scores[doc_id] = scores.get(doc_id, 0.0) + 5.0
            
        for doc_id in mega_boost_docs:
            scores[doc_id] = scores.get(doc_id, 0.0) + 7.0

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [doc_id for doc_id, _ in sorted_docs[:candidate_k]]

        eval_docs = top_candidates[:max_rerank_k]
        rr_scores = self.reranker.score_candidates(query, eval_docs) if (self.reranker and self.reranker.is_active) else np.zeros(len(eval_docs))

        return {
            "top_candidates": top_candidates,
            "rr_scores": rr_scores,
            "candidate_scores": scores,
            "mega_boost_docs": mega_boost_docs,
            "exact_matched_docs": exact_matched_docs
        }

    def fuse_from_cached_scores(self, cached_item, top_k=5, rerank_top_k=30, st1_weight=0.75):
        """Kết hợp điểm từ dữ liệu đã cache trong RAM - cực nhanh (0.0001s / câu)"""
        if self.reranker and self.reranker.is_active:
            return self.reranker.fuse_and_rank(
                top_docs=cached_item["top_candidates"],
                rr_scores=cached_item["rr_scores"],
                candidate_scores=cached_item["candidate_scores"],
                mega_boost_docs=cached_item["mega_boost_docs"],
                exact_matched_docs=cached_item["exact_matched_docs"],
                top_k=top_k,
                st1_weight=st1_weight,
                rerank_top_k=rerank_top_k
            )
        return cached_item["top_candidates"][:top_k]

def create_submission(predictions, output_json="submission.json", output_zip="submission.zip"):
    """
    Tạo file submission.json và đóng gói thành submission.zip theo đúng chuẩn BTC
    """
    formatted_preds = {}
    for qid, docs in predictions.items():
        # Đảm bảo mỗi câu hỏi có tối đa 5 IDs
        formatted_preds[str(qid)] = {
            "answer": [str(d) for d in docs[:5]]
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(formatted_preds, f, ensure_ascii=False, indent=4)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_json, arcname="submission.json")

    print(f"\n🎉 Đã tạo thành công file nộp bài: '{output_zip}' (bên trong chứa '{output_json}')")

if __name__ == "__main__":
    # 1. Nạp dữ liệu
    corpus = load_corpus()
    train_data = load_train_data()
    test_data = load_test_data()

    # 2. Đánh giá thử trên 500 câu Validation để xem điểm Recall tăng lên bao nhiêu
    val_items = list(train_data.items())[:500]
    val_questions = {k: v["question"] for k, v in val_items}
    val_truth = {k: v["answer"] for k, v in val_items}

    hybrid = HybridSearcher(corpus)

    print("\n🔍 Đang chạy tìm kiếm HYBRID trên 500 câu hỏi mẫu...")
    val_preds = {}
    for qid, question in tqdm(val_questions.items(), desc="Hybrid Validation"):
        val_preds[qid] = hybrid.search(question, top_k=5)

    results = compute_metrics(val_preds, val_truth, k=5)
    print("\n" + "="*45)
    print(f"📊 KẾT QUẢ HYBRID SEARCH (BM25 + DENSE):")
    print(f"👉 Recall@5   : {results['Recall'] * 100:.2f}%")
    print(f"👉 Precision@5: {results['Precision'] * 100:.2f}%")
    print("="*45)

    # 3. Chạy dự đoán trên toàn bộ 999 câu hỏi Public Test và tạo file nộp bài
    print("\n📦 Đang sinh kết quả cho 999 câu hỏi Public Test...")
    test_preds = {}
    for qid, item in tqdm(test_data.items(), desc="Predicting Public Test"):
        question = item["question"]
        test_preds[qid] = hybrid.search(question, top_k=5)

    create_submission(test_preds)
