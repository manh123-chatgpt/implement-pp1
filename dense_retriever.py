import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from data_loader import load_corpus, load_train_data
from evaluator import compute_metrics

import re

# BAAI/bge-m3: Legal-Aware Chunking + Max-Pooling trên toàn bộ nội dung luật
MODEL_NAME = "BAAI/bge-m3"
_BGEM3_CACHE_CANDIDATES = [
    "corpus_embeddings_bgem3_resolved.pkl",
    "/kaggle/working/corpus_embeddings_bgem3_resolved.pkl",
    "/kaggle/input/notebooks/thurdayafternoon/legal-ir/corpus_embeddings_bgem3_resolved.pkl",
    "/kaggle/input/datasets/thurdayafternoon/pkl-cache/corpus_embeddings_bgem3_resolved.pkl",
]
EMBEDDINGS_CACHE = next((p for p in _BGEM3_CACHE_CANDIDATES if os.path.exists(p)), _BGEM3_CACHE_CANDIDATES[0])

def chunk_document(text, chunk_size=1200, overlap=200, max_chunks=150):
    """
    Fallback: Chia tài liệu theo ký tự cố định.
    Dùng khi tài liệu không có cấu trúc pháp lý (Điều).
    """
    lines = text.split("\n", 1)
    title = lines[0] if lines else ""
    body = lines[1] if len(lines) > 1 else text
    
    if len(body) <= chunk_size:
        return [f"{title}\n{body}"]
        
    chunks = []
    start = 0
    while start < len(body) and len(chunks) < max_chunks:
        end = start + chunk_size
        chunk_text = body[start:end]
        chunks.append(f"{title}\n{chunk_text}")
        start += (chunk_size - overlap)
        
    return chunks

def _split_by_sub_structure(text, max_chars):
    """
    Chia nhỏ một phần văn bản pháp lý (VD: 1 Điều quá dài) theo cấu trúc con:
    Level 1: Khoản (Paragraph) — ký hiệu số Ả Rập: "1. ", "2. ", "3. "
    Level 2: Điểm (Point) — ký hiệu chữ cái: "a) ", "b) ", "c) "
    Level 3: Gạch đầu dòng: "- "
    
    Trả về danh sách các phần đã chia nhỏ. Nếu không tìm thấy cấu trúc con,
    trả về danh sách chỉ chứa text gốc.
    """
    # Level 1: Tách theo Khoản (1., 2., 3., ...)
    # Pattern: dòng mới + số + dấu chấm + khoảng trắng (tránh match số hiệu luật kiểu 136/2020)
    khoan_parts = re.split(r'(?=\n\d+\.\s)', text)
    khoan_parts = [p.strip() for p in khoan_parts if p.strip()]
    
    if len(khoan_parts) >= 2:
        # Thành công chia theo Khoản!
        result = []
        for kp in khoan_parts:
            if len(kp) <= max_chars:
                result.append(kp)
            else:
                # Khoản vẫn quá dài → thử chia tiếp theo Điểm (a), b), c))
                diem_parts = re.split(r'(?=\n[a-zđ]\)\s)', kp)
                diem_parts = [dp.strip() for dp in diem_parts if dp.strip()]
                if len(diem_parts) >= 2:
                    result.extend(diem_parts)
                else:
                    # Thử chia theo gạch đầu dòng (- )
                    dash_parts = re.split(r'(?=\n-\s)', kp)
                    dash_parts = [dp.strip() for dp in dash_parts if dp.strip()]
                    if len(dash_parts) >= 2:
                        result.extend(dash_parts)
                    else:
                        result.append(kp)
        return result
    
    # Level 2: Nếu không có Khoản, thử tách theo Điểm trực tiếp
    diem_parts = re.split(r'(?=\n[a-zđ]\)\s)', text)
    diem_parts = [p.strip() for p in diem_parts if p.strip()]
    if len(diem_parts) >= 2:
        return diem_parts
    
    # Level 3: Thử tách theo gạch đầu dòng
    dash_parts = re.split(r'(?=\n-\s)', text)
    dash_parts = [p.strip() for p in dash_parts if p.strip()]
    if len(dash_parts) >= 2:
        return dash_parts
    
    # Không tìm thấy cấu trúc con
    return [text]


def legal_chunk_document(text, max_chunk_chars=1200, overlap=200, max_chunks=150):
    """
    Chia tài liệu theo cấu trúc pháp lý phân cấp (Hierarchical Legal Chunking):
    
    Level 0: Điều (Article) — "Điều 1", "Điều 2", ...
    Level 1: Khoản (Paragraph) — "1. ", "2. ", "3. ", ...
    Level 2: Điểm (Point) — "a) ", "b) ", "c) ", ...
    Level 3: Gạch đầu dòng — "- "
    
    Quy tắc:
    - 91% tài liệu có cấu trúc "Điều" → chia theo Điều trước.
    - Nếu 1 Điều quá dài (> max_chunk_chars) → chia tiếp theo Khoản/Điểm/Gạch đầu dòng.
    - Nếu vẫn quá dài → fallback về character chunking.
    - Nếu không có cấu trúc "Điều" → fallback hoàn toàn.
    """
    lines = text.split("\n", 1)
    title = lines[0] if lines else ""
    body = lines[1] if len(lines) > 1 else text
    
    # Chia theo ranh giới "Điều X" (lookahead để giữ lại từ "Điều")
    dieu_parts = re.split(r'(?=Điều\s+\d+)', body)
    dieu_parts = [p.strip() for p in dieu_parts if p.strip()]
    
    # Nếu không tìm thấy cấu trúc Điều → thử chia theo Khoản/Điểm trực tiếp
    if len(dieu_parts) < 2:
        # Có thể văn bản chỉ là 1 Điều duy nhất nhưng có nhiều Khoản
        sub_parts = _split_by_sub_structure(body, max_chunk_chars)
        if len(sub_parts) >= 2:
            dieu_parts = sub_parts
        else:
            return chunk_document(text, chunk_size=max_chunk_chars, overlap=overlap, max_chunks=max_chunks)
    
    # Bước 2: Xử lý từng Điều — nếu quá dài, chia nhỏ theo Khoản/Điểm
    all_semantic_parts = []
    for part in dieu_parts:
        if len(part) <= max_chunk_chars:
            all_semantic_parts.append(part)
        else:
            # Điều quá dài → chia nhỏ theo cấu trúc con (Khoản → Điểm → Gạch đầu dòng)
            sub_parts = _split_by_sub_structure(part, max_chunk_chars)
            
            # Giữ lại header của Điều (VD: "Điều 15. Quyền của người lao động")
            dieu_header = ""
            first_newline = part.find("\n")
            if first_newline > 0 and first_newline < 200:
                dieu_header = part[:first_newline].strip()
            
            for sp in sub_parts:
                # Nếu sub_part không bắt đầu bằng "Điều", thêm header để giữ ngữ cảnh
                if dieu_header and not sp.startswith("Điều"):
                    sp = f"{dieu_header}\n{sp}"
                all_semantic_parts.append(sp)
    
    # Bước 3: Gộp các phần ngắn lại và xử lý phần vẫn quá dài
    chunks = []
    buffer = ""
    
    for part in all_semantic_parts:
        if buffer and len(buffer) + len(part) <= max_chunk_chars:
            buffer = buffer + "\n" + part
        else:
            if buffer:
                chunks.append(f"{title}\n{buffer}")
            
            # Phần vẫn quá dài sau khi chia theo cấu trúc con → character chunking
            if len(part) > max_chunk_chars:
                start = 0
                while start < len(part) and len(chunks) < max_chunks:
                    end = start + max_chunk_chars
                    chunks.append(f"{title}\n{part[start:end]}")
                    start += max_chunk_chars - overlap
                buffer = ""
            else:
                buffer = part
        
        if len(chunks) >= max_chunks:
            break
    
    # Đừng quên phần cuối cùng
    if buffer and len(chunks) < max_chunks:
        chunks.append(f"{title}\n{buffer}")
    
    return chunks[:max_chunks] if chunks else [text[:max_chunk_chars]]

class DenseSearcher:
    """BGE-M3 với Legal-Aware Chunking (chia theo Điều luật)"""
    def __init__(self, corpus, model_name=MODEL_NAME, use_cache=True):
        self.unique_doc_ids = list(corpus.keys())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Đang tải mô hình SOTA Embedding '{model_name}' trên thiết bị: {device.upper()} (FP16 Mode)...")
        
        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"use_safetensors": True, "torch_dtype": torch.float16 if device == "cuda" else torch.float32}
        )
        self.model.max_seq_length = 512

        loaded = False
        if use_cache and os.path.exists(EMBEDDINGS_CACHE):
            print(f"⚡ Đang nạp Full-Chunked Embeddings từ '{EMBEDDINGS_CACHE}'...")
            try:
                with open(EMBEDDINGS_CACHE, "rb") as f:
                    data = pickle.load(f)
                self.unique_doc_ids = data["unique_doc_ids"]
                self.chunk_doc_ids = data["chunk_doc_ids"]
                self.corpus_embeddings = data["embeddings"]
                loaded = True
                print(f"✅ Nạp thành công cache BGE-M3 ({len(self.corpus_embeddings)} vectors).")
            except Exception as e:
                print(f"⚠️ Cache BGE-M3 '{EMBEDDINGS_CACHE}' bị lỗi hoặc không hoàn chỉnh ({e}). Đang tái tạo embeddings...")
                loaded = False

        if not loaded:
            print("⚡ Đang LEGAL CHUNKING toàn bộ văn bản (bge-m3 FP16)...")
            all_chunks = []
            chunk_doc_ids = []
            
            for doc_id in tqdm(self.unique_doc_ids, desc="Legal Chunking Corpus"):
                full_text = corpus[doc_id]
                # LEGAL CHUNKING: Chia theo Điều luật, max_chunk=1200 cho BGE-M3 (512 tokens)
                doc_chunks = legal_chunk_document(full_text, max_chunk_chars=1200, overlap=200, max_chunks=150)
                for ch in doc_chunks:
                    all_chunks.append(ch)
                    chunk_doc_ids.append(doc_id)

            print(f"📊 Tổng số chunks: {len(all_chunks)} (trung bình {len(all_chunks)/len(self.unique_doc_ids):.1f} chunks/doc)")
            
            self.chunk_doc_ids = np.array(chunk_doc_ids)
            self.corpus_embeddings = self.model.encode(
                all_chunks,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            with open(EMBEDDINGS_CACHE, "wb") as f:
                pickle.dump({
                    "unique_doc_ids": self.unique_doc_ids,
                    "chunk_doc_ids": self.chunk_doc_ids,
                    "embeddings": self.corpus_embeddings
                }, f)
            print(f"✅ Đã lưu legal-chunked embeddings vào '{EMBEDDINGS_CACHE}'.")

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        chunk_scores = np.dot(self.corpus_embeddings, query_embedding)
        
        doc_max_scores = {}
        for doc_id, score in zip(self.chunk_doc_ids, chunk_scores):
            if doc_id not in doc_max_scores or score > doc_max_scores[doc_id]:
                doc_max_scores[doc_id] = score
                
        sorted_docs = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

# Fine-tuned Vietnamese Bi-Encoder Super - GIỜ CŨNG CÓ LEGAL CHUNKING
import os as _os
_model_candidates = [
    "fine_tuned_vietnamese_bi_encoder",
    "/kaggle/working/fine_tuned_vietnamese_bi_encoder",
    "/kaggle/input/datasets/thurdayafternoon/uit-legal-finetuned-model/fine_tuned_vietnamese_bi_encoder",
    "/kaggle/input/uit-legal-finetuned-model/fine_tuned_vietnamese_bi_encoder",
    "/kaggle/input/uit-legal-finetuned-model",
    "bkai-foundation-models/vietnamese-bi-encoder"
]
MODEL_FINETUNED_DIR = next((p for p in _model_candidates if _os.path.exists(p) or p == "bkai-foundation-models/vietnamese-bi-encoder"), _model_candidates[-1])
_FT_CACHE_CANDIDATES = [
    "corpus_embeddings_finetuned_resolved.pkl",
    "/kaggle/working/corpus_embeddings_finetuned_resolved.pkl",
    "/kaggle/input/notebooks/thurdayafternoon/legal-ir/corpus_embeddings_finetuned_resolved.pkl",
    "/kaggle/input/datasets/thurdayafternoon/pkl-cache/corpus_embeddings_finetuned_resolved.pkl",
]
CACHE_FINETUNED = next((p for p in _FT_CACHE_CANDIDATES if _os.path.exists(p)), _FT_CACHE_CANDIDATES[0])

class FineTunedDenseSearcher:
    """RoBERTa Fine-tuned với Legal-Aware Chunking (chia theo Điều luật)"""
    def __init__(self, corpus, model_path=MODEL_FINETUNED_DIR, use_cache=True):
        self.unique_doc_ids = list(corpus.keys())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Đang nạp mô hình FINE-TUNED SUPER từ '{model_path}'...")
        self.model = SentenceTransformer(model_path, device=device)
        self.model.max_seq_length = 256

        loaded = False
        if use_cache and os.path.exists(CACHE_FINETUNED):
            print(f"⚡ Đang nạp Fine-tuned Full-Chunked Embeddings từ '{CACHE_FINETUNED}'...")
            try:
                with open(CACHE_FINETUNED, "rb") as f:
                    data = pickle.load(f)
                self.unique_doc_ids = data["unique_doc_ids"]
                self.chunk_doc_ids = data["chunk_doc_ids"]
                self.corpus_embeddings = data["embeddings"]
                loaded = True
                print(f"✅ Nạp thành công cache Fine-tuned Bi-Encoder ({len(self.corpus_embeddings)} vectors).")
            except Exception as e:
                print(f"⚠️ Cache Fine-tuned Bi-Encoder '{CACHE_FINETUNED}' bị lỗi hoặc không hoàn chỉnh ({e}). Đang tái tạo embeddings...")
                loaded = False

        if not loaded:
            print("⚡ Đang LEGAL CHUNKING + Encode từ mô hình Fine-tuned Super...")
            all_chunks = []
            chunk_doc_ids = []
            
            for doc_id in tqdm(self.unique_doc_ids, desc="Legal Chunking for RoBERTa"):
                full_text = corpus[doc_id]
                # RoBERTa max_seq=256 (~600 ký tự), chunk nhỏ hơn BGE-M3
                doc_chunks = legal_chunk_document(full_text, max_chunk_chars=600, overlap=100, max_chunks=150)
                for ch in doc_chunks:
                    all_chunks.append(ch)
                    chunk_doc_ids.append(doc_id)

            print(f"📊 Tổng số chunks RoBERTa: {len(all_chunks)} (trung bình {len(all_chunks)/len(self.unique_doc_ids):.1f} chunks/doc)")
            
            self.chunk_doc_ids = np.array(chunk_doc_ids)
            self.corpus_embeddings = self.model.encode(
                all_chunks,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            with open(CACHE_FINETUNED, "wb") as f:
                pickle.dump({
                    "unique_doc_ids": self.unique_doc_ids,
                    "chunk_doc_ids": self.chunk_doc_ids,
                    "embeddings": self.corpus_embeddings
                }, f)
            print(f"✅ Đã lưu fine-tuned legal-chunked embeddings vào '{CACHE_FINETUNED}'.")

    def search(self, query, top_k=5):
        """Max-Pooling: Lấy score cao nhất trong tất cả chunks của mỗi doc"""
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        chunk_scores = np.dot(self.corpus_embeddings, query_embedding)
        
        doc_max_scores = {}
        for doc_id, score in zip(self.chunk_doc_ids, chunk_scores):
            if doc_id not in doc_max_scores or score > doc_max_scores[doc_id]:
                doc_max_scores[doc_id] = score
                
        sorted_docs = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

# E5-Large Multilingual: Mô hình 560M params mạnh mẽ với Instruction-based Encoding
MODEL_E5_NAME = "intfloat/multilingual-e5-large"
_E5_CACHE_CANDIDATES = [
    "corpus_embeddings_e5_resolved.pkl",
    "/kaggle/working/corpus_embeddings_e5_resolved.pkl",
    "/kaggle/input/notebooks/thurdayafternoon/legal-ir/corpus_embeddings_e5_resolved.pkl",
    "/kaggle/input/datasets/thurdayafternoon/pkl-cache/corpus_embeddings_e5_resolved.pkl",
]
CACHE_E5 = next((p for p in _E5_CACHE_CANDIDATES if _os.path.exists(p)), _E5_CACHE_CANDIDATES[0])

class E5LargeSearcher:
    """E5-Large với Legal-Aware Chunking. BẮT BUỘC thêm prefix 'query: ' và 'passage: '"""
    def __init__(self, corpus, model_name=MODEL_E5_NAME, use_cache=True):
        self.unique_doc_ids = list(corpus.keys())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Đang tải mô hình SOTA E5-Large '{model_name}' trên thiết bị: {device.upper()} (FP16 Mode)...")
        
        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"use_safetensors": True, "torch_dtype": torch.float16 if device == "cuda" else torch.float32}
        )
        # E5-Large hỗ trợ tới 512 tokens
        self.model.max_seq_length = 512

        loaded = False
        if use_cache and os.path.exists(CACHE_E5):
            print(f"⚡ Đang nạp E5-Large Embeddings từ '{CACHE_E5}'...")
            try:
                with open(CACHE_E5, "rb") as f:
                    data = pickle.load(f)
                self.unique_doc_ids = data["unique_doc_ids"]
                self.chunk_doc_ids = data["chunk_doc_ids"]
                self.corpus_embeddings = data["embeddings"]
                loaded = True
                print(f"✅ Nạp thành công cache E5-Large ({len(self.corpus_embeddings)} vectors).")
            except Exception as e:
                print(f"⚠️ Cache E5-Large '{CACHE_E5}' bị lỗi hoặc không hoàn chỉnh ({e}). Đang tái tạo embeddings...")
                loaded = False

        if not loaded:
            print("⚡ Đang LEGAL CHUNKING toàn bộ văn bản cho E5-Large...")
            all_chunks = []
            chunk_doc_ids = []
            
            for doc_id in tqdm(self.unique_doc_ids, desc="Legal Chunking for E5"):
                full_text = corpus[doc_id]
                # E5 cũng nhận tối đa 512 tokens (~1200 chars)
                doc_chunks = legal_chunk_document(full_text, max_chunk_chars=1200, overlap=200, max_chunks=150)
                for ch in doc_chunks:
                    # BẮT BUỘC prefix 'passage: ' cho E5
                    all_chunks.append(f"passage: {ch}")
                    chunk_doc_ids.append(doc_id)

            print(f"📊 Tổng số chunks E5: {len(all_chunks)} (trung bình {len(all_chunks)/len(self.unique_doc_ids):.1f} chunks/doc)")
            
            self.chunk_doc_ids = np.array(chunk_doc_ids)
            self.corpus_embeddings = self.model.encode(
                all_chunks,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            with open(CACHE_E5, "wb") as f:
                pickle.dump({
                    "unique_doc_ids": self.unique_doc_ids,
                    "chunk_doc_ids": self.chunk_doc_ids,
                    "embeddings": self.corpus_embeddings
                }, f)
            print(f"✅ Đã lưu E5-Large embeddings vào '{CACHE_E5}'.")

    def search(self, query, top_k=5):
        # BẮT BUỘC prefix 'query: ' cho E5
        e5_query = f"query: {query}"
        query_embedding = self.model.encode(e5_query, normalize_embeddings=True)
        chunk_scores = np.dot(self.corpus_embeddings, query_embedding)
        
        doc_max_scores = {}
        for doc_id, score in zip(self.chunk_doc_ids, chunk_scores):
            if doc_id not in doc_max_scores or score > doc_max_scores[doc_id]:
                doc_max_scores[doc_id] = score
                
        sorted_docs = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

# Vietnamese Legal Embedding: Mô hình chuyên biệt huấn luyện ĐỘC QUYỀN trên văn bản pháp luật VN
MODEL_LEGAL_NAME = "bqbbao6/vietnamese-legal-embedding"
CACHE_LEGAL = "corpus_embeddings_vietnamese_legal_resolved.pkl"

class LegalEmbeddingSearcher:
    """Legal Embedding Searcher"""
    def __init__(self, corpus, model_name=MODEL_LEGAL_NAME, use_cache=True):
        self.unique_doc_ids = list(corpus.keys())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Đang tải mô hình CHUYÊN GIA LUẬT '{model_name}' trên thiết bị: {device.upper()} (FP16 Mode)...")
        
        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
        )
        self.model.max_seq_length = 512

        if use_cache and os.path.exists(CACHE_LEGAL):
            print(f"⚡ Đang nạp Legal Embeddings từ '{CACHE_LEGAL}'...")
            with open(CACHE_LEGAL, "rb") as f:
                data = pickle.load(f)
                self.unique_doc_ids = data["unique_doc_ids"]
                self.chunk_doc_ids = data["chunk_doc_ids"]
                self.corpus_embeddings = data["embeddings"]
        else:
            print("⚡ Đang LEGAL CHUNKING toàn bộ văn bản cho mô hình chuyên biệt...")
            all_chunks = []
            chunk_doc_ids = []
            
            for doc_id in tqdm(self.unique_doc_ids, desc="Legal Chunking for Legal Model"):
                full_text = corpus[doc_id]
                chunks = legal_chunk_document(full_text, max_chunk_chars=1200, overlap=200, max_chunks=50)
                
                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_doc_ids.append(doc_id)
            
            self.chunk_doc_ids = chunk_doc_ids
            
            print(f"⚡ Đang mã hóa (embed) {len(all_chunks)} chunks chuyên sâu. Quá trình này sẽ mất 5-10 phút...")
            self.corpus_embeddings = self.model.encode(
                all_chunks,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            with open(CACHE_LEGAL, "wb") as f:
                pickle.dump({
                    "unique_doc_ids": self.unique_doc_ids,
                    "chunk_doc_ids": self.chunk_doc_ids,
                    "embeddings": self.corpus_embeddings
                }, f)
            print(f"✅ Đã lưu legal-chunked embeddings vào '{CACHE_LEGAL}'.")

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        chunk_scores = np.dot(self.corpus_embeddings, query_embedding)
        
        doc_max_scores = {}
        for doc_id, score in zip(self.chunk_doc_ids, chunk_scores):
            if doc_id not in doc_max_scores or score > doc_max_scores[doc_id]:
                doc_max_scores[doc_id] = score
                
        sorted_docs = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]


# =====================================================================
# 🌟 THẾ HỆ MỚI: Jina-Embeddings-v3 (8.192 tokens Context, SOTA Toàn cầu)
# =====================================================================
MODEL_JINA_NAME = "jinaai/jina-embeddings-v3"
_JINA_CACHE_CANDIDATES = [
    "corpus_embeddings_jina_v3_resolved.pkl",
    "/kaggle/working/corpus_embeddings_jina_v3_resolved.pkl",
    "/kaggle/input/notebooks/thurdayafternoon/legal-ir/corpus_embeddings_jina_v3_resolved.pkl",
    "/kaggle/input/datasets/thurdayafternoon/pkl-cache/corpus_embeddings_jina_v3_resolved.pkl",
]
CACHE_JINA = next((p for p in _JINA_CACHE_CANDIDATES if os.path.exists(p) and os.path.getsize(p) > 1024*1024), _JINA_CACHE_CANDIDATES[0])

class JinaEmbeddingSearcher:
    """
    Jina Embeddings v3:
    - Context Window siêu dài 8.192 tokens (đọc trọn vẹn mọi điều luật dài mà không bị mất chữ).
    - Task-specific LoRA:
        + 'retrieval.passage' cho văn bản luật
        + 'retrieval.query' cho câu hỏi
    """
    def __init__(self, corpus, model_name=MODEL_JINA_NAME, use_cache=True):
        self.unique_doc_ids = list(corpus.keys())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🌟 Đang tải mô hình SOTA '{model_name}' trên thiết bị: {device.upper()} (Context 8.192)...")
        
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
        )
        self.model.max_seq_length = 8192

        if use_cache and os.path.exists(CACHE_JINA) and os.path.getsize(CACHE_JINA) > 1024*1024:
            print(f"⚡ Đang nạp Jina-v3 Embeddings từ '{CACHE_JINA}'...")
            with open(CACHE_JINA, "rb") as f:
                data = pickle.load(f)
                self.unique_doc_ids = data["unique_doc_ids"]
                self.chunk_doc_ids = data["chunk_doc_ids"]
                self.corpus_embeddings = data["embeddings"]
        else:
            print("⚡ Đang chuẩn bị văn bản cho Jina-v3 (hỗ trợ context dài 4000 ký tự)...")
            all_chunks = []
            chunk_doc_ids = []
            
            for doc_id in tqdm(self.unique_doc_ids, desc="Chuẩn bị dữ liệu Jina-v3"):
                full_text = corpus[doc_id]
                # Context 8k cho phép chunk kích thước lớn 4000 ký tự mà không bị cắt cụt
                chunks = legal_chunk_document(full_text, max_chunk_chars=4000, overlap=300, max_chunks=20)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_doc_ids.append(doc_id)
            
            self.chunk_doc_ids = chunk_doc_ids
            
            print(f"⚡ Đang mã hóa {len(all_chunks)} đoạn văn bản với Jina-v3 (task='retrieval.passage')...")
            self.corpus_embeddings = self.model.encode(
                all_chunks,
                batch_size=32 if device == "cuda" else 8,
                show_progress_bar=True,
                normalize_embeddings=True,
                task="retrieval.passage"
            )
            
            with open(CACHE_JINA, "wb") as f:
                pickle.dump({
                    "unique_doc_ids": self.unique_doc_ids,
                    "chunk_doc_ids": self.chunk_doc_ids,
                    "embeddings": self.corpus_embeddings
                }, f)
            print(f"✅ Đã lưu Jina-v3 embeddings vào '{CACHE_JINA}'.")

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(query, normalize_embeddings=True, task="retrieval.query")
        chunk_scores = np.dot(self.corpus_embeddings, query_embedding)
        
        doc_max_scores = {}
        for doc_id, score in zip(self.chunk_doc_ids, chunk_scores):
            if doc_id not in doc_max_scores or score > doc_max_scores[doc_id]:
                doc_max_scores[doc_id] = score
                
        sorted_docs = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]


# =====================================================================
# ⚖️ THẾ HỆ MỚI: Qwen3 Vietnamese Legal Embedding (Chuyên gia Luật VN)
# =====================================================================
MODEL_QWEN_LEGAL = "CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3"
MODEL_QWEN_FALLBACK = "Qwen/Qwen3-Embedding-0.6B"

_QWEN_CACHE_CANDIDATES = [
    "corpus_embeddings_qwen3_legal_resolved.pkl",
    "/kaggle/working/corpus_embeddings_qwen3_legal_resolved.pkl",
    "/kaggle/input/notebooks/thurdayafternoon/legal-ir/corpus_embeddings_qwen3_legal_resolved.pkl",
    "/kaggle/input/datasets/thurdayafternoon/pkl-cache/corpus_embeddings_qwen3_legal_resolved.pkl",
]
CACHE_QWEN = next((p for p in _QWEN_CACHE_CANDIDATES if os.path.exists(p) and os.path.getsize(p) > 1024*1024), _QWEN_CACHE_CANDIDATES[0])

class QwenLegalEmbeddingSearcher:
    """
    Qwen3 Vietnamese Legal Embedding:
    - Huấn luyện chuyên sâu trên dữ liệu Pháp luật Việt Nam.
    - Context 4.096 - 8.192 tokens.
    - Nhẹ (~0.6B tham số), tốc độ tính toán nhanh và tiết kiệm VRAM.
    """
    def __init__(self, corpus, model_name=MODEL_QWEN_LEGAL, use_cache=True):
        self.unique_doc_ids = list(corpus.keys())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"⚖️ Đang tải mô hình Qwen Legal '{model_name}' trên thiết bị: {device.upper()}...")
        try:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
            )
        except Exception as e:
            print(f"⚠️ Không tải được {model_name}: {e}. Chuyển sang fallback '{MODEL_QWEN_FALLBACK}'...")
            self.model = SentenceTransformer(
                MODEL_QWEN_FALLBACK,
                device=device,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
            )
        self.model.max_seq_length = 4096

        if use_cache and os.path.exists(CACHE_QWEN) and os.path.getsize(CACHE_QWEN) > 1024*1024:
            print(f"⚡ Đang nạp Qwen Legal Embeddings từ '{CACHE_QWEN}'...")
            with open(CACHE_QWEN, "rb") as f:
                data = pickle.load(f)
                self.unique_doc_ids = data["unique_doc_ids"]
                self.chunk_doc_ids = data["chunk_doc_ids"]
                self.corpus_embeddings = data["embeddings"]
        else:
            print("⚡ Đang chuẩn bị văn bản cho Qwen Legal (context 3000 ký tự)...")
            all_chunks = []
            chunk_doc_ids = []
            
            for doc_id in tqdm(self.unique_doc_ids, desc="Chuẩn bị dữ liệu Qwen Legal"):
                full_text = corpus[doc_id]
                chunks = legal_chunk_document(full_text, max_chunk_chars=3000, overlap=250, max_chunks=25)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_doc_ids.append(doc_id)
            
            self.chunk_doc_ids = chunk_doc_ids
            
            print(f"⚡ Đang mã hóa {len(all_chunks)} đoạn văn bản với Qwen Legal...")
            self.corpus_embeddings = self.model.encode(
                all_chunks,
                batch_size=32 if device == "cuda" else 8,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            with open(CACHE_QWEN, "wb") as f:
                pickle.dump({
                    "unique_doc_ids": self.unique_doc_ids,
                    "chunk_doc_ids": self.chunk_doc_ids,
                    "embeddings": self.corpus_embeddings
                }, f)
            print(f"✅ Đã lưu Qwen Legal embeddings vào '{CACHE_QWEN}'.")

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        chunk_scores = np.dot(self.corpus_embeddings, query_embedding)
        
        doc_max_scores = {}
        for doc_id, score in zip(self.chunk_doc_ids, chunk_scores):
            if doc_id not in doc_max_scores or score > doc_max_scores[doc_id]:
                doc_max_scores[doc_id] = score
                
        sorted_docs = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]
