import os
import gc
import json
import torch
from tqdm import tqdm

def expand_queries_with_llm(
    query_dict,
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    cache_path="llm_expanded_queries_cache.json",
    batch_size=16,
    max_new_tokens=40,
    force_recompute=False
):
    """
    Sử dụng LLM nhỏ trong Whitelist (mặc định Qwen2.5-1.5B-Instruct hoặc Vi-Qwen2)
    để dịch câu hỏi dân gian sang thuật ngữ pháp lý chuẩn và văn bản luật liên quan.
    
    Tự động:
    1. Kiểm tra cache .json: Nếu đã chạy rồi thì nạp lại trong 0.01s.
    2. Batching sinh siêu tốc (greedy decode, max 40 tokens) trên GPU.
    3. Giải phóng 100% VRAM (del model + cuda.empty_cache()) ngay sau khi hoàn thành.
    
    Args:
        query_dict (dict): {qid: question_text}
        model_name (str): Tên model HF hoặc đường dẫn local
        cache_path (str): Đường dẫn lưu file cache JSON
        batch_size (int): Kích thước batch sinh trên GPU (16 cho T4 16GB)
        max_new_tokens (int): Số lượng token tối đa cho thuật ngữ pháp lý
        force_recompute (bool): Bắt buộc chạy lại dù đã có cache
        
    Returns:
        dict: {qid: expanded_query_text} (Câu hỏi gốc + thuật ngữ pháp lý)
    """
    if not query_dict:
        return {}

    # 1. Kiểm tra Cache
    cached_results = {}
    if os.path.exists(cache_path) and not force_recompute:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_results = json.load(f)
            print(f"📦 Đã tìm thấy cache LLM Query Expansion ({len(cached_results)} câu) tại: {cache_path}")
        except Exception as e:
            print(f"⚠️ Lỗi đọc cache ({e}), sẽ tiến hành mở rộng mới.")
            cached_results = {}

    # Lọc ra các câu hỏi chưa có trong cache
    missing_items = [(qid, q) for qid, q in query_dict.items() if qid not in cached_results]
    
    if not missing_items:
        print(f"✅ Toàn bộ {len(query_dict)} câu hỏi đã có sẵn trong cache! Không cần nạp LLM.")
        return {qid: cached_results[qid] for qid in query_dict if qid in cached_results}

    print(f"🚀 Cần mở rộng {len(missing_items)} / {len(query_dict)} câu hỏi bằng LLM: {model_name}")

    # 2. Nạp Model & Tokenizer
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("⚠️ Chưa cài transformers. Trả về câu hỏi gốc.")
        return dict(query_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Thiết bị chạy LLM: {device.upper()}")
    
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() else torch.float32)

    try:
        print(f"⏳ Đang nạp mô hình {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Bắt buộc cho decoder batch generation

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        if device == "cpu":
            model = model.to("cpu")
        model.eval()
        print("✅ Nạp LLM thành công!")
    except Exception as e:
        print(f"❌ Không thể nạp mô hình {model_name}: {e}")
        print("⚠️ Fallback: Tiếp tục với câu hỏi gốc.")
        return dict(query_dict)

    # 3. Batch Inference sinh từ khóa pháp lý
    system_prompt = (
        "Bạn là chuyên gia tra cứu pháp luật Việt Nam. "
        "Nhiệm vụ: Chuyển đổi câu hỏi của người dân sang 3-5 thuật ngữ pháp lý chính thức, hành vi pháp lý, hoặc tên luật/nghị định liên quan. "
        "Chỉ trả về danh sách từ khóa ngắn gọn cách nhau bằng dấu phẩy. Tuyệt đối không giải thích dông dài."
    )

    pbar = tqdm(range(0, len(missing_items), batch_size), desc="LLM Expanding Queries")
    for start_idx in pbar:
        batch = missing_items[start_idx:start_idx + batch_size]
        prompts = []
        for qid, q_text in batch:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Câu hỏi: {q_text}\nThuật ngữ pháp lý:"}
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"{system_prompt}\nCâu hỏi: {q_text}\nThuật ngữ pháp lý:"
            prompts.append(prompt)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Cắt lấy phần token mới sinh ra
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for (qid, q_text), generated_text in zip(batch, decoded):
            clean_terms = generated_text.strip().replace("\n", " ")
            # Loại bỏ tiền tố thừa nếu có
            clean_terms = clean_terms.replace("Thuật ngữ pháp lý:", "").strip()
            # Kết hợp: Câu hỏi gốc + các thuật ngữ pháp lý bổ sung
            if clean_terms:
                expanded_q = f"{q_text} {clean_terms}"
            else:
                expanded_q = q_text
            cached_results[qid] = expanded_q

    # 4. Lưu Cache ra đĩa
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cached_results, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu cache mở rộng truy vấn vào: {cache_path}")
    except Exception as e:
        print(f"⚠️ Lỗi khi lưu cache: {e}")

    # 5. GIẢI PHÓNG 100% VRAM (BẮT BUỘC ĐỂ KHÔNG CHÁY VRAM Ở GIAI ĐOẠN SAU)
    print("🧹 Đang dọn dẹp và giải phóng hoàn toàn VRAM của LLM...")
    del model
    del tokenizer
    del inputs
    del outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✨ VRAM đã sạch 100%, sẵn sàng cho Stage 1 & Reranker!")

    # Trả về kết quả theo đúng thứ tự của query_dict đầu vào
    return {qid: cached_results.get(qid, query_dict[qid]) for qid in query_dict}


if __name__ == "__main__":
    # Test thử chức năng mở rộng
    sample_queries = {
        "q1": "đi xe máy vượt đèn đỏ bị phạt bao nhiêu tiền",
        "q2": "ly hôn đơn phương cần những giấy tờ gì và nộp ở đâu"
    }
    print("Testing expand_queries_with_llm function structure...")
