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


# Danh bạ phục hồi 20 văn bản bị rỗng của BTC (trích xuất từ URL thuvienphapluat chính thức)
RESCUED_EMPTY_DOCS = {
    "10533": "QCVN 20:2015/BLĐTBXH Quy chuẩn kỹ thuật quốc gia về an toàn lao động đối với sàn nâng dùng để nâng người đơn vị lắp đặt trang thiết bị kỹ thuật",
    "131890": "TCVN 8400-24:2014 Bệnh động vật Quy trình chẩn đoán Bệnh viêm phế quản truyền nhiễm ở gà thiết bị dụng cụ máu gà",
    "149317": "TCVN 13274:2020 Truy xuất nguồn gốc Hướng dẫn định dạng mã dùng cho truy vết vật phẩm tài sản doanh nghiệp",
    "177151": "TCVN 10065:2013 Yêu cầu an toàn sản phẩm tiêu dùng Trang sức dành cho trẻ em",
    "181693": "Quyết định cơ chế quản lý tài chính về bảo hiểm xã hội bảo hiểm thất nghiệp bảo hiểm y tế",
    "187338": "QCVN 02-23:2017/BNNPTNT Cơ sở sản xuất kinh doanh thủy sản nhỏ lẻ",
    "191261": "Thông tư vị trí việc làm cơ cấu viên chức cơ sở giáo dục mầm non công lập",
    "196918": "TCVN 9251:2012 Bìa hồ sơ lưu trữ",
    "208668": "Nghị định hướng dẫn Luật Giáo dục đại học",
    "210808": "TCVN 5687:2010 Thông gió Điều hòa không khí Tiêu chuẩn thiết kế",
    "232489": "QCVN 03:2014/BCT Trang thiết bị phụ trợ phương tiện sử dụng etanol nhiên liệu",
    "255762": "TCVN 7795:2021 Biệt thự du lịch Xếp hạng",
    "263763": "QCVN 01:2019/BCT An toàn trong sản xuất thử nghiệm vật liệu nổ công nghiệp nổ mìn thăm dò địa chấn sông biển hủy vật liệu cháy nổ",
    "288457": "Luật Bảo hiểm y tế 2008 Khái niệm bảo hiểm y tế phương thức chi trả khám chữa bệnh nguyên tắc bảo hiểm y tế",
    "34810": "Thông tư quy trình xử lý vi phạm hành chính của Bộ đội Biên phòng",
    "55497": "QCVN 16-1:2015/BYT Quy chuẩn kỹ thuật quốc gia đối với thuốc lá điếu công bố hợp quy",
    "56098": "TCVN 6102:2020 Phòng cháy chữa cháy Chất chữa cháy Bột chữa cháy",
    "57978": "TCVN 10778:2015 Hồ chứa Xác định các mực nước đặc trưng",
    "67660": "QCVN 46:2022/BTNMT Quy chuẩn kỹ thuật quốc gia về quan trắc khí tượng",
    "71014": "QCVN 02-03:2009/BNNPTNT Cơ sở chế biến thủy sản ăn liền"
}

def extract_title_from_url(url):
    """Trích xuất tên văn bản & số hiệu từ URL khi passage bị rỗng"""
    if not url:
        return ""
    filename = url.split('/')[-1]
    base = filename.rsplit('.', 1)[0]
    base = re.sub(r'-\d+$', '', base)
    title = base.replace('-', ' ').strip()
    title = re.sub(r'\b(QCVN|TCVN)\s+(\d+)\s+(\d+)\s+(\d+)\s+([A-Za-z]+)\b', r'\1 \2-\3:\4/\5', title, flags=re.IGNORECASE)
    title = re.sub(r'\b(QCVN|TCVN)\s+(\d+)\s+(\d+)\s+([A-Za-z]+)\b', r'\1 \2:\3/\4', title, flags=re.IGNORECASE)
    title = re.sub(r'\b(QCVN|TCVN)\s+(\d+)\s+(\d+)\b', r'\1 \2:\3', title, flags=re.IGNORECASE)
    return title

def rescue_empty_document(doc_id, url=""):
    """Cứu văn bản rỗng bằng cách điền tiêu đề chuẩn từ danh bạ hoặc URL"""
    doc_id = str(doc_id)
    if doc_id in RESCUED_EMPTY_DOCS:
        return RESCUED_EMPTY_DOCS[doc_id]
    extracted = extract_title_from_url(url)
    return extracted if extracted else f"Văn bản quy phạm pháp luật ID {doc_id}"

def load_corpus(corpus_dir=CORPUS_DIR, force_original=False, clean_noise=True):
    resolved_clean = os.path.join(WORK_DIR, "legal_corpus_resolved_clean.json")
    resolved_path = os.path.join(WORK_DIR, "legal_corpus_resolved.json")

    if clean_noise and os.path.exists(resolved_clean):
        print(f"♻️ Đang nạp Corpus SIÊU CẤP ĐÃ LÀM SẠCH NHIỄU từ '{resolved_clean}'...")
        with open(resolved_clean, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        # Kiểm tra và đảm bảo không có văn bản nào bị rỗng
        rescued_count = 0
        for doc_id, text in corpus.items():
            if not text or not text.strip():
                rescued_text = rescue_empty_document(doc_id)
                corpus[doc_id] = f"{rescued_text} {rescued_text} {rescued_text}\n{rescued_text}"
                rescued_count += 1
        if rescued_count > 0:
            print(f"  🛡️ Đã tự động phục hồi {rescued_count} văn bản bị rỗng trong cache!")
        return corpus

    if not force_original and os.path.exists(resolved_path):
        print(f"♻️ Đang nạp Corpus SIÊU CẤP (Graph Resolved) từ '{resolved_path}'...")
        with open(resolved_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        
        # Tự động phục hồi các văn bản bị rỗng trong corpus resolved
        rescued_count = 0
        for doc_id, text in corpus.items():
            if not text or not text.strip():
                rescued_text = rescue_empty_document(doc_id)
                corpus[doc_id] = f"{rescued_text} {rescued_text} {rescued_text}\n{rescued_text}"
                rescued_count += 1
        if rescued_count > 0:
            print(f"  🛡️ Đã tự động phục hồi {rescued_count} văn bản bị rỗng từ danh bạ/link!")

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
    rescued_count = 0
    for file_path in tqdm(json_files):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            doc_id = str(data["id"])
            title = data.get("name", "")
            passage = data.get("passage", "")
            
            # Cứu văn bản nếu cả title và passage đều rỗng
            if not title and not passage:
                rescued = rescue_empty_document(doc_id, data.get("link", ""))
                title = rescued
                passage = rescued
                rescued_count += 1

            if clean_noise:
                passage = clean_legal_boilerplate(passage)
            
            # Title Boosting 3x: Lặp lại tiêu đề 3 lần để từ khóa số hiệu/tên luật có trọng số vượt trội
            title_boost = f"{title} {title} {title}".strip()
            full_text = f"{title_boost}\n{passage}".strip()
            corpus[doc_id] = full_text

    if rescued_count > 0:
        print(f"🛡️ Đã tự động phục hồi thành công {rescued_count} văn bản rỗng từ metadata/link!")

    print(f"Đã nạp thành công {len(corpus)} văn bản vào Corpus (với Title 3x Boosting & Cleaned).")
    return corpus

# Danh sách các câu hỏi nhiễu / mâu thuẫn mốc thời gian cần loại khỏi tập Train
EXCLUDED_TRAIN_QIDS = {
    "5726": "Năm 2023, người lao động có bao nhiêu ngày nghỉ lễ, tết? (Mâu thuẫn mốc thời gian 2023 vs Bộ luật Lao động 2019)",
}

def load_train_data(train_path=os.path.join(DATA_DIR, "train.json"), filter_noisy=True):
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if filter_noisy:
        original_len = len(data)
        filtered_data = {k: v for k, v in data.items() if str(k) not in EXCLUDED_TRAIN_QIDS}
        removed = original_len - len(filtered_data)
        if removed > 0:
            print(f"🧹 Đã tự động loại bỏ {removed} câu hỏi nhiễu khỏi tập Train (ví dụ: QID {list(EXCLUDED_TRAIN_QIDS.keys())}).")
        return filtered_data
    return data

def load_test_data(test_path=os.path.join(DATA_DIR, "public-official.json")):
    with open(test_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    corpus = load_corpus()
    train_data = load_train_data()
    test_data = load_test_data()

    print(f"Số lượng câu hỏi Train (sau lọc): {len(train_data)}")
    print(f"Số lượng câu hỏi Test: {len(test_data)}")
    
    sample_id = next(iter(corpus))
    print(f"\nMẫu tài liệu [ID: {sample_id}]:")
    print(corpus[sample_id][:300], "...")