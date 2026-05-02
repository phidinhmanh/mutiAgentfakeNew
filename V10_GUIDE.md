# Cách chạy TRUST Agent V10 để đạt hiệu suất tối ưu

Hệ thống TRUST Agent V10 đã được tối ưu hóa mạnh mẽ về cả độ chính xác (Accuracy) và tốc độ (Latency). Dưới đây là các hướng dẫn quan trọng để vận hành bộ Kit này.

## 1. Các cải tiến chính trong V10
- **Zero Tolerance cho Số liệu**: Mọi con số (số tiền, ngày tháng, tỷ lệ) phải khớp chính xác đến từng đơn vị. Chỉ cần một sai lệch nhỏ sẽ được đánh FAKE.
- **Soft REAL Strategy**: Tin thật chỉ cần đúng chủ đề và sự kiện chính, không bắt lỗi "thiếu thông tin phụ".
- **Semantic Cache**: Tự động nhận diện các thực thể (EXO, Mbappe, Thủ tướng...) để tái sử dụng kết quả kiểm chứng, giúp Latency giảm xuống < 10s cho các claim lặp lại.
- **Async Batching**: Xử lý song song 5-10 mẫu cùng lúc.

## 2. Cách chạy Benchmark
Sử dụng script `benchmark_v8.py` (đã được nâng cấp lên logic V10):

```bash
# Chạy 20 mẫu tinh hoa với batch size 5
python scripts/benchmark_v8.py --batch-size 5 --limit 20 --output benchmark_v10_report.json
```

## 3. Cấu hình phần cứng & API
- **LLM**: Khuyến nghị sử dụng **Google Gemini** (Gemini 1.5 Pro) để có khả năng đọc hiểu tiếng Việt và trích xuất số liệu tốt nhất.
- **Cache**: Đảm bảo thư mục `retrieval_index` và bộ nhớ RAM đủ trống để duy trì Semantic Cache trong suốt phiên làm việc.
- **Network**: DuckDuckGo Scraper yêu cầu kết nối ổn định để tránh bị timeout khi lấy nội dung từ các trang báo lớn (VnExpress, Tuổi Trẻ).

## 4. Giải thích tệp Benchmark
File `standard_benchmark_v4.json` chứa:
- So sánh hiệu năng giữa V8 (30% Accuracy) và V10 (60% Accuracy).
- Chi tiết từng mẫu thử và lý do (Reasoning) cho kết quả V10.

---
*Bộ Kit này đã sẵn sàng để gửi đi.*
