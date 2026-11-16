# Báo cáo Day 7 — Retrieval Context Engineering

## Dữ liệu và benchmark

- Corpus: export Day 6 gồm 32 chunk từ `94/2026/NĐ-CP`, trang 1–32.
- Schema đầu vào: `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata`; adapter nằm trong `starter/convert_day6.py`.
- Câu hỏi gốc: 15 dev và 5 test có evidence. Bổ sung `data/day7_abstention_questions.json` gồm 2 dev và 2 test thiếu evidence.
- Gold/evidence chỉ được dùng bởi evaluator để chấm; không được đưa vào query, retriever hoặc agent.
- Nguồn dữ liệu là file Day 6 đã có trong repo; không gửi dữ liệu hay secret ra dịch vụ ngoài.

## Retrieval

| Cấu hình | Phương pháp | Cấu hình | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate |
|---|---|---|---|---:|---:|---:|
| Baseline | BM25 | top-k=3, budget=6000 ký tự | dev | 13/17 | 14/17 | 13/17 |
| Dense | hashed dense vector | 384 chiều, SHA-256 sign-v1 | dev | 10/17 | 14/17 | 13/17 |
| Chọn | Hybrid RRF | BM25 + dense, RRF k=60 | dev | 14/17 | 14/17 | 14/17 |
| Baseline | BM25 | top-k=3, budget=6000 ký tự | test | 1/7 | 3/7 | 2/7 |
| Dense | hashed dense vector | 384 chiều, SHA-256 sign-v1 | test | 1/7 | 3/7 | 2/7 |
| Chọn | Hybrid RRF | BM25 + dense, RRF k=60 | test | 1/7 | 3/7 | 3/7 |

Thời gian median trên dev lần lượt khoảng 2.08 ms, 2.41 ms và 3.46 ms;
trên test khoảng 4.88 ms, 1.60 ms và 2.98 ms. Chi phí dịch vụ local là 0.
Trace và summary được sinh trong `results/dev` và `results/test` bằng các lệnh
trong [IMPLEMENTATION.md](../../IMPLEMENTATION.md).

## Agent và context

`run_agent` giới hạn số lượt search, `top_k` và context budget; deduplicate theo
chunk ID, giữ source/page trong context, và chỉ chấp nhận citation vừa có trong
context vừa xuất hiện trong câu trả lời. Lỗi provider/retriever được để lộ để
trace, không fallback sang kết quả giả. `OpenAIController` dùng JSON response
với hai action `search` và `answer`; API key chỉ đọc từ biến môi trường.

## Hai lỗi dev và cải tiến

1. TF-IDF đơn giản dễ bỏ sót cách diễn đạt khác nhau hoặc số liệu ngắn. BM25
   giữ xử lý lexical nhưng dùng chuẩn hóa độ dài và IDF tốt hơn.
2. Dense đơn lẻ có thể ưu tiên tương đồng hình thức và bỏ sót thuật ngữ pháp
   lý chính xác. Hybrid RRF được chọn để kết hợp tín hiệu lexical và dense;
   context budget vẫn được cố định để so sánh công bằng.

## Giới hạn

Hashed dense index là adapter offline xác định, không phải embedding model
ngữ nghĩa đã huấn luyện. Khi có vector store Day 6 thật, thay bằng adapter
`Retriever` tương thích và ghi model/version/dimension vào trace. Các metric
model answer, citation support và abstention cần người chấm đọc trace; code
không suy luận grounding chỉ từ citation ID.
