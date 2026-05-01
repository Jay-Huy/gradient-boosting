# Boosting Project Skeleton

This repository provides a minimal PyTorch project structure with:

- `main.py` as the entrypoint
- YAML-based configuration that decides the run type
- separate orchestration for train and inference, with validation handled inside training by `eval_step`
- base classes for data, model, and score/metric logic
- registry-based factories in `src/utils/factory.py` that map logical component keys to module/class pairs

The implementation is intentionally lightweight so the structure can be refined with you later.

## Project Structure

```text
boosting/
├── configs/                   # Thư mục chứa file YAML cấu hình (hyperparameters, model params, etc.)
│   ├── gpt2_shakespear.yaml
│   └── gpt2_shakespear_boosting.yaml
├── data/                      # Dữ liệu train/val
│   └── shakespeare_char/
├── results/                   # Lưu output models, metrics và checkpoints khi chạy
│   └── runs/
├── src/                       # Source code chính của project
│   ├── data/                  # Các module xử lý data (data loading, preprocessing)
│   ├── models/                # Định nghĩa các model (ví dụ: gpt2, ensemble, etc.)
│   ├── orchestrator/          # Chứa logic loop train/eval, checkpointing, early stopping
│   ├── scores/                # Hàm tính loss và các metrics evaluation
│   └── utils/                 # Các tiện ích: logging, config loader, factory
├── main.py                    # Entrypoint chính để khởi chạy project
├── sample_validation.py (Đang có bug)       # Script hỗ trợ xem kết quả validation (infer từ checkpoint)
└── visualization.py           # Sinh biểu đồ/ảnh trực quan hóa quá trình training
```

## How Gradient Boosting Works Here

Dự án này sử dụng mô phỏng kỹ thuật **Gradient Boosting** trực tiếp trên các Neural Networks (ví dụ điển hình là GPT-2) thay vì decision trees truyền thống. Quy trình hoạt động của việc ensemble được thực hiện dưới cơ chế "tuần tự" (sequential) qua từng **learners**:

1. **Khởi tạo Ensemble:** Mô hình ensemble (`GPT2BoostingLanguageModel`) chứa một chuỗi các "mô hình con" (weak learners) cấu trúc giống nhau nhưng hoạt động tuần tự.
2. **Học phần Residuals (Phần dư):** 
   - Thay vì mọi mô hình đều train lại từ đầu với mục tiêu dự đoán đúng target text tiếp theo, thì ngoại trừ model đầu tiên (Learner 1), các learner đời sau (Learner 2, 3...) sẽ nhận nhiệm vụ hiệu chỉnh lỗi sai của một Cụm ensemble đã train trước đó.
   - Hàm `_negative_ce_gradient` được sử dụng để lấy "Pseudo-Residuals" (cũng chính là negative gradient của hàm Cross-Entropy). Pseudo-Residuals này đại diện cho sai số chưa học được.
   - Khi đó, Learner hiện tại sẽ được thay đổi hàm loss (`_stage_loss`) thành Mean Squared Error (MSE) dựa trên cái Residuals nói trên để triệt tiêu lỗi đi.
3. **Line Search:** Sau khi train xong một Learner, mô hình sẽ thực hiện nội suy một hằng số học `alpha` tối ưu bằng hàm `line_search_active_learner_alpha()`. Điều này quyết định trọng số đóng góp của Learner vừa luyện vào cục Model bự.
4. **Cộng dồn (Inference) #Đang có bug:** Khi chạy thật hoặc test, output cuối (`logits`) sẽ bằng tổng các `logits` mà mỗi learner xả ra nhân với cái trọng số `alpha` của nó. Sau đó chỉ việc `argmax` là ra từ tiếp theo.

---

## Quickstart

```
python main.py --config configs/gpt2_shakespear_boosting_checkpointed.yaml
```
