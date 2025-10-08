# 📄 Hướng dẫn đọc dữ liệu EMR & EHG

Tài liệu này mô tả cách đọc dữ liệu cho các bài toán **metadata_classification**, **metadata_regression**, **EHG**, và **kết hợp EHG + EMR**.

---

## 🧠 1. Bài toán EMR (metadata_classification hoặc metadata_regression)

**Ví dụ:**
```python
folder_path = "dataset"
all_Labels, all_EMR_data, all_Name_files = read_EMR(folder_path)
