Hướng Dẫn Đọc Dữ Liệu
1. Bài Toán EMR (Metadata Classification hoặc Metadata Regression)
folder_path = "dataset"
all_labels, all_EMR_data, all_name_files = read_EMR(folder_path)

Mô tả:

all_labels: List chứa các nhãn [0 hoặc 1, số ngày sinh].
all_EMR_data: Mảng dictionary chứa các cột cần thiết cho huấn luyện.
all_name_files: Tên các file thỏa mãn điều kiện trên 37 tuần tuổi (không cần quan tâm).

2. Bài Toán EHG và Bài Toán Kết Hợp EHG và EMR
folder_path = "Dataset"
all_labels, all_EMR_data, all_EHG_datas = read_EHG(folder_path)

Mô tả:

all_labels: List chứa các nhãn [0 hoặc 1, số ngày sinh].
all_EMR_data: Mảng dictionary chứa các cột cần thiết cho huấn luyện.
all_EHG_datas: Mảng list chứa dữ liệu tần số đã được chuẩn hóa.
