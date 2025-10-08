Hướng dẫn đọc dữ liệu
1. Bài toán EMR (metadata_classification hoặc metadata_regression)
folder_path = "dataset"
all_labels, all_EMR_data, all_name_files = read_EMR(folder_path)


all_labels: List chứa các nhãn [0 hoặc 1, số ngày sinh].
all_EMR_data: Mảng dictionary chứa các cột cần thiết cho huấn luyện.
all_name_files: Tên các file thỏa mãn điều kiện trên 37 tuần tuổi (không cần quan tâm).

2. Bài toán EHG và bài toán kết hợp EHG và EMR
folder_path = "Dataset"
all_labels, all_EMR_data, all_EHG_datas = read_EHG(folder_path)


all_labels: List chứa các nhãn [0 hoặc 1, số ngày sinh].
all_EMR_data: Mảng dictionary chứa các cột cần thiết cho huấn luyện.
all_EHG_datas: Mảng list chứa dữ liệu tần số đã được chuẩn hóa.
