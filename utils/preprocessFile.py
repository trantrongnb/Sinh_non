import numpy as np
import pandas as pd
from read_data import read_data
from scipy.io import loadmat
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

import numpy as np
from scipy.signal import stft
from scipy.signal.windows import hamming

def preprocessFile(file_hea,file_mat):
    # file_hea="dataset/tpehg1039m.hea"
    # file_mat='dataset/tpehg1039m.mat'
    fs=20
    t0=100
    selected_channels=(0,2,4)
    metadata,label=read_data(file_hea)
    if metadata is None and label is None:
        return None,None,None,None
    # --- 1. load .mat và chuẩn bị ---
    mat = loadmat(file_mat)
    if 'val' not in mat:
        raise ValueError("File .mat không chứa biến 'val'")
    val = mat['val']  # shape (n_channels, n_samples) theo PhysioNet
    n_channels, n_samples = val.shape

    # --- 2. xác định chỉ số bắt đầu (bỏ t0 giây đầu) ---
    start_idx = int(t0 * fs)            # giống MATLAB: t0*fs
    if start_idx >= n_samples:
        raise ValueError("t0 quá lớn (bỏ hết dữ liệu). Kiểm tra fs/t0/file.")
    # chọn kênh (MATLAB dùng [1 3 5] => Python index 0,2,4)
    sig = val[list(selected_channels), start_idx:]   # shape (len(selected), N')
    # tạo trục thời gian tương ứng với phần đã cắt
    t = np.arange(start_idx, n_samples) / fs        # bắt đầu = start_idx/fs == t0

    # --- 3. chuyển sang shape (n_samples, n_channels) để filtfilt theo trục thời gian ---
    ehg = sig.T.astype(np.float64)   # shape (N', n_ch). Dùng float để filtfilt ổn định.

    # --- 4. thiết kế và áp dụng low-pass Butterworth bậc 4 (fc = 4 Hz) ---
    high_frequency_cutoff = 4.0
    fc_low = high_frequency_cutoff / (fs / 2.0)   # normalized (0..1), fs/2 = Nyquist
    b_low, a_low = butter(N=4, Wn=fc_low, btype='low', analog=False)
    signal_lp = filtfilt(b_low, a_low, ehg, axis=0)   # lọc theo trục thời gian

    # --- 5. thiết kế và áp dụng high-pass Butterworth bậc 4 (fc = 0.05 Hz) ---
    fc_high = 0.05 / (fs / 2.0)
    b_high, a_high = butter(N=4, Wn=fc_high, btype='high', analog=False)
    signal_filtered = filtfilt(b_high, a_high, signal_lp, axis=0)

    # --- 6. downsample bằng lấy mẫu cách quãng (tương tự MATLAB downsample, không thêm anti-alias) ---
    f_ds = high_frequency_cutoff * 2.4
    k_down = int(round(fs / f_ds))
    if k_down < 1:
        k_down = 1
    downsample_signal = signal_filtered[::k_down, :]   # (N'_down, n_ch)
    t_downsample = t[::k_down]                         # thời gian tương ứng
    return metadata,label,downsample_signal, t_downsample

a = preprocessFile("dataset/tpehg1022m.hea","dataset/tpehg1022m.mat")
print(a)