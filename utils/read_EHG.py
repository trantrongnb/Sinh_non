import numpy as np
import pandas as pd
from utils.read_EMR import read_EMR
from scipy.io import loadmat
from scipy.signal import butter,filtfilt

import numpy as np
from scipy.signal import stft
from scipy.signal.windows import hamming
import os
from scipy.signal import stft, windows

def read_EHG(folder_path):

    file_EHG=[]
    file_EMR=[]

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".hea"):
            file_EMR.append(file)
        else:
            file_EHG.append(file)
    
    all_EHG_data=[]
    labels=[]

    all_Labels,all_EMR_data,all_Name_files=read_EMR(folder_path)

    fs=20
    t0=100
    selected_channels=(0,2,4)

    for file_hea,file_mat in zip(file_EMR,file_EHG):
        if file_hea in all_Name_files:
            file_mat=os.path.join("dataset",file_mat)
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

    #-----------------------------------------------------#
            # 1. Cắt tín hiệu về đúng 30 phút
            max_time = 30 * 60  # 30 phút = 1800 giây
            if t_downsample[-1] > max_time:
                idx = np.argmin(np.abs(t_downsample - max_time))
                t_downsample = t_downsample[:idx+1]
                # MATLAB chọn kênh số 3 => Python là chỉ số 2
                downsample_signal = downsample_signal[:idx+1, 2]
            else:
                downsample_signal = downsample_signal[:, 2]


            #2. Tính tần số lấy mẫu
            fs = 1.0 / (t_downsample[1] - t_downsample[0])


            #3. Tính chiều dài cửa sổ
            f0 = 0.1
            window_length = round(fs / f0 * 6)
            window = windows.hamming(window_length)


            #4. Tính STFT
            f, t_spec, Zxx = stft(
                downsample_signal,
                fs=fs,
                window=window,
                nperseg=window_length,
                noverlap=window_length//2,
                boundary=None,
                padded=False
            )

            # 5. Biến về log(abs)
            s = np.log(np.abs(Zxx) + 1e-8)   # tránh log(0)

            # 6. Zero-pad theo thời gian (cột)
            target_cols = 106
            if s.shape[1] < target_cols:
                pad_cols = target_cols - s.shape[1]
                s = np.hstack([s, np.zeros((s.shape[0], pad_cols))])

            #7. Lấy nửa phổ dương
            mid = s.shape[0] // 2
            s_half = s[mid:, :target_cols]
            all_EHG_data.append(s_half)
    return all_Labels,all_EMR_data,all_EHG_data