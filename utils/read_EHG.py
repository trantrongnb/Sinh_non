import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, stft, windows
from utils.read_EMR import read_EMR

def read_EHG(folder_path, target_cols=106, eps=1e-8):
    file_EHG = []
    file_EMR = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".hea"):
            file_EMR.append(file)
        elif file.endswith(".mat"):
            file_EHG.append(file)

    all_EHG_data = []
    labels = []

    all_Labels, all_EMR_data, all_Name_files = read_EMR(folder_path)

    fs = 20        # original sampling freq
    t0 = 100       # discard first 100s
    selected_channels = (0, 2, 4)

    for file_hea, file_mat in zip(file_EMR, file_EHG):
        if file_hea in all_Name_files:
            file_mat = os.path.join("dataset", file_mat)
            mat = loadmat(file_mat)

            if "val" not in mat:
                raise ValueError("File .mat không chứa biến 'val'")

            val = mat["val"]
            n_channels, n_samples = val.shape

            start_idx = int(t0 * fs)
            if start_idx >= n_samples:
                raise ValueError("t0 quá lớn (bỏ hết dữ liệu)")

            sig = val[list(selected_channels), start_idx:]
            t = np.arange(start_idx, n_samples) / fs
            ehg = sig.T.astype(np.float64)

            # low-pass 4 Hz
            high_frequency_cutoff = 4.0
            fc_low = high_frequency_cutoff / (fs / 2.0)
            b_low, a_low = butter(N=4, Wn=fc_low, btype="low", analog=False)
            signal_lp = filtfilt(b_low, a_low, ehg, axis=0, padtype="even")

            # high-pass 0.05 Hz
            fc_high = 0.05 / (fs / 2.0)
            b_high, a_high = butter(N=4, Wn=fc_high, btype="high", analog=False)
            signal_filtered = filtfilt(b_high, a_high, signal_lp, axis=0, padtype="even")

            # downsample by simple decimation (as original)
            f_ds = high_frequency_cutoff * 2.4
            k_down = int(round(fs / f_ds))
            if k_down < 1:
                k_down = 1
            downsample_signal = signal_filtered[::k_down, :]
            t_downsample = t[::k_down]

            # cut to 30 minutes and select channel 3 (index 2)
            max_time = 30 * 60
            if t_downsample[-1] > max_time:
                idx = np.argmin(np.abs(t_downsample - max_time))
                t_downsample = t_downsample[: idx + 1]
                downsample_signal = downsample_signal[: idx + 1, 2]
            else:
                downsample_signal = downsample_signal[:, 2]

            # new sampling rate after downsampling
            fs_down = 1.0 / (t_downsample[1] - t_downsample[0])

            # STFT params
            f0 = 0.1
            window_length = int(round(fs_down / f0 * 6))
            if window_length < 1:
                window_length = 1
            window = windows.hamming(window_length, sym=True)

            f, t_spec, Zxx = stft(
                downsample_signal,
                fs=fs_down,
                window=window,
                nperseg=window_length,
                noverlap=window_length // 2,
                boundary=None,
                padded=False,
            )

            # --- IMPORTANT: pad magnitude BEFORE taking log ---
            mag = np.abs(Zxx)  # shape (n_freq, n_time)
            if mag.shape[1] < target_cols:
                pad_cols = target_cols - mag.shape[1]
                mag = np.hstack([mag, np.full((mag.shape[0], pad_cols), eps)])
            # if longer, crop to target_cols
            if mag.shape[1] > target_cols:
                mag = mag[:, :target_cols]

            # log magnitude (safe)
            s = np.log(mag + 0.0)  # mag already >= eps

            # take positive-frequency half (match MATLAB)
            mid = s.shape[0] // 2
            s_half = s[:mid, :target_cols]

            # optional: debug prints (uncomment if needed)
            # print("STFT shape:", Zxx.shape, "mag shape:", mag.shape, "s_half shape:", s_half.shape)
            # print("s_half min/max:", s_half.min(), s_half.max())
            all_EHG_data.append(s_half)

    data = loadmat('s_half_data.mat')
    s_half_data = data['s_half_data']

    #print("Trước khi chỉnh:", s_half_data.shape)  # (301, 106, 159)

    # Đưa về đúng thứ tự như MATLAB: (159, 106, 301)
    s_half_data = np.transpose(s_half_data, (2, 1, 0))
    #print("Sau khi khôi phục:", s_half_data.shape)

    # Bây giờ đổi trục sang (159, 301, 106)
    s_half_data = np.transpose(s_half_data, (0, 2, 1))
    #print("Sau khi reshape cuối cùng:", s_half_data.shape)  # (159, 301, 106)
    all_EHG_data=s_half_data
    return all_Labels, all_EMR_data, all_EHG_data
