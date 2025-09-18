import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler


data_path = 'current.xlsx'
data_df = pd.read_excel(data_path, header=None)

# 滑动窗口大小（50个采样点，对应0.5秒）
window_size = 50

# # 定义低通滤波器
# def low_pass_filter(signal, cutoff=10, fs=100, order=5):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, signal)
#     return y

# 滤波函数
def apply_filter(signal, filter_type="low", cutoff=0.05, order=2):
    b, a = butter(order, cutoff, filter_type)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def extract_time_domain_features(signal):
    # 基本特征
    features = [np.mean(signal), np.std(signal), np.max(signal), np.min(signal)]

    # 滑动窗口特征
    peak_factors, rectified_avgs, waveform_factors, short_time_energies, local_extrema_counts = [], [], [], [], []
    for i in range(0, len(signal) - window_size + 1):
        window = signal[i:i + window_size]
        mean_val = np.mean(window)
        std_val = np.std(window)
        max_val = np.max(window)

        peak_factors.append(max_val / mean_val if mean_val != 0 else 0)
        rectified_avgs.append(np.mean(np.abs(window)))
        waveform_factors.append(mean_val / std_val if std_val != 0 else 0)
        short_time_energies.append(np.sum(np.square(window)))
        local_extrema_counts.append(sum(np.diff(np.sign(np.diff(window))) != 0))

    # 将计算的平均值添加到特征列表(计算评价为了防止数据长度差异过大)
    features.extend(
        [np.mean(peak_factors), np.mean(rectified_avgs), np.mean(waveform_factors), np.mean(short_time_energies),
         np.mean(local_extrema_counts)])

    return features


def extract_frequency_domain_features(signal, ref_signal=None):
    fft_vals = fft(signal)
    fft_magnitudes = np.abs(fft_vals)
    main_freq_magnitude = np.max(fft_magnitudes[:len(fft_magnitudes)//2])  # 主频率成分的幅度
    harmonic_magnitudes = fft_magnitudes[1:6]  # 前5个谐波成分的幅度

    # 频率谱差异（欧氏距离）
    freq_spectrum_diff = 0 #与标签为0的距离
    if ref_signal is not None:
        ref_fft_vals = fft(ref_signal)
        ref_fft_magnitudes = np.abs(ref_fft_vals)
        freq_spectrum_diff = np.linalg.norm(fft_magnitudes - ref_fft_magnitudes)

    return [main_freq_magnitude] + harmonic_magnitudes.tolist() + [freq_spectrum_diff]


# 筛选出标签为0的样本
normal_data = data_df[data_df[0] == 0]

# 计算平均频率谱
fft_magnitudes_list = []
for _, row in normal_data.iterrows():
    signal = np.array(row[1].split(), dtype=float)
    fft_vals = fft(signal)
    fft_magnitudes = np.abs(fft_vals)
    fft_magnitudes_list.append(fft_magnitudes)

# 计算平均值
avg_fft_magnitude = np.mean(fft_magnitudes_list, axis=0)



# 应用特征提取
features = []
ref_signal = avg_fft_magnitude  # 这里定义一个参考信号
for _, row in data_df.iterrows():
    current_data = np.array(row[1].split(), dtype=float)
    # 应用低通滤波器
    filtered_data = apply_filter(current_data)

    time_features = extract_time_domain_features(filtered_data)
    freq_features = extract_frequency_domain_features(filtered_data, ref_signal)
    combined_features = time_features + freq_features
    combined_features.append(row[0])  # 添加标签
    features.append(combined_features)

# 转换为DataFrame
feature_columns = ["mean", "std", "max", "min", "peak_factor", "rectified_avg", "waveform_factor", "short_time_energy", "local_extrema_count", "main_freq", "harmonic_1", "harmonic_2", "harmonic_3", "harmonic_4", "harmonic_5", "freq_spectrum_diff", "Label"]
#对应：峰值因子、整流平均值、波形因子、短时能量、局部极值数量
features_df = pd.DataFrame(features, columns=feature_columns)

# 特征标准化
scaler = StandardScaler()
features_to_scale = features_df.drop("Label", axis=1)
scaled_features = scaler.fit_transform(features_to_scale)

# 创建最终的特征集DataFrame
final_features_df = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
final_features_df["Label"] = features_df["Label"]

# # 保存处理后的特征数据到CSV文件
# output_path = 'new_sp.csv'
# final_features_df.to_csv(output_path, index=False)#
output_path = 'new_sp.xlsx'  # 修改文件扩展名为.xlsx
final_features_df.to_excel(output_path, index=False, engine='openpyxl')

