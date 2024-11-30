import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # 追加: numpyのインポート

# サンプリング周波数と波形の生成時間を設定
sampling_rate = 500  # Hz
duration = 10  # 秒

# 1. 正常なECG波形を生成
normal_ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate)

# 2. 異常波形1: Bradycardia（徐脈）
bradycardia_ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=40)

# 3. 異常波形2: Tachycardia（頻脈）
tachycardia_ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=120)

# 4. 異常波形3: Noise（ノイズの多い波形）
noisy_ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, noise=0.5)

# 5. 異常波形4: Arrhythmia（不整脈）
arrhythmia_ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, method="simple", heart_rate=80, randomness=0.5)

# 各波形をリストにまとめる
ecg_signals = {
    "Normal ECG": normal_ecg,
    "Bradycardia ECG": bradycardia_ecg,
    "Tachycardia ECG": tachycardia_ecg,
    "Noisy ECG": noisy_ecg,
    "Arrhythmia ECG": arrhythmia_ecg
}

# 分析結果を格納する辞書
analysis_results = {}

# サブプロットの総数を計算（各信号に対して3つのサブプロット）
total_subplots = len(ecg_signals) * 3

# プロットの設定
plt.figure(figsize=(15, 5 * len(ecg_signals)))  # 各信号ごとに高さを調整
subplot_index = 1

for label, ecg in ecg_signals.items():
    # 1. ECG信号のプロット
    plt.subplot(total_subplots, 1, subplot_index)
    plt.plot(ecg, label=label)
    plt.title(label)
    plt.legend()
    subplot_index += 1
    
    # ECGの前処理（フィルタリングなど）
    signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate)
    
    # R-ピークの検出
    r_peaks = info["ECG_R_Peaks"]
    
    # RR間隔の計算（秒単位）
    rr_intervals = pd.Series(np.diff(r_peaks) / sampling_rate)
    
    # 心拍数の計算（bpm）
    if len(rr_intervals) > 0:
        heart_rate = 60 / rr_intervals.mean()
    else:
        heart_rate = np.nan  # データが不足している場合
    
    # HRVの計算（SDNN）
    if len(rr_intervals) > 1:
        hr_variability = rr_intervals.std()
    else:
        hr_variability = np.nan  # データが不足している場合
    
    # 解析結果を辞書に保存
    analysis_results[label] = {
        "R_Peaks": r_peaks,
        "Average_HR_bpm": heart_rate,
        "HRV_SDNN_sec": hr_variability,
        "RR_Intervals_sec": rr_intervals
    }
    
    # 2. R-ピークをプロット
    plt.subplot(total_subplots, 1, subplot_index)
    plt.plot(ecg, label=label)
    plt.plot(r_peaks, ecg[r_peaks], "ro", label="R-peaks")
    plt.title(f"{label} with R-peaks")
    plt.legend()
    subplot_index += 1
    
    # 3. RR間隔のプロット
    plt.subplot(total_subplots, 1, subplot_index)
    plt.plot(rr_intervals, marker='o')
    plt.title(f"{label} RR Intervals")
    plt.xlabel("Beat Number")
    plt.ylabel("RR Interval (s)")
    subplot_index += 1

# レイアウト調整
plt.tight_layout()
plt.show()

# 解析結果の表示
for label, metrics in analysis_results.items():
    print(f"--- {label} ---")
    if not np.isnan(metrics['Average_HR_bpm']):
        print(f"平均心拍数: {metrics['Average_HR_bpm']:.2f} bpm")
    else:
        print("平均心拍数: 計算不可")
    
    if not np.isnan(metrics['HRV_SDNN_sec']):
        print(f"HRV (SDNN): {metrics['HRV_SDNN_sec']:.4f} s")
    else:
        print("HRV (SDNN): 計算不可")
    
    print(f"RR間隔のサンプル数: {len(metrics['RR_Intervals_sec'])}")
    print()

# 必要に応じて詳細な解析結果をCSVに保存
# for label, metrics in analysis_results.items():
#     df = pd.DataFrame({
#         "RR_Interval_sec": metrics["RR_Intervals_sec"]
#     })
#     df.to_csv(f"{label}_RR_Intervals.csv", index=False)

