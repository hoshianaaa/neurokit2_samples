import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# 各波形を辞書にまとめる
ecg_signals = {
    "Normal ECG": normal_ecg,
    "Bradycardia ECG": bradycardia_ecg,
    "Tachycardia ECG": tachycardia_ecg,
    "Noisy ECG": noisy_ecg,
    "Arrhythmia ECG": arrhythmia_ecg
}

# 解析結果を格納する辞書
analysis_results = {}

# サブプロットの総数を計算（各信号に対して4つのサブプロット：波形、Rピーク付き波形、RR間隔、波形特徴）
total_subplots = len(ecg_signals) * 4

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
    
    # ECGの前処理とRピークの検出
    signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate)
    
    # R-ピークの取得
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
    
    # ECG波形の詳細な特徴を抽出（QRS幅、T波特徴など）
    delineate = nk.ecg_delineate(ecg, r_peaks, sampling_rate=sampling_rate, method="dwt")
    
    # 戻り値が辞書かタプルかを確認
    if isinstance(delineate, dict):
        # 最新バージョン（辞書）の場合
        qrs_start = delineate.get("ECG_QRS_Onsets", [])
        qrs_end = delineate.get("ECG_QRS_Offsets", [])
        t_onsets = delineate.get("ECG_T_Onsets", [])
        t_offsets = delineate.get("ECG_T_Offsets", [])
    elif isinstance(delineate, tuple):
        # 古いバージョン（タプル）の場合
        # タプルの内容を確認
        print(f"戻り値の型がタプルです。内容を確認します: {delineate}")
        # 以下は仮のインデックス。実際の内容に応じて調整してください。
        # 例: (P_Onsets, QRS_Onsets, QRS_Offsets, T_Onsets, T_Offsets)
        if len(delineate) >= 5:
            qrs_start = delineate[1]
            qrs_end = delineate[2]
            t_onsets = delineate[3]
            t_offsets = delineate[4]
        else:
            qrs_start, qrs_end, t_onsets, t_offsets = [], [], [], []
    else:
        # その他の戻り値形式の場合
        qrs_start, qrs_end, t_onsets, t_offsets = [], [], [], []
    
    # QRS幅の計算
    qrs_durations = (np.array(qrs_end) - np.array(qrs_start)) / sampling_rate  # 秒単位
    
    # T波の振幅と持続時間の計算
    t_durations = (np.array(t_offsets) - np.array(t_onsets)) / sampling_rate  # 秒単位
    
    t_amplitudes = []
    for onset, offset in zip(t_onsets, t_offsets):
        if np.isnan(onset) or np.isnan(offset):
            t_amplitudes.append(np.nan)
            continue
        try:
            t_wave = ecg[int(onset):int(offset)]
            if len(t_wave) == 0:
                t_amplitudes.append(np.nan)
                continue
            t_amplitudes.append(np.max(t_wave) - np.min(t_wave))
        except (IndexError, TypeError):
            t_amplitudes.append(np.nan)
    
    # 解析結果を辞書に保存
    analysis_results[label] = {
        "R_Peaks": r_peaks,
        "Average_HR_bpm": heart_rate,
        "HRV_SDNN_sec": hr_variability,
        "RR_Intervals_sec": rr_intervals,
        "QRS_Durations_sec": pd.Series(qrs_durations),
        "T_Durations_sec": pd.Series(t_durations),
        "T_Amplitudes": pd.Series(t_amplitudes)
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
    
    # 4. 波形特徴のプロット（QRS幅、T波振幅など）
    plt.subplot(total_subplots, 1, subplot_index)
    plt.plot(ecg, label=label)
    plt.title(f"{label} Waveform Features")
    
    # QRS幅を色付きで表示（凡例を一度だけ表示するためのフラグ）
    qrs_plotted = False
    for q_start, q_end in zip(qrs_start, qrs_end):
        if np.isnan(q_start) or np.isnan(q_end):
            continue
        plt.axvspan(q_start, q_end, color='green', alpha=0.3, label="QRS Width" if not qrs_plotted else "")
        qrs_plotted = True
    
    # T波を色付きで表示（凡例を一度だけ表示するためのフラグ）
    t_plotted = False
    for t_onset, t_offset in zip(t_onsets, t_offsets):
        if np.isnan(t_onset) or np.isnan(t_offset):
            continue
        plt.axvspan(t_onset, t_offset, color='orange', alpha=0.3, label="T Wave" if not t_plotted else "")
        t_plotted = True
    
    plt.legend()
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
    if len(metrics['QRS_Durations_sec']) > 0:
        print(f"平均QRS幅: {metrics['QRS_Durations_sec'].mean():.4f} s")
    else:
        print("平均QRS幅: 計算不可")
    
    if len(metrics['T_Durations_sec']) > 0:
        print(f"平均T波持続時間: {metrics['T_Durations_sec'].mean():.4f} s")
    else:
        print("平均T波持続時間: 計算不可")
    
    if len(metrics['T_Amplitudes']) > 0:
        print(f"平均T波振幅: {metrics['T_Amplitudes'].mean():.4f} mV")
    else:
        print("平均T波振幅: 計算不可")
    
    print()

# 必要に応じて詳細な解析結果をCSVに保存
# for label, metrics in analysis_results.items():
#     df = pd.DataFrame({
#         "RR_Interval_sec": metrics["RR_Intervals_sec"],
#         "QRS_Duration_sec": metrics["QRS_Durations_sec"],
#         "T_Duration_sec": metrics["T_Durations_sec"],
#         "T_Amplitude_mV": metrics["T_Amplitudes"]
#     })
#     df.to_csv(f"{label}_ECG_Features.csv", index=False)

