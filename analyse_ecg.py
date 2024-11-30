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

# 各信号に対して別々の図を作成
for label, ecg in ecg_signals.items():
    # 図の設定
    plt.figure(figsize=(15, 20))
    
    # 1. ECG信号のプロット
    plt.subplot(6, 1, 1)
    plt.plot(ecg, label=label)
    plt.title(f"{label} - Raw ECG Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    
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
    
    # ECG波形の詳細な特徴を抽出（P波、QRS複合体、T波）
    delineate = nk.ecg_delineate(ecg, r_peaks, sampling_rate=sampling_rate, method="dwt")
    
    # 戻り値が辞書かタプルかを確認
    if isinstance(delineate, dict):
        # 最新バージョン（辞書）の場合
        p_onsets = delineate.get("ECG_P_Onsets", [])
        p_offsets = delineate.get("ECG_P_Offsets", [])
        qrs_onsets = delineate.get("ECG_QRS_Onsets", [])
        qrs_offsets = delineate.get("ECG_QRS_Offsets", [])
        t_onsets = delineate.get("ECG_T_Onsets", [])
        t_offsets = delineate.get("ECG_T_Offsets", [])
    elif isinstance(delineate, tuple):
        # 古いバージョン（タプル）の場合
        # タプルの内容を確認
        print(f"戻り値の型がタプルです。内容を確認します: {delineate}")
        # 以下は仮のインデックス。実際の内容に応じて調整してください。
        # 例: (P_Onsets, P_Offsets, QRS_Onsets, QRS_Offsets, T_Onsets, T_Offsets)
        if len(delineate) >= 6:
            p_onsets = delineate[0]
            p_offsets = delineate[1]
            qrs_onsets = delineate[2]
            qrs_offsets = delineate[3]
            t_onsets = delineate[4]
            t_offsets = delineate[5]
        else:
            p_onsets, p_offsets, qrs_onsets, qrs_offsets, t_onsets, t_offsets = [], [], [], [], [], []
    else:
        # その他の戻り値形式の場合
        p_onsets, p_offsets, qrs_onsets, qrs_offsets, t_onsets, t_offsets = [], [], [], [], [], []
    
    # QRS幅の計算
    qrs_durations = (np.array(qrs_offsets) - np.array(qrs_onsets)) / sampling_rate  # 秒単位
    
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
    
    # P波の振幅と持続時間の計算
    p_durations = (np.array(p_offsets) - np.array(p_onsets)) / sampling_rate  # 秒単位
    
    p_amplitudes = []
    for onset, offset in zip(p_onsets, p_offsets):
        if np.isnan(onset) or np.isnan(offset):
            p_amplitudes.append(np.nan)
            continue
        try:
            p_wave = ecg[int(onset):int(offset)]
            if len(p_wave) == 0:
                p_amplitudes.append(np.nan)
                continue
            p_amplitudes.append(np.max(p_wave) - np.min(p_wave))
        except (IndexError, TypeError):
            p_amplitudes.append(np.nan)
    
    # PR間隔の計算（P波開始からRピークまで）
    pr_intervals = []
    for p_onset, r_peak in zip(p_onsets, r_peaks):
        if np.isnan(p_onset):
            pr_intervals.append(np.nan)
            continue
        pr_interval = (r_peak - p_onset) / sampling_rate
        pr_intervals.append(pr_interval)
    
    pr_intervals = pd.Series(pr_intervals)
    
    # QT間隔の計算（QRS開始からT波終了まで）
    qt_intervals = []
    for q_onset, t_offset in zip(qrs_onsets, t_offsets):
        if np.isnan(q_onset) or np.isnan(t_offset):
            qt_intervals.append(np.nan)
            continue
        qt_interval = (t_offset - q_onset) / sampling_rate
        qt_intervals.append(qt_interval)
    
    qt_intervals = pd.Series(qt_intervals)
    
    # 解析結果を辞書に保存
    analysis_results[label] = {
        "R_Peaks": r_peaks,
        "Average_HR_bpm": heart_rate,
        "HRV_SDNN_sec": hr_variability,
        "RR_Intervals_sec": rr_intervals,
        "P_Durations_sec": pd.Series(p_durations),
        "P_Amplitudes_mV": pd.Series(p_amplitudes),
        "QRS_Durations_sec": pd.Series(qrs_durations),
        "T_Durations_sec": pd.Series(t_durations),
        "T_Amplitudes_mV": pd.Series(t_amplitudes),
        "PR_Intervals_sec": pr_intervals,
        "QT_Intervals_sec": qt_intervals
    }
    
    # 2. R-ピークをプロット
    plt.subplot(6, 1, 2)
    plt.plot(ecg, label=label)
    plt.plot(r_peaks, ecg[r_peaks], "ro", label="R-peaks")
    plt.title(f"{label} - R-peaks")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    
    # 3. RR間隔のプロット
    plt.subplot(6, 1, 3)
    plt.plot(rr_intervals, marker='o')
    plt.title(f"{label} - RR Intervals")
    plt.xlabel("Beat Number")
    plt.ylabel("RR Interval (s)")
    
    # 4. P波の特徴をプロット
    plt.subplot(6, 1, 4)
    plt.plot(ecg, label=label)
    plt.title(f"{label} - P Wave")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    for p_onset, p_offset in zip(p_onsets, p_offsets):
        if np.isnan(p_onset) or np.isnan(p_offset):
            continue
        plt.axvspan(p_onset, p_offset, color='blue', alpha=0.3, label="P Wave")
    plt.legend()
    
    # 5. QRS複合体の特徴をプロット
    plt.subplot(6, 1, 5)
    plt.plot(ecg, label=label)
    plt.title(f"{label} - QRS Complex")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    for q_onset, q_offset in zip(qrs_onsets, qrs_offsets):
        if np.isnan(q_onset) or np.isnan(q_offset):
            continue
        plt.axvspan(q_onset, q_offset, color='green', alpha=0.3, label="QRS Complex")
    plt.legend()
    
    # 6. T波の特徴をプロット
    plt.subplot(6, 1, 6)
    plt.plot(ecg, label=label)
    plt.title(f"{label} - T Wave")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    for t_onset, t_offset in zip(t_onsets, t_offsets):
        if np.isnan(t_onset) or np.isnan(t_offset):
            continue
        plt.axvspan(t_onset, t_offset, color='orange', alpha=0.3, label="T Wave")
    plt.legend()
    
    # レイアウト調整
    plt.tight_layout()
    plt.show()
    
    # 解析結果の表示
    print(f"--- {label} ---")
    if not np.isnan(heart_rate):
        print(f"平均心拍数: {heart_rate:.2f} bpm")
    else:
        print("平均心拍数: 計算不可")
    
    if not np.isnan(hr_variability):
        print(f"HRV (SDNN): {hr_variability:.4f} s")
    else:
        print("HRV (SDNN): 計算不可")
    
    print(f"RR間隔のサンプル数: {len(rr_intervals)}")
    if len(rr_intervals) > 0:
        print(f"平均RR間隔: {rr_intervals.mean():.4f} s")
    else:
        print("平均RR間隔: 計算不可")
    
    if len(p_durations) > 0:
        print(f"平均P波持続時間: {p_durations.mean():.4f} s")
    else:
        print("平均P波持続時間: 計算不可")
    
    if len(p_amplitudes) > 0:
        print(f"平均P波振幅: {p_amplitudes.mean():.4f} mV")
    else:
        print("平均P波振幅: 計算不可")
    
    if len(qrs_durations) > 0:
        print(f"平均QRS幅: {qrs_durations.mean():.4f} s")
    else:
        print("平均QRS幅: 計算不可")
    
    if len(t_durations) > 0:
        print(f"平均T波持続時間: {t_durations.mean():.4f} s")
    else:
        print("平均T波持続時間: 計算不可")
    
    if len(t_amplitudes) > 0:
        print(f"平均T波振幅: {np.nanmean(t_amplitudes):.4f} mV")
    else:
        print("平均T波振幅: 計算不可")
    
    if len(pr_intervals) > 0:
        print(f"平均PR間隔: {pr_intervals.mean():.4f} s")
    else:
        print("平均PR間隔: 計算不可")
    
    if len(qt_intervals) > 0:
        print(f"平均QT間隔: {qt_intervals.mean():.4f} s")
    else:
        print("平均QT間隔: 計算不可")
    
    print("\n")
    
    # 必要に応じて詳細な解析結果をCSVに保存
    # df = pd.DataFrame({
    #     "RR_Interval_sec": analysis_results[label]["RR_Intervals_sec"],
    #     "P_Duration_sec": analysis_results[label]["P_Durations_sec"],
    #     "P_Amplitude_mV": analysis_results[label]["P_Amplitudes_mV"],
    #     "QRS_Duration_sec": analysis_results[label]["QRS_Durations_sec"],
    #     "T_Duration_sec": analysis_results[label]["T_Durations_sec"],
    #     "T_Amplitude_mV": analysis_results[label]["T_Amplitudes_mV"],
    #     "PR_Interval_sec": analysis_results[label]["PR_Intervals_sec"],
    #     "QT_Interval_sec": analysis_results[label]["QT_Intervals_sec"]
    # })
    # df.to_csv(f"{label}_ECG_Features.csv", index=False)

