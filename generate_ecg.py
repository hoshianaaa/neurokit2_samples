import neurokit2 as nk
import matplotlib.pyplot as plt

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

# 6. プロットで比較
plt.figure(figsize=(12, 10))

# 正常波形
plt.subplot(5, 1, 1)
plt.plot(normal_ecg, label="Normal ECG")
plt.title("Normal ECG")
plt.legend()

# 徐脈波形
plt.subplot(5, 1, 2)
plt.plot(bradycardia_ecg, label="Bradycardia ECG")
plt.title("Bradycardia (Heart Rate: 40 bpm)")
plt.legend()

# 頻脈波形
plt.subplot(5, 1, 3)
plt.plot(tachycardia_ecg, label="Tachycardia ECG")
plt.title("Tachycardia (Heart Rate: 120 bpm)")
plt.legend()

# ノイズの多い波形
plt.subplot(5, 1, 4)
plt.plot(noisy_ecg, label="Noisy ECG")
plt.title("Noisy ECG")
plt.legend()

# 不整脈波形
plt.subplot(5, 1, 5)
plt.plot(arrhythmia_ecg, label="Arrhythmia ECG")
plt.title("Arrhythmia ECG")
plt.legend()

# レイアウト調整
plt.tight_layout()
plt.show()

