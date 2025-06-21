import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from absorption_spectrum import *
from dataset import *

def awgn(signals, snr_db):
    signals = np.asarray(signals, dtype=np.float32)
    noisy_signals = np.empty_like(signals)
    for i in range(signals.shape[0]):
        signal = signals[i]
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
        noisy_signals[i] = signal + noise
    return noisy_signals

def matched_filter(val_data, train_data, peak_positions_val):
    from scipy.signal import correlate, find_peaks
    
    val_data_np = val_data.numpy() if isinstance(val_data, tf.Tensor) else val_data
    num_traces = val_data_np.shape[0]
    signal_length = val_data_np.shape[1]
    
    # Erzeuge verrauschte Versionen mit verschiedenen SNR-Werten (dB)
    snr_db_values = np.arange(0, 21, 2)
    detection_rates = []
    detected_peaks = []
    
    for snr_db in snr_db_values:
        detections = 0
        for i in range(num_traces):
            noisy_trace = awgn(val_data, snr_db)
            # Matched Filter (Korrelation)
            mf_output = correlate(noisy_trace[i], val_data[i], mode='same')
            detected_peak, _ = find_peaks(mf_output, height=0.5 * np.max(mf_output), distance=50)
            # Treffer, wenn ein Peak im Bereich ±5 Samples
            if any(np.abs(detected_peaks - peak_positions_val[i]) <= 5):
                detections += 1
        detected_peaks.append(detected_peak)
        detection_rate = detections / signal_length
        detection_rates.append(detection_rate)
    
    return detection_rates, detected_peaks
    
def windowing(data):
    window_size = 3528
    threshold = 1
    pulse_windows, pulse_positions = [], []

    for seq in data:
        for i in range(2000, len(seq) - window_size + 1, window_size):
            window = seq[i:i + window_size]
            if np.max(window) > threshold:
                pulse_windows.append(window)
                pulse_positions.append(i)
                break
        else:
            pulse_windows.append(np.zeros(window_size))
            pulse_positions.append(None)
    return np.array(pulse_windows)

def evaluate_peak_detection(predicted_peaks, ground_truth_peaks, val_data, tolerance=50):
    tp = 0
    fp = 0
    fn = 0

    for idx, (gt_peaks, pred_peaks) in enumerate(zip(ground_truth_peaks, predicted_peaks)):
        gt_peaks = np.atleast_1d(gt_peaks).tolist()
        pred_peaks = np.atleast_1d(pred_peaks).tolist()
        matched_gt = set()

        for p in pred_peaks:
            match = False
            for i, gt in enumerate(gt_peaks):
                if abs(p - gt) <= tolerance and i not in matched_gt:
                    tp += 1
                    matched_gt.add(i)
                    match = True
                    break
            if not match:
                fp += 1

        fn += len(gt_peaks) - len(matched_gt)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

test_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions))
#test_data_noisy = awgn(test_data, snr_db=5)

# Validation
transfer_functions_val = []
distances_2 = np.arange(0.3, 0.329 + 0.001, 500e-6)
for distance in distances_2:
    tfct_val = np.exp(distance * 1j * complex_refractive_index * w_vector / speed_of_light) * \
               np.exp(distance * 1j * 1.0027 * w_vector / speed_of_light)
    tfct_val = np.abs(tfct_val) * np.exp(-1j * np.angle(tfct_val))
    transfer_functions_val.append(tfct_val)
val_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions_val))
scales = np.random.uniform(0.5, 1.5, size=(val_data.shape[0], 1))
val_data_varied = val_data * scales

train_data = windowing(test_data)
val_data = windowing(val_data_varied)

val_data = awgn(val_data, snr_db=5)

#peak_positions_train = np.argmax(train_data, axis=1)
peak_positions_val = np.argmax(val_dataset, axis=1)
detection_rates, predicted_peaks = matched_filter(val_dataset, train_dataset, peak_positions_val)

############# Evaluation after computing #####################
result = evaluate_peak_detection(predicted_peaks, peak_positions_val, val_data)

# Plot Detection Rate vs SNR
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(np.arange(0, 21, 2), detection_rates, marker='o', markersize=10)
plt.xlabel('SNR (dB)')
plt.ylabel('Detection Rate')
plt.title('Matched Filter Detection Rate vs SNR')
plt.grid(True)
plt.ylim(0, 1.05)
plt.savefig(f"plots/snr_vs_detectionrate.png", dpi=300)
plt.show()
