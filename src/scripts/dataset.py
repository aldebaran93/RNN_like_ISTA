# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 17:27:14 2025

@author: leots
"""

import numpy as np
from absorption_spectrum import *
from sklearn.utils import shuffle
import tensorflow as tf
import scipy
import matplotlib.pyplot as plt
from scipy.io import loadmat

def get_trace_slice(t_vector, trace, t_min, t_max):
    trace = dgmm(t_vector, 100e-12, -670.81e-12, -670.39e-12, 0.19, 0.24, -5.43, 3.82)
    trace = trace.astype(np.complex128)
    mask = (t_vector >= t_min) & (t_vector <= t_max)
    return trace[mask]

def windowing(data):
    window_size = 12500
    threshold = 0
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

def varying_amplitude(data):
    train_data = tf.convert_to_tensor(windowing(data), dtype=tf.float32)
    num_samples = tf.shape(data)[0]
    amplitudes = tf.random.uniform(shape=(num_samples, 1), minval=1, maxval=11, dtype=tf.int32)
    amplitudes = tf.cast(amplitudes, tf.float32)
    train_data = data * amplitudes
    return train_data

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

def multi_pulse(val_data):
    val_data_np = val_data.numpy() if isinstance(val_data, tf.Tensor) else val_data
    num_traces = val_data_np.shape[0]
    signal_length = val_data_np.shape[1]
    
    # Abstand zwischen Pulsen (z. B. von 1000 auf 200)
    start_distance = 1000
    min_distance = 0
    distances = np.linspace(start_distance, min_distance, num_traces).astype(int)
    
    new_data = []
    peak_positions = []
    
    for i, d in enumerate(distances):
        # Wähle zwei verschiedene Pulse
        idx1, idx2 = np.random.choice(len(val_data_np), 2, replace=False)
        pulse1 = val_data_np[idx1]
        pulse2 = val_data_np[idx2]
    
        # Berechne Einfügepositionen (mit Sicherheitsabstand zu Rändern)
        pulse_len = 400  # Fester Ausschnitt – passe nach Bedarf an
        center1 = signal_length // 2 - d // 2
        center2 = center1 + d
    
        if center1 - pulse_len//2 < 0 or center2 + pulse_len//2 > signal_length:
            continue  # überspringen, wenn zu eng
    
        # Ausschneiden (z.B. 400 Samples rund um das Peak-Zentrum)
        mid1 = np.argmax(pulse1)
        mid2 = np.argmax(pulse2)
    
        segment1 = pulse1[mid1 - pulse_len//2 : mid1 + pulse_len//2]
        segment2 = pulse2[mid2 - pulse_len//2 : mid2 + pulse_len//2]
    
        if len(segment1) != pulse_len or len(segment2) != pulse_len:
            continue
    
        # Neues leeres Signal
        combined = np.zeros(signal_length)
        combined[center1 - pulse_len//2 : center1 + pulse_len//2] += segment1
        combined[center2 - pulse_len//2 : center2 + pulse_len//2] += segment2
    
        # Neue Peaks
        peak_positions.append([center1, center2])
        new_data.append(combined)
    
    # In Arrays konvertieren
    new_data = np.array(new_data)
    peak_positions = np.array(peak_positions)
    return new_data, peak_positions

def extract_data():
    mat_data = scipy.io.loadmat("noTXVoltage.mat")
    firstchannel = mat_data['firstchannel'].squeeze()  # ensure 1D array
    mean_value = np.mean(firstchannel)
    std_value = np.std(firstchannel)
    print("Mean value is: ", mean_value)
    print("Standard deviation value is: ", std_value)
    return firstchannel, mean_value, std_value
    
def freq_noise_data():
    # Load the .mat file (assumes variable name is 'firstchannel')
    mat = loadmat('noTXVoltage.mat')
    data = mat['firstchannel'][0]  # Assuming you want the first row
    
    # Sampling information
    Ts = 33.3333e-15  # Sampling time = 33.3333 fs
    fs = 1 / Ts       # Sampling frequency in Hz
    N = len(data)     # Number of samples
    
    # Time and frequency vectors
    freqs = np.fft.fftshift(np.fft.fftfreq(N, Ts))
    spectrum = np.fft.fftshift(np.fft.fft(data))
    
    # Plot the magnitude spectrum (in THz)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs * 1e-12, np.abs(spectrum))  # Convert Hz to THz
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Magnitude')
    plt.title('Spectrum of the Signal')
    plt.grid(True)
    plt.xlim(-0.02,0.02)
    plt.tight_layout()
    plt.show()
    return data

def pink_noise(signal_array, alpha=1.0, relative_amplitude=0.01):
    n_signals, n_samples = signal_array.shape
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]  # avoid div by 0
    scaling_factors = 1 / (freqs ** alpha)

    noise = np.zeros_like(signal_array)

    for i in range(n_signals):
        random_phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))
        amplitude_spectrum = np.random.randn(len(freqs)) * scaling_factors
        spectrum = amplitude_spectrum * random_phases
        signal = np.fft.irfft(spectrum, n=n_samples)

        # Normalize noise to unit std
        signal /= np.std(signal)

        # Scale noise to relative amplitude of original signal
        signal_amplitude = np.max(np.abs(signal_array[i]))
        noise[i] = signal * signal_amplitude * relative_amplitude

    return noise


data = loadmat('noTXVoltage.mat')['firstchannel']
#data = spectrum
trace_counts = [10, 100, 1000]
means = []
stds = []
for n in trace_counts:
    segment = data[:n, :]  # z. B. (10, 3000), (100, 3000), ...
    
    mean = np.mean(segment)
    std = np.std(segment)
    
    print(f"{n} Traces:")
    print(f"→ Mittelwert: {mean:.6f}")
    print(f"→ Standardabweichung: {std:.6f}\n")
steps = np.arange(10, 1001, 10)
for n in steps:
    subset = data[:n, :]  # erste n Messungen
    means.append(np.mean(subset))
    stds.append(np.std(subset))

plt.figure(figsize=(10, 6))
plt.plot(steps, means, label='Mittelwert', marker='o')
plt.plot(steps, stds, label='Standardabweichung', marker='s')
plt.xlabel("Anzahl der Traces")
plt.ylabel("Wert")
plt.title("Frequenzbereeich Verlauf von Mittelwert und Standardabweichung mit wachsender Anzahl von Traces")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
        

# Training data
test_data, mean_value, std_value = extract_data()
transfer_functions = []
train_dataset = []
distances = np.arange(0.3, 0.329 + 0.001, 1e-3)  # 30 cm to 33 cm, step 0.5 mm
transfer_functions = calc_transfer_function(distances)
test_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions))
train_data = windowing(test_data)
train_data_amp = varying_amplitude(train_data)
train_data_amp = train_data_amp.numpy()
train_data_multi, train_peak_positions = multi_pulse(train_data)
train_dataset = np.concatenate([train_data, train_data_amp, train_data_multi])
train_dataset = shuffle(train_dataset, random_state=42)
#train_peak = np.concatenate([np.argmax(train_data), np.argmax(train_data_amp), train_peak_positions])

# Validierung
transfer_functions_val = []
val_dataset = []
distances_2 = np.arange(0.3, 0.329 + 0.001, 500e-6)
transfer_functions_val = calc_transfer_function(distances_2)
val_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions_val))
val_train = windowing(val_data)
val_data_amp = awgn(varying_amplitude(val_train), snr_db=60)
val_data_multi, val_peak_positions = multi_pulse(val_train)
val_data_multi_noise = awgn(val_data_multi, snr_db=60)
val_dataset = np.concatenate([val_train, val_data_amp, val_data_multi, val_data_multi_noise], axis=0)
val_dataset = shuffle(val_dataset, random_state=42)
relative_amplitude = 0.01  # e.g., noise will be 1% of signal amplitude
noise = pink_noise(val_dataset, alpha=1.0, relative_amplitude=relative_amplitude)
#noise_level = 1
val_dataset = val_dataset + noise

def metric(predicted_peaks, peak_positions_val, max_dist=100):
    predicted_peaks = np.array(predicted_peaks)
    true_peaks = np.array(peak_positions_val)

    all_y_true = []
    all_y_pred = []

    n_signals = predicted_peaks.shape[0]

    for row_idx in range(n_signals):
        preds = predicted_peaks[row_idx]
        trues = true_peaks[row_idx]
        matched_true = set()

        y_true = []
        y_pred = []

        # Evaluate predicted peaks for this signal
        for pred in preds:
            if pred == 0:
                continue  # ignore zero placeholders
            match_found = False
            for i, true in enumerate(trues):
                if true == 0:
                    continue
                if i not in matched_true and abs(pred - true) <= max_dist:
                    match_found = True
                    matched_true.add(i)
                    break
            if match_found:
                y_true.append(1)  # correct match
                y_pred.append(1)
            else:
                y_true.append(0)  # false positive
                y_pred.append(1)

        # Add false negatives: unmatched true peaks
        for i, true in enumerate(trues):
            if true == 0:
                continue
            if i not in matched_true:
                y_true.append(1)  # missed true peak
                y_pred.append(0)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    # Compute and print metrics
    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return all_y_pred, all_y_true, (precision, recall, f1)

def confusion_matrice(y_pred, y_true):

    # Get raw confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert to percentages
    cm_percent = cm / np.sum(cm) * 100
    
    # Create annotated labels
    labels = np.array([["{0:.4f}%".format(val) for val in row] for row in cm_percent])
    
    # Plotting
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=labels, fmt='', cmap='Blues', cbar=True,
                xticklabels=["Present", "Absent"],
                yticklabels=["Present", "Absent"])
    
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("(a)", loc='center', fontsize=12)
    plt.tight_layout()
    plt.show()
# =============== 3. Trainings- und Validierungsdaten ===============

peak_positions_train = tf.convert_to_tensor(np.argmax(train_dataset, axis=1), dtype=tf.float32)

def get_peaks(signal_array, max_peaks=2):
    num_signals = signal_array.shape[0]
    peak_positions = np.zeros((num_signals, max_peaks), dtype=int)  # Initialize with 0
    
    for i in range(num_signals):
        signal = signal_array[i]

        # Find positive and negative peaks
        pos_peaks, _ = find_peaks(signal)
        neg_peaks, _ = find_peaks(-signal)
        
        # Combine and sort by absolute amplitude
        all_peaks = np.concatenate((pos_peaks, neg_peaks))
        all_amplitudes = np.abs(signal[all_peaks])
        
        if len(all_peaks) > 0:
            top_indices = np.argsort(all_amplitudes)[::-1][:max_peaks]
            top_peaks = all_peaks[top_indices]
            peak_positions[i, :len(top_peaks)] = np.sort(top_peaks)  # sort by time
        # else: remains as zeros
        if len(top_peaks) == 2:
            val1, val2 = np.abs(signal[top_peaks[0]]), np.abs(signal[top_peaks[1]])
            if val2 < 1 * val1:
                top_peaks = top_peaks[:1]  # keep only the stronger one
    
    return peak_positions

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.optimize import curve_fit

# === PARAMETER ===
Ts = 33.3333e-15             # Abtastperiode in Sekunden
fs = 1 / Ts                  # Abtastrate in Hz (~30 THz)
nperseg = 1024               # Länge der Welch-Fenster

# === DATEN LADEN ===
# Wähle den richtigen Ladebefehl je nach Format
# traces = np.load("rauschen.npy")          # für .npy-Datei
# traces = np.genfromtxt("rauschen.csv", delimiter=',')  # für .csv
# from scipy.io import loadmat
traces = loadmat("noTXVoltage.mat")['firstchannel']       # für .mat

# === PSD FÜR JEDE SPUR BERECHNEN ===
psd_list = []

for trace in traces:
    f, Pxx = welch(trace, fs=fs, nperseg=nperseg)
    psd_list.append(Pxx)

psd_list = np.array(psd_list)
psd_mean = np.mean(psd_list, axis=0)

# === PSD PLOT ===
plt.figure(figsize=(10, 5))
plt.loglog(f / 1e12, psd_mean, label="Mittlere PSD")
plt.xlabel("Frequenz [THz]")
plt.ylabel("PSD [V²/Hz]")
plt.title("Gemittelte PSD über 1000 THz-Rauschtraces")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === FIT: PSD ~ A / f^alpha + N0 ===
def powerlaw_psd(f, A, alpha, N0):
    return A / (f**alpha + 1e-20) + N0

# Fit ohne f=0 (vermeide Division durch 0)
popt, _ = curve_fit(powerlaw_psd, f[1:], psd_mean[1:], p0=[1e-22, 1, 1e-24])
A_fit, alpha_fit, N0_fit = popt

print(f"Erkannte Rauschform:")
print(f"  ➤ A     = {A_fit:.2e}")
print(f"  ➤ alpha = {alpha_fit:.2f} → {'weißes Rauschen' if np.isclose(alpha_fit, 0, atol=0.2) else '1/f-Rauschen' if np.isclose(alpha_fit, 1, atol=0.2) else '1/f²-Rauschen' if np.isclose(alpha_fit, 2, atol=0.2) else 'gemischt/unbekannt'}")
print(f"  ➤ N0    = {N0_fit:.2e}")

# === FIT-PLOT ===
psd_fit = powerlaw_psd(f, *popt)

plt.figure(figsize=(10, 5))
plt.loglog(f / 1e12, psd_mean, label='Gemittelte PSD')
plt.loglog(f / 1e12, psd_fit, '--', label=f'Fit: $A/f^{{{alpha_fit:.2f}}} + N_0$')
plt.xlabel("Frequenz [THz]")
plt.ylabel("PSD [V²/Hz]")
plt.title("Rauschanalyse per Power-Law-Fit")
plt.ylim(10e-21, 0.9e-17)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




