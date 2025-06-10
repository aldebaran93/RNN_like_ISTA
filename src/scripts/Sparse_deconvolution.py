import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from scipy.linalg import toeplitz
from sklearn.linear_model import Lasso
from absorption_spectrum import *

def get_trace_slice(t_vector, trace, t_min, t_max):
    trace = dgmm(t_vector, 100e-12, -670.81e-12, -670.39e-12, 0.19, 0.24, -5.43, 3.82)
    trace = trace.astype(np.complex128)
    mask = (t_vector >= t_min) & (t_vector <= t_max)
    return trace[mask]

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

def sparse_deconvolution(y, h, alpha=0.001, threshold_ratio=0.01):
    from scipy.linalg import toeplitz
    from sklearn.linear_model import Lasso
    import numpy as np

    y = np.asarray(y).flatten()
    h = np.asarray(h).flatten()

    L = len(y) - len(h) + 1
    if L <= 0:
        raise ValueError("Signal too short compared to pulse")

    H = toeplitz(np.r_[h, np.zeros(L - 1)], np.zeros(L))

    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    lasso.fit(H, y)
    sparse_x = lasso.coef_

    threshold = threshold_ratio * np.max(np.abs(sparse_x))
    pulse_positions = np.where(np.abs(sparse_x) > threshold)[0]

    return sparse_x, pulse_positions

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

thz_pulse_init = get_trace_slice(t_vector, trace, 98e-12, 105e-12)
thz_pulse_init = np.real(thz_pulse_init)    #.astype(np.float32)

test_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions))

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
val_data = windowing(val_data_varied)

val_data = multi_pulse(val_data)

train_data = windowing(test_data)

val_data = awgn(val_data[0], snr_db=-5)
############# Evaluation after computing #####################
os.makedirs("plots", exist_ok=True)
for i in range(val_data.shape[0]):
    y = val_data[i]
    sparse_x, pulse_positions = sparse_deconvolution(y, thz_pulse_init)
    plt.figure(figsize=(12, 4))
    plt.plot(y, label='THz signal')
    plt.scatter(pulse_positions, y[pulse_positions], color='red', marker='x', label='Predicted peaks')
    plt.plot(pulse_positions, y[pulse_positions], label='reconstructed THz signal')
    plt.title(f'THz Signal vs. Predicted Pulse Peaks (Signal {i})')
    plt.xlabel('Time Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/rnn_ista_plot_at_{i}.png", dpi=300)
    plt.show()
