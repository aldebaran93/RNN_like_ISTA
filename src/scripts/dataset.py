# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 17:27:14 2025

@author: leots
"""

import numpy as np
from absorption_spectrum import *
from sklearn.utils import shuffle
import tensorflow as tf

def get_trace_slice(t_vector, trace, t_min, t_max):
    trace = dgmm(t_vector, 100e-12, -670.81e-12, -670.39e-12, 0.19, 0.24, -5.43, 3.82)
    trace = trace.astype(np.complex128)
    mask = (t_vector >= t_min) & (t_vector <= t_max)
    return trace[mask]

def calc_transfer_function(distances):
    transfer_functions = []
    for distance in distances:
        tfct = np.exp(distance * 1j * complex_refractive_index * w_vector / speed_of_light) * \
               np.exp(distance * 1j * 1.0027 * w_vector / speed_of_light)
        tfct = np.abs(tfct) * np.exp(-1j * np.angle(tfct))
        transfer_functions.append(tfct)
    transfer_functions = np.array(transfer_functions)
    return transfer_functions
    
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
# Training data

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
val_data_amp = awgn(varying_amplitude(val_train), snr_db=-5)
val_data_multi, val_peak_positions = multi_pulse(val_train)
val_data_multi_noise = awgn(val_data_multi, snr_db=-5)
val_dataset = np.concatenate([val_train, val_data_amp, val_data_multi, val_data_multi_noise], axis=0)
val_dataset = shuffle(val_dataset, random_state=42)

