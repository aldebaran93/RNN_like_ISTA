import numpy as np
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt
from absorption_spectrum import *


# Konstanten
f = freq_vector #np.linspace(start_frequency, int(stop_frequency), int(step_frequency))  # in Hz

# Funktion für FSPL (in dB)
def fspl(frequency, distance):
    return 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / speed_of_light)

# Funktion für frequenzabhängigen Antennengewinn (in dBi)
def antenna_gain(frequency):
    gain = np.zeros_like(frequency)
    gain[frequency < 50e9] = 0
    mask = (frequency >= 50e9) & (frequency <= 400e9)
    gain[mask] = 15.4 * (frequency[mask] - 50e9) / (400e9 - 50e9)
    gain[frequency > 400e9] = 15.4
    return gain

# Gesamtsystemverlust (FSPL - 2*Gain)
plt.figure(figsize=(10, 6))
for d in distances:
    loss = fspl(f, d) - 2 * antenna_gain(f)
    plt.plot(f * 1e-9, loss, label=f"{d} m")

plt.title("Frequenzabhängiger Pfadverlust inkl. Antennengewinn")
plt.xlabel("Frequenz (GHz)")
plt.ylabel("Totaler Verlust (dB)")
plt.grid(True)
plt.legend(title="Distanz")
plt.tight_layout()
plt.show()

fspl_matrix = np.zeros_like(transfer_functions)

for idx, d in enumerate(distances):
    fspl_dB = fspl(freq_vector[1:], d)
    fspl_matrix[idx, 1:] = fspl_dB  # FSPL ist für DC (0 Hz) undefiniert, daher ab Index 1

# Gesamt-Dämpfung (Wasser + Freiraum)
total_attenuation = transfer_functions + fspl_matrix
plt.figure(figsize=(10, 6))
for i, d in enumerate(distances):
    plt.plot(f * 1e-9, total_attenuation[i], label=f"{d} m")

plt.title("Frequenzabhängiger Pfadverlust + Antennengewinn + Wasserdampf")
plt.xlabel("Frequenz (GHz)")
plt.ylabel("Totaler Verlust (dB)")
plt.grid(True)
plt.legend(title="Distanz")
plt.tight_layout()
plt.show()

for i, distance in enumerate(distances):
    td = np.fft.irfft(total_attenuation[i])
    td = td.astype(np.complex128)
    #trace = dgmm(t_vector, 100e-12, -670.81e-12, -670.39e-12, 0.19, 0.24, -5.43, 3.82)
    #trace = resample(trace, 3000)
    #trace = trace.astype(np.complex128)
    convolved = np.convolve(trace, td, mode="full")
    convolved = convolved[0:trace.size]
    if plot:
        # Convolved Signal
        plt.figure()
        plt.plot(t_vector * 1e12, trace, label="Generic THz-pulse")
        plt.plot(t_vector * 1e12, convolved, label=f"Convolved Pulse (Distance: {distance * 100:.1f} cm)")
        plt.plot(t_vector*1e12, np.real(np.fft.irfft(np.fft.rfft(trace)*transfer_functions[i])), label="TF Calculated")
        plt.ylabel("Amplitude (a.u.)")
        plt.xlabel("Delay Time (ps)")
        plt.xlim(98, 115)
        plt.legend()
        plt.grid()

    received_signal_spectrum = np.convolve(trace, total_attenuation[i], mode="full")
    # Rücktransformation ins Zeitbereich
    received_signals = np.real(np.fft.irfft(np.fft.rfft(received_signal_spectrum)))  #np.fft.ifft(np.fft.ifftshift(received_signal_spectrum, axes=1), axis=1).real