# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 20:13:57 2025

@author: leots
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from absorption_spectrum import *
from dataset import *

# Load data
y = np.concatenate([train_dataset, val_dataset], axis=0)
thz_pulse = get_trace_slice(t_vector, trace, 98e-12, 105e-12)
thz_pulse = np.real(thz_pulse).astype(np.float32)

# Peak ground truth
peak_idx = np.argmax(y, axis=1)
peak_gt = np.zeros_like(y)
for i in range(len(peak_idx)):
    peak_gt[i, peak_idx[i]] = 1

# Train-test split
train_dataset, val_dataset, train_peaks, val_peaks = train_test_split(y, peak_gt, test_size=0.2, random_state=42)

# Convert to tensors
train_dataset = tf.convert_to_tensor(train_dataset, dtype=tf.float32)
train_peaks = tf.convert_to_tensor(train_peaks, dtype=tf.float32)
val_dataset = tf.convert_to_tensor(val_dataset, dtype=tf.float32)
val_peaks = tf.convert_to_tensor(val_peaks, dtype=tf.float32)

# Custom ISTA layer
class ISTA(tf.keras.layers.Layer):
    def __init__(self, signal_length, thz_pulse, T=5, lam=0.1):
        super(ISTA, self).__init__()
        self.T = T
        self.lam = lam
        self.signal_length = signal_length
        self.thz_pulse_init = tf.convert_to_tensor(thz_pulse, dtype=tf.float32)
        thz_pulse_reshaped = self.thz_pulse_init[None, :, None]
        self.thz_pulse = tf.Variable(thz_pulse_reshaped, trainable=True, name='learned_thz_pulse')
        self.step_size = tf.Variable(0.1, trainable=True)

    def call(self, y):
        x = tf.zeros_like(y)
        for _ in range(self.T):
            thz_pulse = tf.reverse(self.thz_pulse, axis=[0])
            thz_pulse = tf.reshape(thz_pulse, (-1, 1, 1))
            y = tf.expand_dims(y, axis=2)

            conv_output = tf.nn.conv1d(tf.expand_dims(y, -1), self.thz_pulse, stride=1, padding='SAME')[:, :, 0]
            residual = y - conv_output
            update = tf.nn.conv1d(tf.expand_dims(residual, -1), tf.reverse(self.thz_pulse, [1]), stride=1, padding='SAME')[:, :, 0]
            x = x + self.step_size * update
            x = tf.sign(x) * tf.nn.relu(tf.abs(x) - self.lam)
        return x

# Complete model
class LearnedISTA(tf.keras.Model):
    def __init__(self, signal_length, thz_pulse, T=5, lam=0.1):
        super(LearnedISTA, self).__init__()
        self.ista = ISTA(signal_length, thz_pulse, T=T, lam=lam)
        self.global_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(signal_length, activation='sigmoid')
        self.signal_length = signal_length
        self.thz_pulse_init = tf.convert_to_tensor(thz_pulse, dtype=tf.float32)[None, :, None]

    def call(self, y):
        x = self.ista(y)
        pooled = self.global_pool(tf.expand_dims(x, -1))  # shape: (batch, features)
        peak_pred = self.dense2(self.dense1(pooled))  # shape: (batch, signal_length)
        return x, peak_pred

# Instantiate model
signal_length = train_dataset.shape[1]
model = LearnedISTA(signal_length, thz_pulse, T=10, lam=0.1)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training parameters
epochs = 50
batch_size = 32
train_losses, val_losses = [], []

# Training loop
for epoch in range(epochs):
    # Shuffle training set
    indices = tf.random.shuffle(tf.range(len(train_dataset)))
    train_dataset_shuffled = tf.gather(train_dataset, indices)
    train_peaks_shuffled = tf.gather(train_peaks, indices)

    # Mini-batch training
    for i in range(0, len(train_dataset), batch_size):
        y_batch = train_dataset_shuffled[i:i + batch_size]
        peak_batch = train_peaks_shuffled[i:i + batch_size]

        with tf.GradientTape() as tape:
            x_pred, peak_pred = model(y_batch)
            loss_peak = tf.reduce_mean(tf.square(peak_pred - peak_batch))
            loss_recon = tf.reduce_mean(tf.square(x_pred - y_batch))
            loss_pulse_reg = tf.reduce_mean(tf.square(model.ista.thz_pulse - model.thz_pulse_init))
            loss = 1.0 * loss_recon + 0.001 * loss_peak + 0.01 * loss_pulse_reg

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Validation
    x_val_pred, peak_val_pred = model(val_dataset)
    val_loss_peak = tf.reduce_mean(tf.square(peak_val_pred - val_peaks))
    val_loss_recon = tf.reduce_mean(tf.square(x_val_pred - val_dataset))
    val_loss_total = 1.0 * val_loss_recon + 0.001 * val_loss_peak

    print(f"Epoch {epoch + 1}: Train Loss = {loss.numpy():.6f}, Val Loss = {val_loss_total.numpy():.6f}")
    train_losses.append(loss.numpy())
    val_losses.append(val_loss_total.numpy())

# Plot loss curves
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Example prediction
x_example, peak_example = model(val_dataset)
for i in range(val_dataset.shape[0]):
    plt.figure(figsize=(10, 3))
    plt.plot(val_dataset[i].numpy(), label='Input y')
    plt.plot(x_example[i].numpy(), label='Predicted x')
    plt.plot(val_peaks[i].numpy(), label='True Peak', linestyle='--')
    plt.plot(peak_example[i].numpy(), label='Predicted Peak', linestyle=':')
    plt.legend()
    plt.title(f'Example {i+1}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
