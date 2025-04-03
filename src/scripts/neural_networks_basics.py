import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import matplotlib.pyplot as plt
from absorption_spectrum import *

# Custom soft-thresholding layer
class SoftThresholdLayer(layers.Layer):
    def __init__(self, threshold):
        super(SoftThresholdLayer, self).__init__()
        self.threshold = tf.constant(threshold, dtype=tf.float32)

    def call(self, inputs):
        return tf.sign(inputs) * tf.maximum(tf.abs(inputs) - self.threshold, 0.0)

# Custom loss that combines reconstruction (L2) and L1 sparsity penalty.
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_value):
        super(CustomLoss, self).__init__()
        self.lambda_value = lambda_value

    def call(self, y_true, y_pred):
        l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        l1_loss = self.lambda_value * tf.reduce_sum(tf.abs(y_pred))
        return l2_loss + l1_loss

# ISTA-RNN Model with BPTT integration (focused on peak prediction)
class ISTA_RNN(Model):
    def __init__(self, input_dim, num_iterations=10, threshold=0.01):
        super(ISTA_RNN, self).__init__()
        self.num_iterations = num_iterations
        self.input_dim = input_dim
        
        # Shared layers for the iterative update
        self.We = layers.Dense(input_dim, use_bias=False)
        self.hidden = layers.Dense(input_dim, activation='tanh')
        self.dropout = layers.Dropout(0.3)
        self.soft_threshold = SoftThresholdLayer(threshold)
        
        # Two branches: one for reconstruction and one for peak prediction
        self.reconstruction = layers.Dense(input_dim, activation='linear')
        # For peak prediction, use a single unit with linear activation.
        self.peak_output = layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        hidden_state = tf.zeros((batch_size, self.input_dim))
        
        # Unroll ISTA iterations
        for t in range(self.num_iterations):
            x = self.We(inputs)
            hidden_state = self.hidden(hidden_state) + x
            hidden_state = self.soft_threshold(hidden_state)
            if training:
                hidden_state = self.dropout(hidden_state, training=training)
        
        # Reconstruction branch uses the final hidden state.
        recon = self.reconstruction(hidden_state)
        # Peak branch also uses the final hidden state.
        peak = self.peak_output(hidden_state)
        return {'reconstruction': recon, 'peak': peak}

# Define a custom peak accuracy metric with tolerance.
def peak_accuracy(y_true, y_pred, tolerance=2):
    # Round the predictions to the nearest integer
    y_pred_int = tf.cast(tf.round(y_pred), tf.int32)
    y_true_int = tf.cast(y_true, tf.int32)
    # Allow for a tolerance in samples
    correct = tf.less_equal(tf.abs(y_pred_int - y_true_int), tolerance)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

# Example windowing function and data processing (unchanged)
def windowing(data):
    window_size = 3528
    threshold = 1
    pulse_windows = []
    pulse_positions = []
    for seq_idx, sequence in enumerate(data):
        found_pulse = False
        for i in range(2000, len(sequence) - window_size + 1, window_size):
            window = sequence[i:i + window_size]
            if np.max(window) > threshold:
                pulse_windows.append(window)
                # Use the index relative to the window (i.e. argmax within the window)
                pulse_positions.append(np.argmax(window))
                found_pulse = True
                break
        if not found_pulse:
            pulse_windows.append(np.zeros(window_size))
            pulse_positions.append(0)
    return np.array(pulse_windows), np.array(pulse_positions)

# Assume trace, transfer_functions, complex_refractive_index, w_vector, speed_of_light
# are defined externally.
test_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions))
train_data_np, peak_positions_train_np = windowing(test_data)

# Process validation data (as before)
transfer_functions_val = []
distances_2 = np.arange(0.3, 0.329 + 0.001, 500e-6)
for distance in distances_2:
    transfer_function_val = np.exp(distance * 1j * complex_refractive_index * w_vector / speed_of_light) * \
                            np.exp(distance * 1j * 1.0027 * w_vector / speed_of_light)
    transfer_function_val = np.abs(transfer_function_val) * np.exp(-1j * np.angle(transfer_function_val))
    transfer_functions_val.append(transfer_function_val)

val_data_np, peak_positions_val_np = windowing(np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions_val)))

train_data = tf.convert_to_tensor(train_data_np, dtype=tf.float32)
val_data = tf.convert_to_tensor(val_data_np, dtype=tf.float32)
peak_positions_train = tf.convert_to_tensor(peak_positions_train_np, dtype=tf.float32)
peak_positions_val = tf.convert_to_tensor(peak_positions_val_np, dtype=tf.float32)

# Set input dimension (using train_data shape)
input_dim = train_data.shape[1]
num_iterations = 20  # number of unrolled ISTA iterations
ista_rnn_model = ISTA_RNN(input_dim, num_iterations=num_iterations)

# Optimizer and loss setup.
optimizer = optimizers.Adam(learning_rate=1e-4)
# Use two losses: one for reconstruction and one for peak prediction.
# Increase the weight on the peak loss to focus training on peak prediction.
loss_weight_recon = 0.5
loss_weight_peak = 1.0
custom_loss_fn = CustomLoss(0.01)

# Prepare datasets.
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, {'reconstruction': train_data,
                                                                   'peak': peak_positions_train}))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, {'reconstruction': val_data,
                                                             'peak': peak_positions_val}))
val_dataset = val_dataset.batch(32)

train_loss_history = []
val_loss_history = []
train_peak_acc_history = []
val_peak_acc_history = []

epochs = 1000
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_peak_acc_avg = tf.keras.metrics.Mean()
    
    # Training loop
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            predictions = ista_rnn_model(x_batch, training=True)
            # Compute individual losses
            loss_recon = tf.reduce_mean(tf.square(y_batch['reconstruction'] - predictions['reconstruction']))
            loss_peak = tf.reduce_mean(tf.square(y_batch['peak'] - predictions['peak']))
            loss_custom = custom_loss_fn(y_batch['reconstruction'], predictions['reconstruction'])
            loss_value = loss_weight_recon * loss_recon + loss_weight_peak * loss_peak + loss_custom
        gradients = tape.gradient(loss_value, ista_rnn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ista_rnn_model.trainable_variables))
        epoch_loss_avg.update_state(loss_value)
        
        # Update peak accuracy metric for this batch
        batch_peak_acc = peak_accuracy(y_batch['peak'], predictions['peak'])
        epoch_peak_acc_avg.update_state(batch_peak_acc)
    
    train_loss = epoch_loss_avg.result()
    train_peak_acc = epoch_peak_acc_avg.result()
    train_loss_history.append(train_loss)
    train_peak_acc_history.append(train_peak_acc)
    
    # Validation loop
    val_epoch_loss_avg = tf.keras.metrics.Mean()
    val_epoch_peak_acc_avg = tf.keras.metrics.Mean()
    for x_batch_val, y_batch_val in val_dataset:
        predictions_val = ista_rnn_model(x_batch_val, training=False)
        loss_recon_val = tf.reduce_mean(tf.square(y_batch_val['reconstruction'] - predictions_val['reconstruction']))
        loss_peak_val = tf.reduce_mean(tf.square(y_batch_val['peak'] - predictions_val['peak']))
        loss_val = loss_weight_recon * loss_recon_val + loss_weight_peak * loss_peak_val + \
                   custom_loss_fn(y_batch_val['reconstruction'], predictions_val['reconstruction'])
        val_epoch_loss_avg.update_state(loss_val)
        
        batch_val_peak_acc = peak_accuracy(y_batch_val['peak'], predictions_val['peak'])
        val_epoch_peak_acc_avg.update_state(batch_val_peak_acc)
        
    val_loss = val_epoch_loss_avg.result()
    val_peak_acc = val_epoch_peak_acc_avg.result()
    val_loss_history.append(val_loss)
    val_peak_acc_history.append(val_peak_acc)
    
    print(f"Training Loss: {train_loss:.4f} - Training Peak Acc: {train_peak_acc:.4f} | " +
          f"Validation Loss: {val_loss:.4f} - Validation Peak Acc: {val_peak_acc:.4f}")

# Prediction and Visualization of Peak Prediction
predictions = ista_rnn_model(val_data, training=False)
predicted_peaks = predictions['peak'].numpy().flatten()

# Evaluate on validation data (final average loss)
test_loss = np.mean(val_loss_history)
print(f"Test Loss: {test_loss}")

# Plot the original signal and predicted peak for a few samples.
x_axis = np.arange(train_data.shape[1])
for i in range(min(val_data.shape[0], 5)):
    plt.figure(figsize=(12, 8))
    plt.plot(x_axis, val_data_np[i], color='blue', alpha=0.7, label='Original')
    # Mark the ground-truth peak with a green circle
    true_peak = int(np.round(peak_positions_val_np[i]))
    plt.scatter(true_peak, val_data_np[i][true_peak], color='green', marker='o', s=100, label='True Peak')
    # Mark the predicted peak with a red X
    pred_peak = int(np.round(predicted_peaks[i]))
    plt.scatter(pred_peak, val_data_np[i][pred_peak], color='red', marker='x', s=100, label='Predicted Peak')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(f'Peak Prediction for Sample {i}')
    plt.legend()
    plt.show()

# Plot training history: Loss and Peak Accuracy
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Loss Over Epochs (Peak Prediction Focus)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_peak_acc_history, label='Training Peak Accuracy')
plt.plot(val_peak_acc_history, label='Validation Peak Accuracy')
plt.title('Peak Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
