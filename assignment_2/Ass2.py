import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu, elu, sigmoid, tanh
import matplotlib.pyplot as plt



# Load data
data_dict = pickle.load(open('california-housing-dataset.pkl', 'rb'))
x_train, y_train = data_dict['x_train'], data_dict['y_train']
x_test, y_test = data_dict['x_test'], data_dict['y_test']

# ----------------------------------------------------------------
# a.)
print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
#
# print(max(y_train))
# print(min(y_train))
#
# med_income = x_train[:, 0]
# house_age = x_train[:, 1]
# avg_rooms = x_train[:, 2]
# avg_brooms = x_train[:, 3]
# pop = x_train[:, 4]
# avg_occup = x_train[:, 5]
# coordinates = x_train[:, -2:]
# target = y_train
#
# sns.scatterplot(x=coordinates[:, 0], y=coordinates[:, 1],
#                 size=target, hue=target, palette="viridis", alpha=0.5)
# plt.legend(title="Value", loc="upper right")
# plt.axis([31, 43, -125, -113])
# plt.show()


# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(scaled_x_train, y_train, test_size=0.2, random_state=42)

learning_rate = 0.001
batch_size = 32
num_epochs = 300
l2_regularization = 5.995851789242676e-06
dropout_rate = 0.30875043077189024

# Model
#tanh
model = Sequential([
    Dense(units=256, input_shape=(x_train.shape[1],), activation='tanh', kernel_regularizer=l2(l2_regularization)),
    BatchNormalization(),
    Dense(units=256, activation='tanh', kernel_regularizer=l2(l2_regularization)),
    Dense(units=256, activation='tanh', kernel_regularizer=l2(l2_regularization)),
    Dropout(dropout_rate),
    Dense(units=1),
])


model.summary()

lr_schedule = schedules.ExponentialDecay(
    learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)

model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mean_squared_error', metrics=['mean_absolute_error'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_val, y_val),
          shuffle=True, verbose=2, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(scaled_x_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Training versus Validation Error")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc = 'best')
plt.show()

# Definition: Learning rate is a hyperparameter that determines the size of the steps taken during the optimization process. It controls how quickly or slowly a neural network learns.
# Function: A higher learning rate makes the model learn faster but may lead to overshooting and instability. A lower learning rate makes learning more stable but slower.
# Tuning: You can experiment with different learning rates to find the value that allows your model to converge to a good solution without diverging.
# Batch Size (batch_size):
#
# Definition: Batch size determines the number of data samples used in each iteration of the training process. The training data is divided into batches, and the model's parameters are updated after each batch.
# Function: Smaller batch sizes can provide a more accurate gradient estimate but require more iterations. Larger batch sizes can speed up training but might lead to noisy gradient estimates.
# Tuning: The optimal batch size depends on your dataset and available computational resources. Common values include 32, 64, 128, or 256, but the best choice may vary.
# Number of Epochs (num_epochs):
#
# Definition: An epoch is a complete pass through the entire training dataset. The number of epochs determines how many times the model sees the entire dataset during training.
# Function: Training for more epochs allows the model to learn more from the data but can lead to overfitting if not controlled.
# Tuning: You should monitor your model's performance on a validation set and stop training when it starts to overfit. Early stopping is a technique used to automatically stop training when validation performance degrades.
# L2 Regularization (l2_regularization):
#
# Definition: L2 regularization (also known as weight decay) is a regularization technique that adds a penalty term to the loss function, discouraging large weights in the model.
# Function: L2 regularization helps prevent overfitting by encouraging the model to have smaller weights, which makes it less likely to fit the training data noise.
# Tuning: You can adjust the strength of L2 regularization (the value you provided) to control the trade-off between fitting the data and regularization. Larger values increase the regularization effect.
# Dropout Rate (dropout_rate):
#
# Definition: Dropout is a regularization technique that randomly sets a fraction of neuron outputs to zero during each training iteration. It helps prevent overfitting.
# Function: Dropout introduces randomness, which forces the model to be more robust and prevents it from relying too heavily on any single neuron.
# Tuning: The dropout rate determines the fraction of neurons that are "dropped out" during training. Common values are 0.2, 0.5, or 0.7. You can experiment with different rates to find the one that works best for your model.