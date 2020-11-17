from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    Flatten(input_shape=(64, )),
    Dense(16, activation='relu'), # hidden layer
    Dense(16, activation='relu'), # hidden layer
    Dense(16, activation='relu'), # hidden layer
    Dense(8, activation='softmax') # output layer
])

model.summary()