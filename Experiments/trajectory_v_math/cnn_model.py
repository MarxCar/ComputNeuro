%matplotlib qt
from keras.models import Sequential, load_model
from keras.datasets import imdb
from keras.layers import Dense, Dropout
from keras.layers import LSTM, RNN, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

pos_set = np.load("./train/pos_set.npy")
neg_set = np.load("./train/neg_set.npy")

complete = np.concatenate((pos_set, neg_set))
y = np.concatenate((np.ones((pos_set.shape[0],1)), np.zeros((neg_set.shape[0],1))))

scaled_data = []

for trajectory in complete:
    std = StandardScaler()
    scaled_data.append(std.fit_transform(trajectory))

scaled_data = np.asarray(scaled_data)
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.25, random_state=42)

model_cnn = Sequential()
model_cnn.add(Conv1D(64, 7, activation="relu", input_shape=(999,2)))
model_cnn.add(MaxPooling1D(5))
model_cnn.add(Conv1D(64, 7, activation="relu"))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dense(1, activation="softmax"))

model_cnn.compile(optimizer="rmsprop",
    loss='binary_crossentropy',
    metrics=['acc'])

history_cnn = model_cnn.fit(X_train, y_train,
    epochs=10,
    validation_split=0.3,
    shuffle=True,
    callbacks=[ModelCheckpoint("./models_1Dcnn/weights.{epoch:02d}-{val_loss:.2f}.hdf5")])
