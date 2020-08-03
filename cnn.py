import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt

IMG_HEIGHT = 50
IMG_WIDTH = 50
EPOCHS = 15
LR = 1e-3
MODEL_NAME = f'nonBalanced-{EPOCHS}-Epoch'
logdir = "logs/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
outputs = [-0.5,-0.2,0, 0.2, 0.5]

print('started')

#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=32,input_shape=(IMG_HEIGHT,IMG_HEIGHT,1), kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))


# 3rd Convolutional Layer
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())


# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(2048,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))


# Output Layer
model.add(Dense(5))
model.add(Activation('softmax'))


# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

model.summary()

print('compiled')

data = np.load('D:\CARLA_0.9.9.4\WindowsNoEditor\PythonAPI\examples\\dataSquare.npy', allow_pickle=True)
X = []
Y = []
for feature, label in data:
    X.append(np.asarray(feature))

    l = label[1]
    if l > -0.1 and l < 0.1:
        Y.append(np.array([0, 0, 1, 0, 0]))
        continue
    elif l >= 0.1 and l < 0.3:
        Y.append(np.array([0, 0, 0, 1, 0]))
        continue
    elif l >= 0.3:
        Y.append(np.array([0, 0, 0, 0, 1]))
        continue
    elif l > -0.3 and l <= -0.1:
        Y.append(np.array([0, 1, 0, 0, 0]))
        continue
    elif l <= -0.3:
        Y.append(np.array([1, 0, 0, 0, 0]))
        continue

    Y.append(np.asarray(label[1]))

X = np.array(X).reshape(-1, 50, 50, 1)
x_train = np.asarray(X[:-1000])
x_test = np.asarray(X[-1000:])
y_train = np.asarray(Y[:-1000])
y_test = np.asarray(Y[-1000:])

print(x_train.shape)
print(y_train.shape)

model.fit(x=x_train, y=y_train, epochs=EPOCHS, shuffle=True, verbose=1, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
model.save(MODEL_NAME)
