from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt
import numpy as np
import random

img_h, img_w = 48, 48
batch_size = 32

train_dir = 'emotion_detection_dataset/train'
test_dir = 'emotion_detection_dataset/test'

train_data = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_data = ImageDataGenerator(rescale=1. / 255)

train_gen = train_data.flow_from_directory(
    train_dir,
    color_mode='grayscale',
    target_size=(img_h, img_w),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_gen = test_data.flow_from_directory(
    test_dir,
    color_mode='grayscale',
    target_size=(img_h, img_w),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img, label = train_gen.__next__()

# i = random.randint(0, img.shape[0] - 1)
# image = img[i]
# lab1 = class_labels[label[i].argmax()]
# plt.imshow(image[:, :, 0], cmap='gray')
# plt.title(lab1)
# plt.show()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Use the same paths for counting images
num_train_imgs = sum([len(files) for r, d, files in os.walk(train_dir)])
num_test_imgs = sum([len(files) for r, d, files in os.walk(test_dir)])

print(f'Number of training images: {num_train_imgs}')
print(f'Number of testing images: {num_test_imgs}')

epochs = 20

# Enable eager execution for debugging
import tensorflow as tf
tf.config.run_functions_eagerly(True)

hist = model.fit(
    train_gen,
    steps_per_epoch=num_train_imgs // batch_size,
    epochs=epochs,
    validation_data=test_gen,
    validation_steps=num_test_imgs // batch_size
)

model.save('emotion_recogniser.h5')

loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs_range = range(1, len(loss) + 1)
plt.plot(epochs_range, loss, 'y', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

plt.plot(epochs_range, acc, 'y', label='Training acc')
plt.plot(epochs_range, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()