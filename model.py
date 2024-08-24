from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt

model = load_model('emotion_recogniser.h5')

# load image and convert it to grayscale
def preprocess_img(img_path, size=(48, 48)):
    img = load_img(img_path, color_mode='grayscale', target_size=size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

img_path = 'sachin_happy.png'

img = preprocess_img(img_path)

prediction = model.predict(img)

predicted = np.argmax(prediction, axis=1)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
label = class_names[predicted]

print(f'Predicted class: {label}')

plt.imshow(img[0, :, :, 0], cmap='gray')
plt.title(f'Prediction: {label}')
plt.axis('off')
plt.show()
