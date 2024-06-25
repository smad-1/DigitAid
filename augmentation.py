# %%
from sklearn.model_selection import train_test_split
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# %%
data_dir = "screenshots"

# %%

data_dir = pathlib.Path(data_dir)

# %%
data_dir

# %%
list(data_dir.glob('*/*.png'))[:5]

# %%
image_count = len(list(data_dir.glob('*/*.png')))

# %%
print(image_count)

# %%
one = list(data_dir.glob('one/*'))

one[:5]

# %%
PIL.Image.open(str(one[1]))

# %%
five = list(data_dir.glob('five/*'))
PIL.Image.open(str(five[0]))

# %%
# Read digit images from disk into numpy array using opencv

digit_images_dict = {
    'zero': list(data_dir.glob('zero/*')),
    'one': list(data_dir.glob('one/*')),
    'two': list(data_dir.glob('two/*')),
    'three': list(data_dir.glob('three/*')),
    'four': list(data_dir.glob('four/*')),
    'five': list(data_dir.glob('five/*')),
    'six': list(data_dir.glob('six/*')),
    'seven': list(data_dir.glob('seven/*')),
    'eight': list(data_dir.glob('eight/*')),
    'nine': list(data_dir.glob('nine/*')),
}

digit_labels_dict = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
}

# %%
digit_images_dict['one'][:5]
# %%
str(digit_images_dict['one'][0])
# %%
img = cv2.imread(str(digit_images_dict['one'][0]))
# %%
img.shape
# %%
cv2.resize(img, (180, 180)).shape
# %%
X, y = [], []

for digit_name, images in digit_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (180, 180))
        X.append(resized_img)
        y.append(digit_labels_dict[digit_name])

X = np.array(X)
y = np.array(y)

# %%
# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# %%
# Preprocessing: scale images
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

# %%
# Build convolutional neural network and train it
num_classes = 10

model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=30)

# %%

model.evaluate(X_test_scaled, y_test)

# %%
# Here we see that while train accuracy is very high (99%), the test accuracy is significantly low (89.99%) indicating overfitting. Let's make some predictions before we use data augmentation to address overfitting

predictions = model.predict(X_test_scaled)

predictions
# %%
score = tf.nn.softmax(predictions[0])

np.argmax(score)
# %%
y_test[0]

# %%
# Define image height and width based on the resizing operation
img_height = 180
img_width = 180
# %%
# Improve Test Accuracy Using Data Augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# %%
# Original Image

plt.axis('off')
plt.imshow(X[0])
# %%
# <matplotlib.image.AxesImage at 0x15c049d6e20>

# Newly generated training sample using data augmentation

plt.axis('off')
plt.imshow(data_augmentation(X)[0].numpy().astype("uint8"))
# %%
# <matplotlib.image.AxesImage at 0x15c049d6490>

# Train the model using data augmentation and a drop out layer
num_classes = 10

model = Sequential([
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=30)

# %%
model.evaluate(X_test_scaled, y_test)

# %%
# Define the path to save the model
models_dir = "models"
model_name = "augmented_cnn"

# Create the directory if it doesn't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Path to save the model
model_path = os.path.join(models_dir, model_name + ".keras")

# Save the model
model.save(model_path)

# %%
