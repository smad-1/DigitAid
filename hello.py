# # DATA COLLECTION FROM PAINT

# # %%
# import numpy as np  # type: ignore # pip install numpy
# from sklearn import metrics
# from sklearn.svm import SVC
# import joblib
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from sklearn.utils import shuffle  # pip install scikit-learn
# import pandas as pd  # pip install pandas
# import cv2
# import csv
# import glob
# import pyscreenshot as ImageGrab
# import time
# import Augmentor
# from PIL import Image
# import os
# import tensorflow as tf
# import PIL

# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

# # %%


# def take_screenshot(digit, images_folder):
#     for i in range(70, 90):
#         time.sleep(3)
#         im = ImageGrab.grab(bbox=(160, 300, 650, 800))  # x1,y1,x2,y2
#         print("Image Captured", i)
#         im.save(images_folder + digit + str(i) + '.png')
#         print("Image Saved", i)


# # %%


# # Digit 0

# # images_folder = "./screenshots/zero/"
# # digit = "zero"
# # take_screenshot(digit, images_folder)
# # %%

# # Digit 1
# # images_folder = "./screenshots/one/"
# # digit = "one"
# # take_screenshot(digit, images_folder)

# # %%
# # Digit 2
# # images_folder = "./screenshots/two/"
# # digit = "two"
# # take_screenshot(digit, images_folder)

# # %%
# # Digit 3
# # images_folder = "./screenshots/three/"
# # digit = "three"
# # take_screenshot(digit, images_folder)

# # %%
# # Digit 4
# # images_folder = "./screenshots/four/"
# # digit = "four"
# # take_screenshot(digit, images_folder)

# # %%
# # Digit 5
# # images_folder = "./screenshots/five/"
# # digit = "five"
# # take_screenshot(digit, images_folder)
# # %%
# # Digit 6
# # images_folder = "./screenshots/six/"
# # digit = "six"
# # take_screenshot(digit, images_folder)

# # %%
# # Digit 7
# # images_folder = "./screenshots/seven/"
# # digit = "seven"
# # take_screenshot(digit, images_folder)

# # %%
# # Digit 8
# # images_folder = "./screenshots/eight/"
# # digit = "eight"
# # take_screenshot(digit, images_folder)

# # %%

# # Digit 9
# # images_folder = "./screenshots/nine/"
# # digit = "nine"
# # take_screenshot(digit, images_folder)

# # %%

# # GENERATE DATASET FROM IMAGES - DO NOT EXECUTE/RUN THIS CELL THIS AGAIN!!


# # 0 -> background, 1 -> digit

# header = ["label"]
# for i in range(0, 784):
#     header.append("pixel" + str(i))

# with open('dataset.csv', 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(header)

# folders = ['zero', 'one', 'two', 'three', 'four',
#            'five', 'six', 'seven', 'eight', 'nine']

# for label in range(10):
#     dirList = glob.glob("./screenshots/" + folders[label] + "/*.png")

#     for img_path in dirList:
#         im = cv2.imread(img_path)
#         im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # converts to grayscale
#         # blurs image to make it smooth
#         im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
#         # resize to 28x28
#         roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

#         data = []
#         data.append(label)
#         rows, cols = roi.shape

#         # Fill the data array with pixels one by one.
#         for i in range(rows):
#             for j in range(cols):
#                 k = roi[i, j]
#                 if k > 100:
#                     k = 1
#                 else:
#                     k = 0
#                 data.append(k)

#         with open('dataset.csv', 'a') as f:
#             writer = csv.writer(f)
#             writer.writerow(data)

# # %%

# # LOADING DATASET

# # 0,....,1.....,2.....
# # 5,3,1,0,2,5,.......

# data = pd.read_csv('dataset.csv')
# data = shuffle(data)
# print(data)

# # %%

# # TRAINING


# X = data.drop(["label"], axis=1)
# Y = data["label"]
# # %%

# idx = 114
# img = X.loc[idx].values.reshape(28, 28)
# print(Y[idx])
# plt.imshow(img)

# # %%

# # TRAIN-TEST SPLIT

# train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

# # %%

# # fit to model

# classifier = SVC(kernel="linear", random_state=6)
# classifier.fit(train_x, train_y)
# joblib.dump(classifier, "svm_model/digit_recognizer")


# # %%
# # CNN

# X_cnn = X.values.reshape(-1, 28, 28, 1)
# Y_cnn = tf.keras.utils.to_categorical(Y, num_classes=10)

# # %%

# # CNN Classifier

# # Build CNN Model
# cnn_model = Sequential([
#     layers.Conv2D(16, kernel_size=(3, 3), activation='relu',
#                   input_shape=(28, 28, 1)),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#     layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

# cnn_model.compile(optimizer='adam',
#                   loss='categorical_crossentropy', metrics=['accuracy'])

# # %%
# # Train CNN Model
# train_x_cnn, test_x_cnn, train_y_cnn, test_y_cnn = train_test_split(
#     X_cnn, Y_cnn, test_size=0.2)

# # %%
# cnn_model.fit(train_x_cnn, train_y_cnn, validation_data=(
#     test_x_cnn, test_y_cnn), epochs=10, batch_size=200)

# # Save CNN Model
# cnn_model.save('models/cnn_self_dataset.keras')

# # %%
# # Evaluate CNN Model
# cnn_loss, cnn_accuracy = cnn_model.evaluate(test_x_cnn, test_y_cnn)
# print(f"CNN Model Accuracy: {cnn_accuracy * 100:.2f}%")

# # %%

# # Evaluate SVM Model

# prediction = classifier.predict(test_x)
# print("SVM Model Accuracy= ", metrics.accuracy_score(prediction, test_y))

# # %%

# # prediction SVM

# # prediction of image drawn in paint


# model = joblib.load("models/svm_self_dataset")
# images_folder = "predictions/"

# while True:
#     time.sleep(8)
#     img = ImageGrab.grab(bbox=(160, 300, 650, 800))

#     img.save(images_folder+"img.png")
#     im = cv2.imread(images_folder+"img.png")
#     im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

#     # Threshold the image
#     ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
#     roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

#     rows, cols = roi.shape

#     X_predict = []

#     # Fill the data array with pixels one by one.
#     for i in range(rows):
#         for j in range(cols):
#             k = roi[i, j]
#             if k > 100:
#                 k = 1
#             else:
#                 k = 0
#             X_predict.append(k)

#     predictions = model.predict([X_predict])
#     print("Prediction:", predictions[0])
#     cv2.putText(im, "Prediction is: " +
#                 str(predictions[0]), (20, 20), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

#     cv2.startWindowThread()
#     cv2.namedWindow("Result")
#     cv2.imshow("Result", im)

#     key = cv2.waitKey(1000)
#     if key == 13:  # 27 is the ascii value of esc, 13 is the ascii value of enter
#         break
# cv2.destroyAllWindows()

# # %%

# # Prediction with CNN model
# cnn_model = tf.keras.models.load_model('models/cnn_self_dataset.keras')
# images_folder = "predictions/"


# def preprocess_image(image_path):
#     im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     im = cv2.GaussianBlur(im, (15, 15), 0)
#     ret, im_th = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
#     roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)
#     roi = roi.reshape(1, 28, 28, 1)
#     roi = roi.astype('float32')
#     roi /= 255
#     return roi


# while True:
#     time.sleep(8)
#     img = ImageGrab.grab(bbox=(160, 300, 650, 800))

#     img.save(images_folder + "img.png")
#     im = cv2.imread(images_folder + "img.png")
#     im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

#     ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
#     roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

#     roi = roi.reshape(1, 28, 28, 1)
#     roi = roi.astype('float32')
#     roi /= 255

#     predictions = cnn_model.predict(roi)
#     predicted_digit = np.argmax(predictions)
#     print("Prediction:", predicted_digit)
#     cv2.putText(im, "Prediction is: " + str(predicted_digit),
#                 (20, 20), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

#     cv2.startWindowThread()
#     cv2.namedWindow("Result")
#     cv2.imshow("Result", im)

#     key = cv2.waitKey(1000)
#     if key == 13:  # 27 is the ascii value of esc, 13 is the ascii value of enter
#         break
# cv2.destroyAllWindows()

# # %%

# # both together

# # Load models
# svm_model = joblib.load("models/svm_self_dataset")
# cnn_model = tf.keras.models.load_model('models/cnn_self_dataset.keras')

# images_folder = "predictions/"


# def preprocess_image(image_path):
#     im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     im = cv2.GaussianBlur(im, (15, 15), 0)
#     ret, im_th = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
#     roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)
#     roi = roi.reshape(1, 28, 28, 1)
#     roi = roi.astype('float32')
#     roi /= 255
#     return roi


# while True:
#     time.sleep(8)
#     img = ImageGrab.grab(bbox=(160, 300, 650, 800))
#     img_path = images_folder + "img.png"
#     img.save(img_path)

#     im = cv2.imread(img_path)
#     im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
#     ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
#     roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

#     # SVM Prediction
#     X_predict = [1 if roi[i, j] >
#                  100 else 0 for i in range(28) for j in range(28)]
#     svm_prediction = svm_model.predict([X_predict])
#     print("SVM Prediction:", svm_prediction[0])

#     # CNN Prediction
#     cnn_roi = preprocess_image(img_path)
#     cnn_predictions = cnn_model.predict(cnn_roi)
#     cnn_predicted_digit = np.argmax(cnn_predictions)
#     print("CNN Prediction:", cnn_predicted_digit)

#     cv2.putText(im, "SVM: " + str(svm_prediction[0]) + " CNN: " + str(cnn_predicted_digit),
#                 (20, 20), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

#     cv2.startWindowThread()
#     cv2.namedWindow("Result")
#     cv2.imshow("Result", im)

#     key = cv2.waitKey(1000)
#     if key == 13:
#         break

# cv2.destroyAllWindows()

# # %%
