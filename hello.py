# DATA COLLECTION FROM PAINT

# %%
import numpy as np  # pip install numpy
from sklearn import metrics
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle  # pip install scikit-learn
import pandas as pd  # pip install pandas
import cv2
import csv
import glob
import pyscreenshot as ImageGrab
import time


# %%


def take_screenshot(digit, images_folder):
    for i in range(70, 90):
        time.sleep(3)
        im = ImageGrab.grab(bbox=(160, 300, 650, 800))  # x1,y1,x2,y2
        print("Image Captured", i)
        im.save(images_folder + digit + str(i) + '.png')
        print("Image Saved", i)


# %%


# Digit 0

# images_folder = "./screenshots/zero/"
# digit = "zero"
# take_screenshot(digit, images_folder)
# %%

# Digit 1
# images_folder = "./screenshots/one/"
# digit = "one"
# take_screenshot(digit, images_folder)

# %%
# Digit 2
# images_folder = "./screenshots/two/"
# digit = "two"
# take_screenshot(digit, images_folder)

# %%
# Digit 3
# images_folder = "./screenshots/three/"
# digit = "three"
# take_screenshot(digit, images_folder)

# %%
# Digit 4
# images_folder = "./screenshots/four/"
# digit = "four"
# take_screenshot(digit, images_folder)

# %%
# Digit 5
# images_folder = "./screenshots/five/"
# digit = "five"
# take_screenshot(digit, images_folder)
# %%
# Digit 6
# images_folder = "./screenshots/six/"
# digit = "six"
# take_screenshot(digit, images_folder)

# %%
# Digit 7
# images_folder = "./screenshots/seven/"
# digit = "seven"
# take_screenshot(digit, images_folder)

# %%
# Digit 8
# images_folder = "./screenshots/eight/"
# digit = "eight"
# take_screenshot(digit, images_folder)

# %%

# Digit 9
# images_folder = "./screenshots/nine/"
# digit = "nine"
# take_screenshot(digit, images_folder)

# %%

# GENERATE DATASET FROM IMAGES - DO NOT EXECUTE/RUN THIS CELL THIS AGAIN!!


# 0 -> background, 1 -> digit

header = ["label"]
for i in range(0, 784):
    header.append("pixel" + str(i))

with open('dataset.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(header)

folders = ['zero', 'one', 'two', 'three', 'four',
           'five', 'six', 'seven', 'eight', 'nine']

for label in range(10):
    dirList = glob.glob("./screenshots/" + folders[label] + "/*.png")

    for img_path in dirList:
        im = cv2.imread(img_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # converts to grayscale
        # blurs image to make it smooth
        im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
        # resize to 28x28
        roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

        data = []
        data.append(label)
        rows, cols = roi.shape

        # Fill the data array with pixels one by one.
        for i in range(rows):
            for j in range(cols):
                k = roi[i, j]
                if k > 100:
                    k = 1
                else:
                    k = 0
                data.append(k)

        with open('dataset.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)

# %%

# LOADING DATASET

# 0,....,1.....,2.....
# 5,3,1,0,2,5,.......

data = pd.read_csv('dataset.csv')
data = shuffle(data)
print(data)

# %%

# TRAINING

X = data.drop(["label"], axis=1)
Y = data["label"]
# %%

idx = 114
img = X.loc[idx].values.reshape(28, 28)
print(Y[idx])
plt.imshow(img)

# %%

# TRAIN-TEST SPLIT

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

# %%

# fit to model

classifier = SVC(kernel="linear", random_state=6)
classifier.fit(train_x, train_y)
joblib.dump(classifier, "svm_model/digit_recognizer")


# %%

# CNN Classifier

# %%

# accuracy

prediction = classifier.predict(test_x)
print("Accuracy= ", metrics.accuracy_score(prediction, test_y))

# %%

# prediction

# prediction of image drawn in paint


model = joblib.load("svm_model/digit_recognizer")
images_folder = "predictions/"

while True:
    time.sleep(8)
    img = ImageGrab.grab(bbox=(160, 300, 650, 800))

    img.save(images_folder+"img.png")
    im = cv2.imread(images_folder+"img.png")
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
    roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

    rows, cols = roi.shape

    X_predict = []

    # Fill the data array with pixels one by one.
    for i in range(rows):
        for j in range(cols):
            k = roi[i, j]
            if k > 100:
                k = 1
            else:
                k = 0
            X_predict.append(k)

    predictions = model.predict([X_predict])
    print("Prediction:", predictions[0])
    cv2.putText(im, "Prediction is: " +
                str(predictions[0]), (20, 20), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.startWindowThread()
    cv2.namedWindow("Result")
    cv2.imshow("Result", im)

    key = cv2.waitKey(1000)
    if key == 13:  # 27 is the ascii value of esc, 13 is the ascii value of enter
        break
cv2.destroyAllWindows()

# %%
