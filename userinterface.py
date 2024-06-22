# %%
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import pyscreenshot as ImageGrab
# from PIL importing ImageGrab
from PIL import Image, ImageTk
import time
import os
import joblib
import cv2
import csv
import glob
import numpy as np

#%%
def create_rectangular_button(parent, text, command, x, y, width=400, height=40):
    canvas = tk.Canvas(parent, width=width, height=height, bd=0, highlightthickness=0, relief='ridge')
    canvas.place(x=x, y=y)
    
    canvas.create_rectangle(0, 0, width, height, fill='blue', outline='')
    
    button = tk.Button(parent, text=text, command=command, bg='blue', fg='white', bd=0, highlightthickness=0, activebackground='blue', activeforeground='white', font=('Aerial', 15, 'bold'))
    button_window = canvas.create_window(width / 2, height / 2, window=button)
    return button

def create_styled_input(parent, x, y, width=400, height=40):
    canvas = tk.Canvas(parent, width=width, height=height, bd=0, highlightthickness=0, relief='ridge')
    canvas.place(x=x, y=y)
    
  
    canvas.create_rectangle(0, 0, width, height, fill='blue', outline='')
    
    label = tk.Label(parent, text="Input Digit", font=('Aerial', 15, 'bold'), bg='blue', fg='white')
    label.place(x=x+60, y=y+5)
    
    entry = tk.Entry(parent, width=15, bd=0, font=('Aerial', 15))
    entry.place(x=x+175, y=y+7)
    return entry

#the main window
window = tk.Tk()
window.title("DigitAid-Handwritten Digit Recognition for  the Dislexia")

#the provided image
image_path = "C:/Users/HP/Desktop/ml project/DigitAid/digits_bg.png"  #use the local path
background_image_pil = Image.open(image_path)
background_image = ImageTk.PhotoImage(background_image_pil)

image_width, image_height = background_image_pil.size

window.geometry(f"{image_width}x{image_height}")

#label with the background image
background_label = tk.Label(window, image=background_image)
background_label.place(relwidth=1, relheight=1)

#Main Heading in the Welcome page
heading = tk.Label(window, text="Welcome to DigitAid!", font=('Aerial', 20, 'bold'), bg='lightblue', fg='black')
heading.pack(pady=(20, 10))#padding for x,y axis in heading
#center x position for all buttons
center_x = (image_width - 400) // 2

#for the styled input box
t1 = create_styled_input(window, center_x, 100)

def screen_capture():
    #  /*local path for the paint location in desktop*/
    os.startfile("C:/Users/HP/Desktop/Paint - Shortcut.lnk")
    s1 = t1.get()
    os.chdir("C:/Users/HP/Desktop/ml project/DigitAid/captured_images_ui") 

    if not os.path.exists(s1):
        os.mkdir(s1)

    os.chdir("C:/Users/HP/Desktop/ml project/DigitAid/")

    images_folder = "./captured_images_ui/"+s1+"/"
    time.sleep(15)

    for i in range(0, 5):
        time.sleep(5)
        im = ImageGrab.grab(bbox=(160, 300, 650, 800))
        print("Image Captured", i)
        im.save(images_folder + str(i) + '.png')
        print("Image Saved", i)
    messagebox.showinfo("result", "Capturing Screen is completed.")

create_rectangular_button(window, "Capture Data", screen_capture, center_x, 150)

def generate_dataset():
    header = ["label"]
    for i in range(0, 784):
        header.append("pixel"+str(i))
    with open('newdataset.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for label in range(10):
        dirList = glob.glob("./captured_images_ui/"+str(label)+"/*.png")

        for img_path in dirList:
            im = cv2.imread(img_path)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
            roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

            data = []
            data.append(label)
            rows, cols = roi.shape

            for i in range(rows):
                for j in range(cols):
                    k = roi[i, j]
                    if k > 100:
                        k = 1
                    else:
                        k = 0
                    data.append(k)
            with open('newdataset.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)
    messagebox.showinfo("Result", "Generating dataset is completed.")

create_rectangular_button(window, "Generate Dataset", generate_dataset, center_x, 200)

def open_train_model_window():
    train_window = tk.Toplevel(window)
    train_window.title("Select Model to Train")
    train_window.geometry(f"{image_width}x{image_height}")

    train_background_label = tk.Label(train_window, image=background_image)
    train_background_label.place(relwidth=1, relheight=1)
    
    #Heading for the train model window
    train_heading = tk.Label(train_window, text="Select Model to Train", font=('Aerial', 20, 'bold'), bg='lightblue', fg='black')
    train_heading.pack(pady=(20, 10))

    def train_svm():
        import pandas as pd
        from sklearn.utils import shuffle

        data = pd.read_csv('dataset.csv')
        data = shuffle(data)

        X = data.drop(["label"], axis=1)
        Y = data["label"]

        from sklearn.model_selection import train_test_split

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

        from sklearn.svm import SVC

        classifier = SVC(kernel="linear", random_state=6)
        classifier.fit(train_x, train_y)
        joblib.dump(classifier, "models/svm_digit_recognizer")

        from sklearn import metrics
        prediction = classifier.predict(test_x)
        acc = metrics.accuracy_score(prediction, test_y)
        messagebox.showinfo("Result", f"Accuracy = {acc}")

    create_rectangular_button(train_window, "Train SVM Model(Self-Made Dataset)", train_svm, center_x, 100)

    def train_cnn():
        messagebox.showinfo("Info", "CNN training functionality not implemented yet.")

    create_rectangular_button(train_window, "Train CNN Model(Self-Made Dataset)", train_cnn, center_x, 150)

    # def train_other_model():
    #     messagebox.showinfo("Info", "Other model training functionality not implemented yet.")

    # create_rectangular_button(train_window, "Train Other Model", train_other_model, center_x, 200)

create_rectangular_button(window, "Train Model", open_train_model_window, center_x, 250)

def open_live_prediction_window():
    prediction_window = tk.Toplevel(window)
    prediction_window.title("Select Prediction Method")
    prediction_window.geometry(f"{image_width}x{image_height}")

    prediction_background_label = tk.Label(prediction_window, image=background_image)
    prediction_background_label.place(relwidth=1, relheight=1)
    
    #heading for the live prediction window
    prediction_heading = tk.Label(prediction_window, text="Select Prediction Method", font=('Aerial', 20, 'bold'), bg='lightblue', fg='black')
    prediction_heading.pack(pady=(20, 10))

    def predict_self_made_svm():
        os.startfile("C:/Users/HP/Desktop/Paint - Shortcut.lnk")
        model = joblib.load("models/svm_digit_recognizer")
        images_folder = "predictions/"

        time.sleep(15)

        while True:
            time.sleep(8)
            img = ImageGrab.grab(bbox=(160, 300, 650, 800))

            img.save(images_folder+"img.png")
            im = cv2.imread(images_folder+"img.png")
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

            ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
            roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

            rows, cols = roi.shape

            X_predict = []

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
            if key == 13:
                break
        cv2.destroyAllWindows()

    def predict_mnist_svm():
        messagebox.showinfo("Info", "MNIST-SVM prediction functionality not implemented yet.")

    create_rectangular_button(prediction_window, "Using (Self-Made SVM)", predict_self_made_svm, center_x, 100)
    create_rectangular_button(prediction_window, "Using MNIST-SVM", predict_mnist_svm, center_x, 150)

create_rectangular_button(window, "Live prediction", open_live_prediction_window, center_x, 350)

#reference to the background image to avoid garbage collection
window.background_image = background_image

window.mainloop()
# %%
