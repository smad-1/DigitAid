
# DigitAid: Handwritten Digit Recognition for Dyslexia

DigitAid is an innovative tool designed to assist individuals with dyslexia by providing reliable handwritten digit recognition. This project integrates data capture, dataset generation, model training, and live prediction functionalities to improve educational accessibility for dyslexic learners.

## Introduction

Dyslexia is a neurological disorder that affects reading, writing, and spelling skills, impacting approximately 10% of the global population. DigitAid aims to alleviate some of the challenges faced by dyslexic individuals by providing a user-friendly tool for recognizing handwritten digits, which can be integrated into educational technologies to support dyslexic learners.
## Features

- Data Capture: Easily capture handwritten digit samples using a virtual canvas.
- Dataset Generation: Process and generate datasets from captured images for training machine learning models.
- Model Training: Train Support Vector Machine (SVM) and Convolutional Neural Network (CNN) models using the generated datasets.
-Live Prediction: Perform real-time predictions on new handwritten digit samples using trained models.


## Installation

To install and set up DigitAid, follow these steps:

1.Clone the repository
```bash
  git clone https://github.com/your-username/DigitAid.git
  cd DigitAid

```
2.Install the dependencies
```bash
These are the Dependencies that you need to install from your terminal-

tkinter (Standard Library, typically pre-installed with Python)
pyscreenshot
Pillow
os 
joblib
cv2
csv 
glob 
numpy
tensorflow
scikit-learn
pandas
```
3.Install Git LFS:
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
git commit -m "Track large files with Git LFS"

```
4.Run the Application:
```bash
python userinterface.py

```
    
## How To Use
- Capture Data
    1. Input the digit in the input box for which you want to provide your data 
    2. Click the "Capture Data" button.
    3. Draw digits (0-9) on the Paint application canvas.(Keep the canvas color black and use the white oil brush to write the digit)
    4. The system will capture and save the drawn digits as images.


- Generate Dataset
    1. Click the "Generate Dataset" button.
    2. The system processes the captured images, resizing them to 28x28 pixels and normalizing the pixel values.
    3. The processed data is saved in CSV format for model training.

- Train Model
    1. Click the "Train Model" button.
    2. Select the type of model to train: SVM or CNN.
    3. The system trains the selected model using the generated dataset and displays the model's accuracy.

    
- Live Prediction

    1. Click the "Live Prediction" button.
    2. Draw new digits on the Paint application canvas     (Make sure to keep the paint tab in the left side of your screen and like the capture data keep the canvas and paint mode same)
    3. The system captures the new drawings, processes the images, and uses the trained models to predict the digits in real-time, displaying the predictions.

   

   
