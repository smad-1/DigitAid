# %%

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

# %%

# LOAD DATA

train = pd.read_csv('./mnist2/mnist_train.csv')
test = pd.read_csv('./mnist2/mnist_test.csv')

train.head()

# %%
test.head()

# %%

# Binarize images


def binarize_image(data, threshold=128):
    binarized_data = np.where(data > threshold, 255, 0)
    return binarized_data


# Binarize the training and testing images
X_train = binarize_image(train.drop(labels=["label"], axis=1))
y_train = train["label"]
X_test = binarize_image(test.drop(labels=["label"], axis=1))
y_test = test["label"]

# %%
# Shuffle the data
train_shuffled = shuffle(train.values, random_state=0)
test_shuffled = shuffle(test.values, random_state=0)
# %%
# Reshape the data to match the original structure
X_train = pd.DataFrame(X_train, columns=train.columns[1:])
y_train = train["label"]
X_test = pd.DataFrame(X_test, columns=test.columns[1:])
y_test = test["label"]

# print(f'X_train = {X_train.shape}, y =
# {y_train.shape}, X_test = {X_test.shape}')

# %%

# Visualize some digits

plt.figure(figsize=(14, 12))
for digit_num in range(0, 30):
    plt.subplot(7, 10, digit_num+1)
    grid_data = X_train.iloc[digit_num].values.reshape(28, 28)
    plt.imshow(grid_data, interpolation="none", cmap="afmhot")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()

# %%

sns.set(style="darkgrid")
counts = sns.countplot(x="label", data=train, palette="Set1")

# %%

# NORMALIZING DATA

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train)
normalized_X_train = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)

# %%

# PCA to reduce dimensions

pca = PCA(n_components=0.90)
pca_X_train = pca.fit_transform(normalized_X_train)
pca_X_test = pca.transform(normalized_X_test)
# print(f'{pca.explained_variance_} \n Number of PCA Vectors = {
#       len(pca.explained_variance_)}')

# %%

# plotting PCA output

f, ax = plt.subplots(1, 1)
for i in range(10):
    ax.scatter(pca_X_train[y_train == i, 0],
               pca_X_train[y_train == i, 1], label=i)
ax.set_xlabel("PCA Analysis")
ax.legend()
f.set_size_inches(16, 6)
ax.set_title("Digits (training set)")
plt.show()

# %%
# train (using predefined gamma and c)

classifier = svm.SVC(gamma=0.00728932024638, C=2.82842712475)
classifier.fit(pca_X_train, y_train)

# joblib.dump(classifier, "models/svm_mnist")

# Save the model and PCA object
joblib.dump((classifier, pca, scaler), "models/svm_mnist")

train_accuracy = classifier.score(pca_X_train, y_train)
print(f"Training Accuracy: {train_accuracy*100:.3f}%")

# %%

# prediction

predictions = classifier.predict(pca_X_test)
accuracy = metrics.accuracy_score(predictions, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
report = classification_report(y_test, predictions)
print(report)

# %%
# Plotting Accuracy
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter(y_test, predictions, color='blue', alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('SVM Predictions vs True Values')
plt.show()

# %%
# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %%
