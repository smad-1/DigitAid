# # %%
# from sklearn.utils import shuffle
# from sklearn import metrics
# import joblib
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # %%

# # load data

# d = pd.read_csv("./mnist/mnist_train.csv")

# print(d.head(5))

# # %%

# Y = d['label']

# X = d.drop("label", axis=1)

# print(X.head(5))

# # %%

# idx = 114
# img = X.loc[idx].values.reshape(28, 28)
# print(Y[idx])
# plt.imshow(img)

# # %%

# # train

# train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
# # %%

# # fit to model
# classifier = SVC(kernel="linear", random_state=6, verbose=True)
# classifier.fit(train_x, train_y)
# joblib.dump(classifier, "models/svm_mnist")


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

# %%

# LOAD DATA

train = pd.read_csv('./mnist/mnist_train.csv')
test = pd.read_csv('./mnist/mnist_test.csv')

train.head()

# %%
test.head()
# %%

train_shuffled = shuffle(train.values, random_state=0)
X = train.drop(labels=["label"], axis=1)
y = train["label"]
# X_test = test.values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0)

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
normalized_X_val = scaler.transform(X_val)

# %%

# PCA to reduce dimensions

pca = PCA(n_components=0.90)
pca_X_train = pca.fit_transform(normalized_X_train)
pca_X_val = pca.transform(normalized_X_val)
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
joblib.dump((classifier, pca), "models/svm_mnist")

train_accuracy = classifier.score(pca_X_train, y_train)
print(f"Training Accuracy: {train_accuracy*100:.3f}%")

# %%

# prediction

predictions = classifier.predict(pca_X_val)
print("Accuracy= ", metrics.accuracy_score(predictions, y_val))

# %%
