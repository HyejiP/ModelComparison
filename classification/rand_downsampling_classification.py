import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from random import sample
from sklearn import metrics 
import matplotlib.pyplot as plt

# Load the data
mat = scipy.io.loadmat('data/mnist_10digits.mat')
xtrain = mat['xtrain']
ytrain = np.ravel(mat['ytrain'])
xtest = mat['xtest']
ytest = np.ravel(mat['ytest'])

# Scale the data so that the values range [0,1] instead of [0,255]
xtrain = xtrain / 255
xtest = xtest / 255

# Randomly downsample the training set (for KNN and SVM fitting)
inds = sample(range(len(xtrain)), 5000)
small_xtrain = xtrain[inds]
small_ytrain = ytrain[inds]


# Tune K(# of neighbors) for KNN model
k_s = list(range(1, 15))
accuracy = []
for i in k_s:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn_pred = knn.fit(small_xtrain, small_ytrain).predict(xtest)
    accuracy.append(metrics.accuracy_score(knn_pred, ytest))

plt.plot(k_s, accuracy, linestyle='dashed', marker='o', markerfacecolor='red')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy by Number of Neighbors')
plt.show()

best_k = k_s[np.argmax(accuracy)]
print('best k for KNN model: ', best_k)

# Fit the KNN model with the best parameter K and make predictions on 'xtest'
knn = KNeighborsClassifier(n_neighbors=best_k)
knn_pred = knn.fit(small_xtrain, small_ytrain).predict(xtest)
print('**** Analysis on the KNN model ****')
print('\n-Confusion Matrix: \n', confusion_matrix(ytest, knn_pred))
print('\n-Precision, Recall, F1-score: \n', classification_report(ytest, knn_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

# Fit Logistic Regression model and make predictions on 'xtest'
lr = LogisticRegression(max_iter=1000)
lr_pred = lr.fit(xtrain, ytrain).predict(xtest)
print('**** Analysis on the Logistic Regression model ****')
print('\n-Confusion Matrix: \n', confusion_matrix(ytest, lr_pred))
print('\n-Precision, Recall, F1-score: \n', classification_report(ytest, lr_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

# Fit simple SVM model and make predictions on 'xtest'
svm = SVC(kernel='linear')
svm_pred = svm.fit(small_xtrain, small_ytrain).predict(xtest)
print('**** Analysis on the simple SVM model ****')
print('\n-Confusion Matrix: \n', confusion_matrix(ytest, svm_pred))
print('\n-Precision, Recall, F1-score: \n', classification_report(ytest, svm_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

# Fit kernel SVM model and make predictions on 'xtest'
kern_svm = SVC(kernel='rbf')
kern_svm_pred = kern_svm.fit(small_xtrain, small_ytrain).predict(xtest)
print('**** Analysis on the kernel SVM model ****')
print('\n-Confusion Matrix: \n', confusion_matrix(ytest, kern_svm_pred))
print('\n-Precision, Recall, F1-score: \n', classification_report(ytest, kern_svm_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

# Fit Neural Network model and make predictions on 'xtest'
neural = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000)
neural_pred = neural.fit(xtrain, ytrain).predict(xtest)
print('**** Analysis on the Neural Network model ****')
print('\n-Confusion Matrix: \n', confusion_matrix(ytest, neural_pred))
print('\n-Precision, Recall, F1-score: \n', classification_report(ytest, neural_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))