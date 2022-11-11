import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# First, load the marriage data
marriage = pd.read_csv('data/marriage.csv', header=None).to_numpy()

# Divide the marriage data into data portion and label portion, and scale the data portion
data = marriage[:,:-1]
scale = StandardScaler()
data = scale.fit_transform(data)
target = marriage[:,-1]

# Divide the data and label into training and test set
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=10)
print('-Count of each label in test set: \n', np.unique(test_target, return_counts=True)) # just to check if the labels are shuffled well

# Train Naive Bayes classifier and make predictions
nb = GaussianNB()
nb_pred = nb.fit(train_input, train_target).predict(test_input)

# Train Logistic Regression classifier and make predictions
lr = LogisticRegression()
lr_pred = lr.fit(train_input, train_target).predict(test_input)

# Train KNN classifier and make predictions
knn = KNeighborsClassifier()
knn_pred = knn.fit(train_input, train_target).predict(test_input)

# Calculate accuracy of each classifier
nb_accuracy = sum(test_target == nb_pred) / len(test_target) 
lr_accuracy = sum(test_target == lr_pred) / len(test_target)
knn_accuracy = sum(test_target == knn_pred) / len(test_target)

print('-Accuracy of Naive Bayes Classifier: ', nb_accuracy * 100, '%', 
'\n-Accuracy of Logistic Regression Classifier: ', lr_accuracy * 100, '%', 
'\n-Accuracy of KNN Classifier: ', knn_accuracy * 100, '%')


########## Below, we will test different random seeds, to make sure all three models' predictions are exactly the same #########
seeds = list(range(20))
nb_acc = [] # empty list to contain accuracy at each random seed
lr_acc = []
knn_acc = []
for i in seeds:
    train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=i)

    nb = GaussianNB()
    nb_pred = nb.fit(train_input, train_target).predict(test_input)

    lr = LogisticRegression()
    lr_pred = lr.fit(train_input, train_target).predict(test_input)

    knn = KNeighborsClassifier()
    knn_pred = knn.fit(train_input, train_target).predict(test_input)

    nb_acc.append(sum(test_target == nb_pred) / len(test_target) )
    lr_acc.append(sum(test_target == lr_pred) / len(test_target))
    knn_acc.append(sum(test_target == knn_pred) / len(test_target))

accuracy = [nb_acc, lr_acc, knn_acc]
titles = ['Naive Bayes', 'Logistic Regression', 'KNN']

# Plot the accuracy at each random see per model
fig, axs = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle('Accuracy at Each Random Seed', fontweight='bold', color='orange')
for i in range(3):
    axs[i].plot(seeds, accuracy[i])
    axs[i].set_xlabel('Random Seed')
    axs[i].set_ylabel('Accuracy')
    axs[i].set_title(titles[i], y=-0.01)
plt.show()



################## Below, we will perform PCA first, and then modeling ##################
# Perform PCA on the data portion
m, n = data.shape
C = np.matmul(data.T, data)/m
d = 2
U, _, _ = np.linalg.svd(C)
U = U[:, :d]

pdata = np.dot(data, U) # project the data onto 2 principal components

ptrain_input, ptest_input, ptrain_target, ptest_target = train_test_split(pdata, target, test_size=0.2, random_state=42)

# make predictions using PCA results
pnb_pred = nb.fit(ptrain_input, ptrain_target).predict(ptest_input)
plr_pred = lr.fit(ptrain_input, ptrain_target).predict(ptest_input)
pknn_pred = knn.fit(ptrain_input, ptrain_target).predict(ptest_input)


# Plot Naive Bayes Classifier on PCA results
colors = ['navy', 'salmon']
lw = 2

h = 0.01
cmap_light = ListedColormap(['#FFAAAA',  '#AAAAFF'])

x_min, x_max = ptrain_input[:,0].min(), ptrain_input[:,0].max() 
y_min, y_max = ptrain_input[:,1].min(), ptrain_input[:,1].max()

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

no_div = plt.scatter(ptest_input[pnb_pred == 0, 0], ptest_input[pnb_pred == 0, 1], color=colors[0])
div = plt.scatter(ptest_input[pnb_pred == 1, 0], ptest_input[pnb_pred == 1, 1], color=colors[1])

plt.legend([no_div, div], ['label0; no divorce', 'label1; divorce'], loc='best', shadow=False, scatterpoints=1)
plt.title('Naive Bayes Classifier on PCA results')

plt.show()

# Plot Logistic Regression on PCA results
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

no_div = plt.scatter(ptest_input[plr_pred == 0, 0], ptest_input[plr_pred == 0, 1], color=colors[0])
div = plt.scatter(ptest_input[plr_pred == 1, 0], ptest_input[plr_pred == 1, 1], color=colors[1])

plt.legend([no_div, div], ['label0; no divorce', 'label1; divorce'], loc='best', shadow=False, scatterpoints=1)
plt.title('Logistic Regression on PCA results')

plt.show()

# Plot KNN  on PCA results
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

no_div = plt.scatter(ptest_input[pknn_pred == 0, 0], ptest_input[pknn_pred == 0, 1], color=colors[0])
div = plt.scatter(ptest_input[pknn_pred == 1, 0], ptest_input[pknn_pred == 1, 1], color=colors[1])

plt.legend([no_div, div], ['label0; no divorce', 'label1; divorce'], loc='best', shadow=False, scatterpoints=1)
plt.title('KNN on PCA results')

plt.show()


