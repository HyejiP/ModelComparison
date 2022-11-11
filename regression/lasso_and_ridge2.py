import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#################### Load data, generate matrices ##########################
# Load the data
mat = scipy.io.loadmat('data/cs.mat')
img = mat['img']

plt.imshow(img, cmap='gray_r')
plt.title('True Image')
plt.show()

# transform image to a column vector
img_vec = img.reshape(50*50, -1)

# compose a matrix of weights of random iid variable ~ N(0,1) 
np.random.seed(30)
A_set = np.random.normal(0, 1, size=1300*2500)
A = A_set.reshape(1300, 2500)

# compose a column vector of error terms  
np.random.seed(30)
eps_set = np.random.normal(0, 5, size=1300)
eps = eps_set.reshape(1300, -1)

y = np.dot(A, img_vec) + eps

print('Shape of A: ', A.shape)
print('Shape of eps: ',eps.shape)
print('Shape of y: ',y.shape)

#################### Perform LASSO Regression ##########################
lasso_mse = [] # an empty list to contain mse of LASSO at each alpha value
lasso_alphas = np.linspace(0.01, 1, num=100)
for i in lasso_alphas:
    lasso = Lasso(alpha=i, max_iter=3000)
    lasso.fit(A, y)
    cv = KFold(n_splits=10, random_state=3, shuffle=True)
    scores = cross_val_score(lasso, A, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    lasso_mse.append(np.mean(np.absolute(scores)))

best_alpha2 = lasso_alphas[np.argmin(lasso_mse)]
print('-Best Alpha for LASSO Model: ', best_alpha2)
final_lasso = Lasso(alpha=best_alpha2, max_iter=3000)
final_lasso.fit(A, y)

re_img_lasso = final_lasso.coef_.reshape(50, 50)

# plot CV curve and the reconstructed image
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('<LASSO>', fontsize=16, color='orange')
axs[0].plot(lasso_alphas, lasso_mse)
axs[0].set_title('MSE by Regularization Parameters')
axs[1].imshow(re_img_lasso, cmap='gray_r')
axs[1].set_title('Reconstructed Image')
plt.show()


#################### Perform Ridge Regression ##########################
ridge_mse= [] # an empty list to contain mse of Ridge at each alpha value
ridge_alphas = range(1, 80+1)
for i in ridge_alphas:
    ridge = Ridge(alpha=i)
    ridge.fit(A, y)
    cv = KFold(n_splits=10, shuffle=True, random_state=2)
    scores = cross_val_score(ridge, A, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    ridge_mse.append(np.mean(np.absolute(scores)))

best_alpha = ridge_alphas[np.argmin(ridge_mse)]
print('-Best Alpha for Ridge Model: ', best_alpha)
final_ridge = Ridge(alpha=best_alpha)
final_ridge.fit(A, y)

re_img_ridge = final_ridge.coef_.reshape(50, 50)

# plot CV curve and the reconstructed image
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('<Ridge>', fontsize=16, color='orange')
axs[0].plot(ridge_alphas, ridge_mse)
axs[0].set_title('MSE by Regularization Parameters')
axs[1].imshow(re_img_ridge, cmap='gray_r')
axs[1].set_title('Reconstructed Image')
plt.show()

