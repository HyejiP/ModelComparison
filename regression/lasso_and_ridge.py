import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import random

# load the data
house = pd.read_csv('data/RealEstate.csv')
house = house.drop(['MLS', 'Location'], axis=1) # remove 'MLS' and 'Location' features
status_dummies = pd.get_dummies(house.Status, prefix='Status', drop_first=True) # perform one-hot encoding for 'Status' feature
house = pd.merge(left=house, right=status_dummies, left_index=True, right_index=True) 
house = house.drop(['Status'], axis=1)

# check correlation between features
data = house.drop('Price', axis=1)
corr = data.corr()
print('---Correlation Between Features---\n', corr)

# make data and actual price into numpy array
data = data.to_numpy()
target = house['Price'].to_numpy()

# scale the data with each MinMaxScaler() and StandardScaler()
mm_scaler = MinMaxScaler()
mm_scaler.fit(data)
data_mm_scaled = mm_scaler.transform(data)

std_scaler = StandardScaler()
std_scaler.fit(data)
data_std_scaled = std_scaler.transform(data)


#################### Below we will perform Ridge Regression ##########################
ridge_mse= [] # an empty list to contain mse of Ridge at each alpha value
ridge_coefs = [] # an empty list to contain coefficients of Ridge at each alpha value
for i in range(1, 80+1):
    ridge = Ridge(alpha=i)
    ridge.fit(data_std_scaled, target)
    ridge_coefs.append(ridge.coef_)
    cv = KFold(n_splits=5, shuffle=True, random_state=2)
    scores = cross_val_score(ridge, data_std_scaled, target, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    ridge_mse.append(np.mean(np.absolute(scores)))

ridge_coefs = np.array(ridge_coefs)

# plot MSE at different alpha values and solution path
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('<Ridge>', fontsize=16, color='orange')
axs[0].plot(range(1, 80+1), ridge_mse)
axs[0].set_title('MSE by Regularization Parameters')
for i in range(6):
    axs[1].plot(ridge_coefs[:,i])
axs[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axs[1].set_title('Solution Path')
axs[1].legend(['bedrooms', 'bathrooms', 'size', 'price/Sq.ft', 'Status_regular', 'Status_shortsale'])
plt.show()

best_alpha = np.argmin(ridge_mse)+1
final_ridge = Ridge(alpha=best_alpha)
final_ridge.fit(data_std_scaled, target)
ridge_pred = final_ridge.predict(data_std_scaled)
ssr_ridge = np.sum((target - ridge_pred)**2)

print('-Best Alpha for Ridge: \n', best_alpha)
print('-Coefficeints of Final Ridge Model: \n', final_ridge.coef_)
print('-Sum of Squared Residuals of Final Ridge Model: \n', ssr_ridge)


#################### Below we will perform LASSO Regression ##########################
lasso_mse = [] # an empty list to contain mse of LASSO at each alpha value
lasso_coefs = [] # an empty list to contain coefficients of LASSO at each alpha value
for i in range(1, 3000+1):
    lasso = Lasso(alpha=i, max_iter=2000)
    lasso.fit(data_mm_scaled, target)
    lasso_coefs.append(lasso.coef_)
    cv = KFold(n_splits=5, random_state=3, shuffle=True)
    scores = cross_val_score(lasso, data_mm_scaled, target, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    lasso_mse.append(np.mean(np.absolute(scores)))

lasso_coefs = np.array(lasso_coefs)

# plot MSE at different alpha values and solution path
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('<LASSO>', fontsize=16, color='orange')
axs[0].plot(range(1, 3000+1), lasso_mse)
axs[0].set_title('MSE by Regularization Parameters')
for i in range(6):
    axs[1].plot(lasso_coefs[:,i])
axs[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axs[1].set_title('Solution Path')
axs[1].legend(['bedrooms', 'bathrooms', 'size', 'price/Sq.ft', 'Status_regular', 'Status_shortsale'])
plt.show()

best_alpha2 = np.argmin(lasso_mse)+1
final_lasso = Lasso(alpha=best_alpha2)
final_lasso.fit(data_mm_scaled, target)
lasso_pred = final_lasso.predict(data_mm_scaled)
ssr_lasso = np.sum((target - lasso_pred)**2)

print('-Best Alpha for LASSO: \n', best_alpha2)
print('-Coefficeints of Final LASSO Model: \n', final_lasso.coef_)
print('-Sum of Squared Residuals of Final LASSO Model: \n', ssr_lasso)

