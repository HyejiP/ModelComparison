############## Below is code for calculating mutual information ###############
import math



mi_prize = (150/16160) * math.log((16160*150)/((150+10)*(150+1000)), 2) + (1000/16160) * math.log((16160*1000)/((1000+15000)*(1000+150)), 2) + (10/16160) * math.log((16160*10)/((150+10)*(10+15000)), 2) + (15000/16160) * math.log((16160*15000)/((1000+15000)*(10+15000)), 2)
mi_hello = (155/16160) * math.log((16160*155)/((155+5)*(155+14000)), 2) + (14000/16160) * math.log((16160*14000)/((14000+2000)*(14000+155)), 2) + (5/16160) * math.log((16160*5)/((155+5)*(5+2000)), 2) + (2000/16160) * math.log((16160*2000)/((14000+2000)*(5+2000)), 2)

print('-Mutual Information of "prize" is: ', mi_prize)
print('-Mutual Information of "hello" is: ', mi_hello)

######################### Below is CUSUM statistic  ########################
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(30)
f0_set = np.random.normal(0, 1, size=100)
f1_set = np.random.normal(1.5, np.sqrt(1.1), size=100) 
seq = np.append(f0_set, f1_set)

ratio = []
w_old = 0 
# calculate log-likelihood ratio and compute W_t
for i in range(len(seq)):
    w_new = max((w_old + np.log(1/np.sqrt(1.1)) - (((seq[i]-1.5)**2)/(2*1.1)) + ((seq[i]**2)/2)), 0)
    ratio.append(w_old)   
    w_old = w_new

# plot original sequential data and CUSUM statistics
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(seq)
axs[0].set_title('Sequential Data')
axs[1].plot(ratio)
axs[1].axhline(y=5, color='r', linestyle='-')
axs[1].set_title('CUSUM Detection')
plt.show()

