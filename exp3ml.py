#!/usr/bin/env python
# coding: utf-8

# # EXPERIMENT NO. 3. 
# ##SIMPLE LIEAR REGRATION 

# # IMPORT THE DATA FILE FROM PC AND PLOT ON MATPLOT 

# In[12]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"]=(6.0, 4.0)


data =pd.read_csv('Salary_Data.csv')
x=data.iloc[:,0]
y=data.iloc[:,1]
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('salary')
plt.title('Traning Data')
plt.show()


# # INITIALIZAYION OF VALUES 

# In[13]:



theta0 = 1.86938628297
theta1 = 50.011412628146

alpha = 0.0001 

m = float(len(X))# no of samples in data set get it
epsilon = 0.00001
CostOld = 9999
diff = 1
iteration = 1


# # PREDICTING THE HOYOTHESIS 

# In[14]:



while abs(diff) > epsilon:          #diff is toooo small then stop itteration
                                    #automatic convergence test ACT some threshold should be  there .. that is epsilon
    Y_pred = theta1 * X + theta0    #h theta(x)=o0+o1x) and prediction for all vector X samples
    Cost = (1 / m) * sum((Y_pred - Y) ** 2)#X&Y capital is all vector not single entity
    #   print(Cost)

    D_theta1 = (1 / m) * sum(X * (Y_pred - Y))# partial derivative wrt to theta1
    D_theta0 = (1 / m) * sum(Y_pred - Y)
    theta1 = theta1 - alpha * D_theta1
    theta0 = theta0 - alpha * D_theta0
    #print(cost)
    diff = CostOld - Cost
    #   print(diff)

    CostOld = Cost
    iteration = iteration + 1
    # plt.scatter(X,Y)
    # plt.plot([min(X), max(X)],[min(Y_pred),max (Y_pred)], color='red')
    # plt.show()




# # PRINTING THE PLOTS OF PREDICTED HYPOTHESIS

# In[15]:


print('Iteration = ', str(iteration))
print("theta0 = ", theta0, ",theta1 = ", theta1)
    

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting the Linear Hypothesis')
plt.show()


test_X =float(input ("enter no of year "))#1.2 #initializing the value of (x theta1 *x)
Y_pred = theta1 * test_X + theta0
print('salary prediction for exprience of ', str(test_X), 'year = ', round(Y_pred,2))


# # COMPARING WITH ACTUAL VALUES THE PREDICTED VALUE 

# In[ ]:


data
print('salary preducted for ',test_X 'years is ',Y_pred)


# In[ ]:




