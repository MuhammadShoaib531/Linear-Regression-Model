#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ## About Dataset
# 
# Content: 
# Each record in the database describes a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. The attributes are deﬁned as follows (taken from the UCI Machine Learning Repository1): CRIM: per capita crime rate by town
# 
# 1. CRIM: Per capita crime rate by town  
# 2. ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.  
# 3. INDUS: Proportion of non-retail business acres per town  
# 4. CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)  
# 5. NOX: Nitric oxides concentration (parts per 10 million)  
# 6. RM: Average number of rooms per dwelling  
# 7. AGE: Proportion of owner-occupied units built prior to 1940  
# 8. DIS: Weighted distances to five Boston employment centers  
# 9. RAD: Index of accessibility to radial highways  
# 10. TAX: Full-value property-tax rate per $10,000
# 
# 11. PTRATIO: Pupil-teacher ratio by town  
# 12. B: 1000(Bk−0.63)², where Bk is the proportion of blacks by town  
# 13. LSTAT: % lower status of the population  
# 14. MEDV: Median value of owner-occupied homes in $1000s  

# In[2]:


# Load the dataset
insurance_data = pd.read_excel('E:\\ML\\class assignmet 1 ML\\insurance.xlsx')
# copy of the data 
data_copy= insurance_data.copy()

# Display the first few rows of the dataset to explore its structure
insurance_data.head()


# In[3]:


def data_summary(df):
    print(insurance_data.info())
    print("...........end...............")
    print(insurance_data.describe())
    print("...........end...............")
    print(insurance_data.duplicated().sum())
    print("...........end...............")
    print(insurance_data.isnull().sum())
data_summary(insurance_data)


# In[4]:


# drop duplicate 
insurance_data.drop_duplicates()


# In[5]:


# Encoding categorical variables: 'sex', 'smoker', 'region'
insurance_data_encoded = pd.get_dummies(insurance_data, columns=['sex', 'smoker', 'region'],dtype= int)

# Displaying the first few rows of the dataset after encoding
insurance_data_encoded.head()


# In[6]:


#data distribution
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
sns.histplot(insurance_data['age'], bins=30, kde=True, ax=axes[0,0])
axes[0,0].set_title('Age Distribution')
sns.histplot(insurance_data['bmi'], bins=30 ,kde =True, ax=axes[0,1])
axes[0,0].set_title('bmi Distribution')
sns.histplot(insurance_data['children'],bins=6,kde=False, ax=axes[1,0])
axes[1,0].set_title('children Distribution')
sns.histplot(insurance_data['charges'],bins=20 ,kde=True, ax=axes[1,1])
axes[1,1].set_title('charges Distribution')


# In[7]:


# standardize and normalize 
standard_scaler =StandardScaler()
insurance_data_encoded[['age','bmi']] = standard_scaler.fit_transform(insurance_data_encoded[['age','bmi']])

min_max_scaler = MinMaxScaler()
insurance_data_encoded['charges'] = min_max_scaler.fit_transform(insurance_data_encoded[['charges']])
insurance_data_encoded.head()


# In[8]:


# Calculating the updated correlation matrix with encoded variables
updated_correlation_matrix = insurance_data_encoded.corr()

# Plotting the heatmap for the updated correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(updated_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlatioin Matrix")
plt.show()



# In[9]:


# Preparing the data for the linear regression model
X = insurance_data_encoded.drop('charges', axis=1) 
y = insurance_data_encoded['charges']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = linear_regression_model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2


# In[ ]:





# In[ ]:




