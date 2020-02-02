#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook is to build a model (Deep Neural Network) with Tensorflow. Below are the differents steps to do that. This notebook is split in several parts:
# 
#     I. Loading Libraries
#     II. Reading data
#     III. Preprocessing and feature study using Statistical Methods
#     IV. Feature Engineering
#     V. Data Split and Manipulation
#     VI. Model Building
#     VII. Training on complete data
#     VIII. Predictions 
#     IX.Conclusion
#     
#  The Goal
#     Each row in the dataset describes the characteristics whichhave an influence on the y- value.
#     Our goal is to predict the y value, given these features.
#  
# 

# ## Loading Libraries

# In[81]:


# Essentials
import numpy as np
import pandas as pd
import datetime
import random

# Plots
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


#Neural Network
import tensorflow as tf


# In[3]:


# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000


# ## Reading data

# #### The intern_data.csv file is used for performing training and validation tasks in the model. 

# In[4]:


data_train= pd.read_csv('intern_data.csv')
data_test= pd.read_csv('intern_test.csv')
print("Dataset shape:",'data_train', data_train.shape, 'data_test', data_test.shape)


# #### The first column in the data set has no relevance to predicting the value of y.

# In[5]:


#Collecting ID's from data sets
train_ID = data_train[data_train.columns[0]]
test_ID = data_test[data_test.columns[0]]


# ## Feature Study

# ### Missing Data

# In[6]:


# Handle missing values
data_train.describe(include= 'all')


# #### - The descripting of the data set above points out that no feature has missing values. All columns have a count of 500, which is the number of obbservations observed in the data set.
# #### - Columns 'c' and 'h' are categorical in nature, whereas all other variables are continuous.
# #### - All columns appear to be in standardized form. 
# #### - All columns lie between the 0 to 1 range and appear to be normalized.

# #### To statistically determine every columns dependance on the 'y' variable, correlation between all is plotted below in the form of a horizontal bar chart. The most correlated features to 'y' are: 
# #### 1. 'e'
# #### 2. 'g'
# #### 3. 'f'
# #### 4. 'b'
# #### 5. 'a'
# #### 6. 'd'
# 
# #### 'd' appears to be negatively correlated to the 'y' variable

# ### Correlations

# In[84]:


#y correlation with all the feature
plt.figure(figsize=(8, 10))
data_train.corr()['y'].sort_values().plot(kind='barh')


# #### - To see a correlation between all features and the dependent varaible, a heat map with correlation as the character is used below. 
# #### - The first column identifies,in order, the most to least correlated features to 'y'. 
# #### - The theory about 'e' being the most correlated feature with 'y' can be confirmed from this plot.
# #### - No multicollinearity can be observed between the different  features.

# In[85]:


corr_= data_train.corr()
cols= corr_.sort_values(by= 'y', ascending=False)['y'].index  #finding the most highly correlated columns with 'y' 
                                                              #could have used nlargest with correlation too 
k = 10
k_corr_matrix = data_train[cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(k_corr_matrix, xticklabels= k_corr_matrix.columns.values,yticklabels= k_corr_matrix.columns.values,annot=True,square= True,linewidths=.5)
b, t = plt.ylim() # the top level and bottom level values 
b += 0.5 # Adding 0.5 to the bottom
t -= 0.5 # Subtracting 0.5 from the top
plt.ylim(b, t) #updating the ylim(bottom, top) values to incorporate new changes
plt.show()


# In[90]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) 
ax1.scatter(data_train['a'],data_train['y'])
ax1.set_title('a and y')
ax1.set(ylabel="y")
ax1.set(xlabel="a")
ax2.scatter(data_train['b'],data_train['y'])
ax2.set(ylabel="y")
ax2.set(xlabel="b")
ax2.set_title('b and y')
ax3.scatter(data_train['d'],data_train['y'])
ax3.set(ylabel="y")
ax3.set(xlabel="a")
ax3.set_title('d and y')
plt.show()


# #### No linear relationship can be identified between the 'y' valriable with all other features of the data-set. 

# In[91]:


f, (ax4, ax5, ax6) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax4.scatter(data_train['e'],data_train['y'])
ax4.set(ylabel="y")
ax4.set(xlabel="e")
ax4.set_title('e and y')
ax5.scatter(data_train['f'],data_train['y'])
ax5.set(ylabel="y")
ax5.set(xlabel="f")
ax5.set_title('f and y')
ax6.scatter(data_train['g'],data_train['y'])
ax1.set(ylabel="y")
ax1.set(xlabel="g")
ax6.set_title('g and y')


# In[94]:


#Checking the distribution of the dependant variable
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 10))
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="y")
ax.set(title="y distribution")
normalized_y = data_train['y']
# helpful_normalized.describe()
sns.distplot(normalized_y, color= 'b')


# In[12]:


#skewness and kurtosis
print("Skewness: " + str(data_train['y'].skew()))
print("Kurtosis: " + str(data_train['y'].kurt()))


# 'y' is skewed to the left. This is a problem with most predictive models as they don't do well with non-normally distributed data. Feature transfomration can be performed to bring the distribution closer to normal.
# 

# ## Feature Engineering

# ### Categorical Variables

# In[95]:


values = data_train.values


# #### Categorical variables need to be converted to a different form, before being introduced to a model for prediction. 
# #### Each category gets encoded into a float value for training.

# In[14]:


# Converting values in the column c to float
c_vals = values[:, 3]
c_vals_float = np.ones(c_vals.shape[0], dtype=np.float32)
c_vals_float[c_vals == 'blue'] = 0.1
c_vals_float[c_vals == 'red'] = 0.4
c_vals_float[c_vals == 'yellow'] = 0.7
c_vals_float[c_vals == 'green'] = 0.95


# #### The 'h' variable has black and white values only, with a majority of them being white as seen from the describe function on the data frame. These categories also get encoded to float values

# In[15]:


# Convert values in the column h to float
h_vals = values[:, 8]
h_vals_float = np.ones(h_vals.shape[0], dtype=np.float32)
h_vals_float[h_vals == 'black'] = 0


# #### Forming arrays whith training data to introduce to the model for training

# In[16]:


# Create numpy array of features and labels
x_arr = np.zeros((values.shape[0], 8), dtype=np.float32)
y_arr = np.zeros(values.shape[0], dtype=np.float32)

x_arr[:, 0:2] = values[:, 1:3]
x_arr[:, 2] = c_vals_float
x_arr[:, 3:7] = values[:, 4:8]
x_arr[:, 7] = h_vals_float

y_arr[:] = values[:, 9]


# ## Data Split and Manipulation

# #### Data is 500 obervations, so the training set has been set to 400 observations. The last 100 obervations are resorurced for model validation

# In[17]:


# Split the data to train and validation set
ind_arr = np.arange(0, values.shape[0] - 1)
random.seed(101)
np.random.shuffle(ind_arr)

x_train = x_arr[ind_arr[0:400], :]
x_valid = x_arr[ind_arr[400:], :]
y_train = y_arr[ind_arr[0:400]]
y_valid = y_arr[ind_arr[400:]]


# ## Model building

# In[ ]:


#### The model architecture is 20, 20, 18 and 1
#### This model architecture is fitting for the size of the dataset I am using.
#### Adding more weights to the model will make it fit but will result in competency of running /


# In[54]:


# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(8,)),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dense(20),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dense(10),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dense(1)
])


# In[55]:


# Compile the NN
model.compile(optimizer='adam', loss='mean_squared_error')


# In[65]:


# Train the model
model.fit(x_train, y_train, epochs=200, batch_size=16, validation_data=(x_valid, y_valid))
print("\n\n TRAINING COMPLETE! STARTING EVALUATION \n\n")


# In[66]:


# Evaluate the model
ev= model.evaluate(x_valid, y_valid, batch_size=16)


# In[67]:


#Predict validation values
pr= model.predict(x_valid)


# In[68]:


l_pr=[]
for i in pr:
    l_pr.append(i[0])

val_pr_df= pd.DataFrame()
val_pr_df['Actual']= y_valid
val_pr_df['Predicted']= l_pr
    


# In[69]:


val_pr_df


# In[77]:


plt.figure()
plt.plot(y_valid, l_pr, 'ro', alpha= 0.3)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()


# ## Training on complete data

# In[23]:


#Training on complete data 
x_train = x_arr[ind_arr[0:500], :]
y_train = y_arr[ind_arr[0:500]]


# In[24]:


# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(8,)),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dense(20),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dense(10),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dense(1)
])


# In[25]:


# Compile the NN
model.compile(optimizer='adam', loss='mean_squared_error')


# In[26]:


# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_valid, y_valid))
print("\n\n TRAINING COMPLETE! \n\n")


# ## Predictions on intern_data.csv

# In[27]:


#Reading test data
test_data = pd.read_csv("intern_test.csv")
values = test_data.values


# ### Categorical Variables

# In[28]:


# Convert values in the column c to float
c_vals = values[:, 3]
c_vals_float = np.ones(c_vals.shape[0], dtype=np.float32)
c_vals_float[c_vals == 'blue'] = 0.1
c_vals_float[c_vals == 'red'] = 0.4
c_vals_float[c_vals == 'yellow'] = 0.7
c_vals_float[c_vals == 'green'] = 0.95

# Convert values in the column h to float
h_vals = values[:, 8]
h_vals_float = np.ones(h_vals.shape[0], dtype=np.float32)
h_vals_float[h_vals == 'black'] = 0


# In[29]:


test_x_arr = np.zeros((values.shape[0], 8), dtype=np.float32)
test_x_arr[:, 0:2] = values[:, 1:3]
test_x_arr[:, 2] = c_vals_float
test_x_arr[:, 3:7] = values[:, 4:8]
test_x_arr[:, 7] = h_vals_float


# ### Predict the NN

# In[30]:


test_predictions = model.predict(test_x_arr)


# In[31]:


test_predictions


# In[32]:


l_test_predictions=[]
for i in test_predictions:
    l_test_predictions.append(i[0])


# ### Dataframe with predictions

# In[33]:


intern_predicted= pd.DataFrame()
intern_predicted['Index']= test_ID


# In[34]:


intern_predicted['Predicted y'] =l_test_predictions


# In[35]:


intern_predicted.set_index('Index', inplace= True)
intern_predicted


# In[36]:


intern_predicted.to_csv('Intern_test_solution.csv')

