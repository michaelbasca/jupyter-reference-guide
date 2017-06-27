
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import warnings

plt.style.use('ggplot') # other options are 'classic', 'grayscale', 'fivethirtyeight', 'ggplot',
                        # seaborn-whitegrid', 'seaborn-white', 'seaborn-pastel'
                        # use "print(plt.style.available)" for more options

warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')


# In[2]:

print(sns.get_dataset_names())
titanic = sns.load_dataset('titanic')


# # Missing Value Analysis
# 
# We want to understand the sparsity of our dataset.  
# One sound approach is to understand the sparsity of the features that are not complete (i.e. with 1 or more NA)

# In[3]:

# Get the dimensions of the titanic data set
titanic.shape


# In[4]:

# Code attributed to Vivek Srinivasan https://www.kaggle.com/viveksrinivasan/kernels
missingValueColumns = titanic.columns[titanic.isnull().any()].tolist()
msno.bar(titanic[missingValueColumns],            figsize=(20,8),color="#34495e",fontsize=20,labels=True,)


# So out of the 15 features we get 4 features that have some NAs.  Let's see how they are distributed among the samples.

# In[5]:

# Code attributed to Vivek Srinivasan https://www.kaggle.com/viveksrinivasan/kernels
msno.matrix(titanic[missingValueColumns], width_ratios=(10,1),            figsize=(20,8),color=(0.204, 0.286, 0.369), fontsize=20, sparkline=True, labels=True)


# In[15]:

msno.heatmap(titanic[missingValueColumns])


# # Some General Plots

# ## A 2 X 1 plot

# In[6]:

# Code attributed to Jake Vanderplas https://github.com/jakevdp/PythonDataScienceHandbook
plt.figure()  # create a plot figure, use figsize argument to change size e.g. figsize = (10, 8)

x = np.linspace(0, 10, 100)

# create the first of two panels and set current axis
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))
plt.title('Sin')

# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.title('Cos');


# Here's another way of plotting using an object oriented interface.

# In[7]:

# Code attributed to Jake Vanderplas https://github.com/jakevdp/PythonDataScienceHandbook
# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)

# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[0].set_title('Sin')
ax[1].plot(x, np.cos(x))
ax[1].set_title('Cos');


# Here's a shorter way of writing the code.

# In[8]:

# Code attributed to Jake Vanderplas https://github.com/jakevdp/PythonDataScienceHandbook
# First create a grid of plots
# ax will be an array of two Axes objects
fig, axes = plt.subplots(2) # use figsize argument to change size e.g. figsize = (10, 8)

for lab, func, ax in zip(['Sin', 'Cos'], [np.sin, np.cos], axes):
    # Call plot() method on the appropriate object
    ax.plot(x, func(x))
    ax.set_title(lab)


# Here's another way of plotting using itertools and the gridspec module from matplotlib.

# In[9]:

# Code attributed to Sebastian Raschka https://sebastianraschka.com/books.html
import matplotlib.gridspec as gridspec
import itertools

gs = gridspec.GridSpec(2, 1)
fig = plt.figure() # use figsize argument to change size e.g. figsize = (10, 8)

for lab, func, grd in zip(['Sin', 'Cos'], 
                          [np.sin, np.cos],
                          itertools.product([0, 1], repeat=1)):
    
    ax = plt.subplot(gs[grd[0]])
    plt.plot(x, func(x))
    plt.title(lab)


# In[10]:

list(itertools.product([0, 1], repeat=3))


# In[11]:

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x));


# In[12]:

plt.plot(x, np.sin(x))
plt.axis('tight');


# In[13]:

plt.plot(x, np.sin(x))
plt.axis('equal');


# In[14]:

plt.axis


# In[ ]:



