#!/usr/bin/env python
# coding: utf-8

# # Assignment 0: Intro to CS 363D
# 
# Your UTCS accounts come with `python` installed but you can’t add packages to it (without root). Especially installing `scipy` and `numpy` on your own can be painful. We recommend using a local install of python, so that you can add any packages you need easily. We will use the python 3 version of [Anaconda](https://docs.continuum.io/), an open-source package and environment manager optimized for data science applications. It’s maintained by Austin-based Continuum Analytics.

# ## Part 1: Setting up your Jupyter Notebook
# 
# ### Installing Anaconda on personal computers
# 
# * Make sure you have 2.5 GB free space available before you start installing. Use `baobab` to clear unused files and Chrome’s `.config` litter.
# * Download the Anaconda - Python 3.6 zip archive from the [official website](https://www.anaconda.com/download/#download). This file is `~500 MB`, and the uncompressed version will be `~2 GB`.
# 
# ~~~~
# $ wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
# # current as of this assignment
# $ bash Anaconda3-4.4.0-Linux-x86_x64.sh
# ~~~~
# 
# * Accept the license agreement and select an installation directory. (Note that the install location defaults to `~/anaconda3`)
# * It will take a few minutes to install. Please be patient.
# * It will ask you whether to add it to your bash path (your `~/.bashrc` file). Note that the default is “no”. We recommend saying **yes**.
# * Start a **new terminal**. Let’s make sure we’re using Anaconda’s Python we just installed.
# 
# ~~~~
# $ which python
# /u/pkar/anaconda3/bin/python
# # Note:- your username will appear in the path. Mine is 'pkar'.
# ~~~~
# 
# * Let’s start a notebook server.
# 
# ~~~~
# $ jupyter notebook
# ~~~~
# 
# * It should open a browser tab and start a local server.
# 
# <img src="imgs/screen1.png">
# 
# * Create a directory for this course (I called mine CS363), and make a new Python 3 notebook.
# * Have a go at using the notebook interface. We’ll see more of what we can do with Jupyter notebooks in Part 2.
# 
# ### Installing Jupyter notebooks on lab computers
# 
# The Anaconda install is too large for your allocated space in the computer lab. But you can still install Jupyter in the lab without installing Anaconda.
# 
# First log into your favorite cs linux machine. Almost all required packages are already installed on those machines.
# 
# ~~~~
# pip3 install --user jupyter
# ~~~~
# 
# Let’s fire up jupyter:
# 
# ~~~~
# ipython3 notebook --no-browser
# ~~~~
# 
# If you’re logged in locally open a browser at the url printed out. Otherwise port forward
# 
# ~~~~
# ssh -NL PORT:localhost:PORT SERVER
# ~~~~
# 
# You can figure out the `PORT` from the url provided by ipython. `SERVER` is the url of the machine you started ipython on.
# 
# Pro tip: If you want ipython to survive your SSH login, start it in a screen session.
# 
# *Note: This guide is inspired from Prof. Philip Krähenbühl's Neural Network course ([CS 342](http://www.philkr.net/cs342/homework/01/install.html)).*
# 
# ### Additional Links
# 
# * [Stanford CS 231N iPython Tutorial](http://cs231n.github.io/ipython-tutorial/)
# * [Jupyter Notebook Beginner Guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/)
# * [Datacamp Jupyter Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook#gs.PV0TCww)
# 
# You may have unforeseen difficulties with this portion of the assignment (not enough disk space, issues with permissions or package versions, conflicts with your personal `.bashrc` or `.zshrc` etc). 
# **Please start early** and work with the TA (during office hours) or other classmates to resolve these.

# ## Part 2: A Simple Data Science Task
# 
# For this task we'll be using the 1994 Adult Census Income data (`adult.csv`) collected by Ronny Kohavi and Barry Becker. This is a reasonably clean dataset with both categorical and integer attributes. The dataset consists of `32.5K` rows with 14 attributes.
# 
# #### Attributes
# 
# You can find a detailed description of the dataset [here](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names).
# 
# | Attribute Name 		| Type 				|
# | --------------------- | ----------------- |
# | age					| continuous		|
# | workclass				| categorical		|
# | fnlwgt				| continuous		|
# | education 			| categorical		|
# | education-num			| categorical		|
# | marital-status		| categorical		|
# | occupation			| categorical		|
# | relationship			| categorical		|
# | race					| categorical		|
# | sex					| categorical		|
# | capital-gain			| continuous		|
# | capital-loss			| continuous		|
# | hours-per-week		| continuous		|
# | native-country		| categorical		|

# In[2]:


# Standard Headers
# You are welcome to add additional headers if you wish
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.__version__


# In[3]:


# Enable inline mode for matplotlib so that IPython displays graphs.
get_ipython().run_line_magic('matplotlib', 'inline')


# You can find more on reading (Comma Separated Value) CSV data as a Pandas dataframe [here](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html).

# In[4]:


# add skipinitialspace=True to skip spaces after delimiter (will be required later for the map function)
adult_data = pd.read_csv("adult.csv", skipinitialspace=True)
# show the head of the data (first 5 values)
type(adult_data)


# In[240]:


# display data types of various columns in a dataframe
adult_data.dtypes


# Q1.
# 1. Group the `adult_data` using the `marital-status` column. You may want to look at the `groupby()` method for dataframes [here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html).
# 2. Display the mean, median and standard deviation statistics of `hours-per-week` column for each `marital-status` column.
# 3. Which marital status category has the maximum average work hours per week? which has the minimum?
# 4. Which marital status category has the most variability in work hours per week? which has the least?

# In[241]:


# your code here
grouped = adult_data.groupby('marital-status',as_index=False)
grouped.mean()


# In[242]:


grouped['hours-per-week'].agg([np.mean, np.median, np.std])


# In[243]:


(grouped['hours-per-week'].mean()).max()
(grouped['hours-per-week'].mean()).min()


# In[244]:


grouped['hours-per-week'].var().max()
grouped['hours-per-week'].var().min()


# Q2. Show the distribution of the dataset with respect to the `education` column. Which plot is most suitable for this? Use matplotlib (or a library of your choice) to plot the distribution.

# In[245]:


adult_data['education'].head()


# In[246]:


adult_data['education'].value_counts().plot(kind='bar')


# In[247]:


# Please don't change this cell!!
continent_dict = {
    'Cambodia' : 'Asia',
    'Canada' : 'North-America',
    'China' : 'Asia',
    'Columbia' : 'South-America',
    'Cuba' : 'North-America',
    'Dominican-Republic' : 'North-America',
    'Ecuador' : 'South-America',
    'El-Salvador' : 'North-America',
    'England' : 'Europe',
    'France' : 'Europe',
    'Germany' : 'Europe',
    'Greece' : 'Europe',
    'Guatemala' : 'North-America',
    'Haiti' : 'North-America',
    'Holand-Netherlands' : 'Europe',
    'Honduras' : 'North-America',
    'Hong' : 'Asia',
    'Hungary' : 'Europe',
    'India' : 'Asia',
    'Iran' : 'Asia',
    'Ireland' : 'Europe',
    'Italy' : 'Europe',
    'Jamaica' : 'North-America',
    'Japan' : 'Asia',
    'Laos' : 'Asia',
    'Mexico' : 'North-America',
    'Nicaragua' : 'North-America',
    'Outlying-US(Guam-USVI-etc)' : 'North-America',
    'Peru' : 'South-America',
    'Philippines' : 'Asia',
    'Poland' : 'Europe',
    'Portugal' : 'Europe',
    'Puerto-Rico' : 'North-America',
    'Scotland' : 'Europe',
    'South' : 'Other',
    'Taiwan' : 'Asia',
    'Thailand' : 'Asia',
    'Trinadad&Tobago' : 'South-America',
    'United-States' : 'North-America',
    'Vietnam' : 'Asia',
    'Yugoslavia' : 'Europe',
    '?' : 'Other'
}


# Q3. Using the dictionary provided above, create a new column called `continent` using the existing `native-country` column in the dataframe. You may want to look at the `map()` method for dataframes.

# In[248]:


# You may want to create a deep copy of the initial dataframe object
# so that you can run this cell multiple times without errors.
adult_data_copy = adult_data.copy()
# your code goes here
continent = adult_data_copy['native-country'].map(continent_dict)
new = pd.DataFrame({'continent':continent,'native-country':adult_data_copy['native-country']})
new.head(15)


# Q4. Plot a bar graph showing the average age of working adults from each continent, and show the standard deviation on the same graph.
# 
# An example bar plot.
# <img src="imgs/screen3.png">

# In[250]:


new1 = pd.DataFrame({'continent':continent,'age':adult_data_copy['age']})
new1.head()


# In[251]:


# your code goes here
q4 = new1.groupby('continent')
q4.std()
q4.mean()


# In[252]:


# data to plot
age_std=(11.963112,13.878938,13.686381,13.430850,12.692335)
age_mean=(38.04321,41.600768,38.540090,38.299270,38.728507)

# create plot
fig, ax = plt.subplots()
index = np.arange(5)
 
rects1 = plt.bar(index,age_mean,bar_width,alpha=opacity,color='b',label='age_mean')
 
rects2 = plt.bar(index + bar_width,age_std,bar_width,alpha=opacity,color='g',label='age_std')
 
plt.xlabel('Continent')
#plt.ylabel('Scores')
#plt.title('Scores by person')
plt.xticks(index + bar_width, ('Asian', 'Europe', 'North-America', 'South-America','Other'))
plt.legend()
plt.tight_layout()
plt.show()


# Q5. To reduce the dimensionality of this dataset, which attribute or attributes would you eliminate? Explain why.

# In[253]:


'''Explain why here (as a comment):
Capital loss and capital gain. Because most values of the two variables are 0 - it does not make sense with
analyzing bunch of null data.

'''


# ## Part 3: Handling Missing Values
# 
# For this task we'll be using a subset of the leaf dataset created by professors from University of Porto, Portugal. This dataset consists in a collection of shape and texture features extracted from digital images of leaf specimens originating from a total of 40 different plant species, but for the purpose of this assignment we're only going to consider 4 plant species.
# 
# You can find more information about the dataset [here](http://archive.ics.uci.edu/ml/datasets/Leaf).
# 
# <img src="imgs/screen4.png">

# In[272]:


leaf_data = pd.read_csv("leaf.csv")
leaf_data.head()


# Q6. Eccentricity of the leaf is a measure of how much the shape of the leaf varies from a perfect circle. Unfortunately the dataset is missing values in the `eccentricity` column. Fill in these missing values with something reasonable.

# In[255]:


# your code goes here
#we use the mean of the eccentricity of each class to fill in the missing blanks.
eccen = leaf_data.groupby('class')


# In[256]:


#Or we use the mean of all eccentricity score to replace the NaN.
leaf_data[['eccentricity']].mean()


# In[257]:


values = {'eccentricity': 0.567526}
leaf_gooddata=leaf_data.fillna(value=values)
leaf_gooddata.head()


# Q7. Normalize the `eccentricity` column. Where `value_norm = (value - mean(value)) / std(value)`. Display the head of the new data frame.

# In[258]:


# your code goes here
df = leaf_gooddata[['eccentricity']]
norm_df = (df-df.mean())/df.std()
norm_df.head()


# Q8. Plot a scatter plot between `smoothness` and normalized `eccentricity`. Place `smoothness` on the X axis.

# In[259]:


# your code goes here
x = leaf_data[['smoothness']]
y = normalized_df[['eccentricity']]

# Plot
plt.scatter(x, y,color='g')
plt.title('Scatter plot')
plt.xlabel('smoothness')
plt.ylabel('normalized eccentricity')
plt.show()


# Q9. Now plot the same scatter plot as Q7 but give a different color to each `class` label. What can you infer from this plot (provide a short answer in the form of comments)?

# In[277]:


data1 = leaf_data[['class','smoothness']]
data1.head()
data2 = pd.concat([data1,norm_df],ignore_index=True)
data2.head()


# In[280]:


data2.dropna(subset=['class'])


# In[279]:


x = leaf_data[['smoothness']]
y = normalized_df[['eccentricity']]

# Plot
plt.scatter(x, y)
for label in range(4):
    plt.scatter(x[leaf_data[['class']] == label], y[leaf_data[['class']] == label], label=‘%s’ % (label))
plt.legend()
plt.title('Scatter plot')
plt.xlabel('smoothness')
plt.ylabel('normalized eccentricity')
plt.show()


# In[266]:


x = leaf_data[['smoothness']]
y = normalized_df[['eccentricity']]

# Plot
plt.scatter(x, y)
for label in range(4):
    plt.scatter(x[leaf_data[['class']] == label], y[norm_df[['class']] == label])
plt.legend()
plt.title('Scatter plot')
plt.xlabel('smoothness')
plt.ylabel('normalized eccentricity')
plt.show()


# In[264]:


# your code goes here
x = leaf_data[['smoothness']]
y = normalized_df[['eccentricity']]

# Plot
plt.scatter(x, y,c=leaf_data[['class']])
plt.title('Scatter plot')
plt.xlabel('smoothness')
plt.ylabel('normalized eccentricity')
plt.show()
#One of the class is dense around a specific area.


# Q10. Calculate the correlation between the normalized `eccentricity` and the `smoothness` column. 

# In[229]:


# your code goes here
leaf_data['smoothness'].corr(norm_df['eccentricity'])


# Q11. Create a plot to determine if there are any outliers in the `average-contrast` attribute.

# In[234]:


# your code goes here
new11 = leaf_data.dropna(subset=['average-contrast'])
new11[['average-contrast']].plot(kind='hist')

