#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings
import os
warnings.filterwarnings("ignore")
py.offline.init_notebook_mode(connected = True)

#import numpy as np: This line imports the NumPy library and gives it an alias np for easier reference in the code.
#import pandas as pd: This line imports the Pandas library and gives it an alias pd for easier reference in the code.
#import matplotlib.pyplot as plt: This line imports the Matplotlib library's pyplot module and gives it an alias plt for easier reference in the code.
#import seaborn as sns: This line imports the Seaborn library and gives it an alias sns for easier reference in the code.
#import plotly as py: This line imports the Plotly library and gives it an alias py for easier reference in the code.
#import plotly.graph_objs as go: This line imports the graph_objs module from the Plotly library and gives it an alias go for easier reference in the code.
#from sklearn.cluster import KMeans: This line imports the KMeans class from the Scikit-learn library's cluster module.
#import warnings: This line imports the warnings module to allow filtering of warnings in the code.
#import os: This line imports the os module for accessing and manipulating files and directories in the operating system.
#warnings.filterwarnings("ignore"): This line filters all warnings in the code to be ignored.
#py.offline.init_notebook_mode(connected = True): This line initializes Plotly in offline mode and connects it to the Jupyter Notebook.


# In[4]:


df = pd.read_csv('Mall_Customers.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.dtypes


# In[8]:


df.isnull().sum()


# In[ ]:





# In[9]:


plt.style.use('fivethirtyeight')


# In[10]:


plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(df[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()


# In[11]:


plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'Gender' , data = df)
plt.show()


# In[12]:


plt.figure(1 , figsize = (15 , 7))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = df)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()


# In[13]:


plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = df[df['Gender'] == gender] ,
                s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 
plt.title('Age vs Annual Income w.r.t Gender')
plt.legend()
plt.show()


# In[14]:


plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,
                data = df[df['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 
plt.title('Annual Income vs Spending Score w.r.t Gender')
plt.legend()
plt.show()


# In[15]:


plt.figure(1 , figsize = (15 , 7))
n = 0 
for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1 
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.violinplot(x = cols , y = 'Gender' , data = df , palette = 'vlag')
    sns.swarmplot(x = cols , y = 'Gender' , data = df)
    plt.ylabel('Gender' if n == 1 else '')
    plt.title('Boxplots & Swarmplots' if n == 2 else '')
plt.show()


# In[17]:


#Segmentation using Age and Spending Score


# In[18]:


'''Age and spending Score'''
X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)


# In[19]:


plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_
h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , 
            s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')
plt.show()


# In[20]:


# Segmentation using Annual Income and Spending Score
'''Annual Income and spending Score'''
X2 = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X2)
    inertia.append(algorithm.inertia_)
    plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()
algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X2)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
h = 0.02
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z2 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z2 = Z2.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')
plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = df , c = labels2 , 
            s = 200 )
plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')
plt.show()


# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data from a CSV file
data = pd.read_csv('Mall_Customers.csv')

# Select the Age column for clustering
X = data[['Age']]

# Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Add the cluster labels to the original DataFrame
data['AgeCluster'] = kmeans.labels_

# Visualize the clusters using a scatter plot
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster = data[data['AgeCluster'] == i]
    plt.scatter(cluster['Age'], cluster['Spending Score (1-100)'], s=50, color=colors[i])
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data from a CSV file
data = pd.read_csv('Mall_Customers.csv')

# Select the Age and Spending Score columns for clustering
X = data[['Age', 'Spending Score (1-100)']]

# Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Add the cluster labels to the original DataFrame
data['AgeSpendingCluster'] = kmeans.labels_

# Visualize the clusters for each gender using scatter plots
cluster_colors = ['red', 'green', 'blue']
for gender in data['Gender'].unique():
    gender_data = data[data['Gender'] == gender]
    for i in range(3):
        cluster = gender_data[gender_data['AgeSpendingCluster'] == i]
        plt.scatter(cluster['Age'], cluster['Spending Score (1-100)'], s=50, color=cluster_colors[i])
    plt.xlabel('Age')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Clusters for ' + gender)
    plt.show()


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data from a CSV file
data = pd.read_csv('Mall_Customers.csv')

# Select the Spending Score column for clustering
X = data[['Spending Score (1-100)']]

# Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Add the cluster labels to the original DataFrame
data['SpendingCluster'] = kmeans.labels_

# Visualize the clusters using a histogram
cluster_colors = ['red', 'green', 'blue']
for i in range(3):
    cluster = data[data['SpendingCluster'] == i]
    plt.hist(cluster['Spending Score (1-100)'], bins=20, color=cluster_colors[i], alpha=0.5, label='Cluster '+str(i))
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Count')
plt.title('Clusters of Spending Score')
plt.legend()
plt.show()



# In[30]:


#for annual ıncome
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data from a CSV file
data = pd.read_csv('Mall_Customers.csv')

# Select the Annual Income column for clustering
X = data[['Annual Income (k$)']]

# Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Add the cluster labels to the original DataFrame
data['IncomeCluster'] = kmeans.labels_

# Visualize the clusters using a scatter plot
cluster_colors = ['red', 'green', 'blue']
for i in range(3):
    cluster = data[data['IncomeCluster'] == i]
    plt.scatter(cluster['Annual Income (k$)'], cluster['Spending Score (1-100)'], s=50, color=cluster_colors[i])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clusters of Annual Income')
plt.show()


# 
# 
# This code performs K-means clustering on the "Annual Income" column of a mall customer dataset and visualizes the resulting clusters using a scatter plot.
# 
# Here's what each line of the code does:
# 
# import pandas as pd: imports the Pandas library for data manipulation and analysis
# import matplotlib.pyplot as plt: imports the Matplotlib library for data visualization
# from sklearn.cluster import KMeans: imports the KMeans clustering algorithm from the Scikit-learn library
# data = pd.read_csv('Mall_Customers.csv'): reads the mall customer dataset from a CSV file and stores it in a Pandas DataFrame called "data"
# X = data[['Annual Income (k$)']]: selects only the "Annual Income" column from the DataFrame and assigns it to variable "X"
# kmeans = KMeans(n_clusters=3, random_state=0).fit(X): performs K-means clustering on the "Annual Income" column with 3 clusters using the KMeans algorithm and assigns the resulting labels to variable "kmeans"
# data['IncomeCluster'] = kmeans.labels_: adds the cluster labels to the original DataFrame as a new column called "IncomeCluster"
# cluster_colors = ['red', 'green', 'blue']: defines a list of colors to use for the scatter plot
# for i in range(3):: loops over the 3 clusters
# cluster = data[data['IncomeCluster'] == i]: selects the data points belonging to cluster "i"
# plt.scatter(cluster['Annual Income (k$)'], cluster['Spending Score (1-100)'], s=50, color=cluster_colors[i]): plots a scatter plot of the selected data points, using the "Annual Income" as the x-axis and "Spending Score" as the y-axis, and assigns a color to the points based on their cluster label
# plt.xlabel('Annual Income (k$)'): sets the x-axis label to "Annual Income (k$)"
# plt.ylabel('Spending Score (1-100)'): sets the y-axis label to "Spending Score (1-100)"
# plt.title('Clusters of Annual Income'): sets the plot title to "Clusters of Annual Income"
# plt.show(): displays the plot
# Overall, this code demonstrates how to perform K-means clustering on a single feature and visualize the resulting clusters using Matplotlib. It can be adapted to perform clustering on other features or using different numbers of clusters.
# 
# 
# The codes above demonstrate the use of K-means clustering algorithm for customer segmentation. The data used in the code includes customer demographic and behavioral data such as age, gender, annual income, and spending score.
# 
# The first part of the code imports the necessary libraries and initializes Plotly to be used in offline mode. The data is then loaded into a Pandas DataFrame and visualized using Matplotlib and Seaborn libraries.
# 
# Next, the code applies K-means clustering algorithm to the data and determines the optimal number of clusters using the elbow method. The algorithm is applied to various features such as age, gender, annual income, and spending score, and the resulting clusters are visualized using Matplotlib and Seaborn libraries.
# 
# Overall, the codes above provide a useful example of how K-means clustering algorithm can be applied to customer segmentation and how data visualization can be used to gain insights into customer behavior and demographics.
# 

# In[ ]:




