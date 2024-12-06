#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[6]:


data = pd.read_csv('/Users/yoonchanghoon/Desktop/online_retail (1).csv', index_col=False)


# In[7]:


data.head()


# In[8]:


data.info()


# In[9]:


for x in data.columns: 
    print(x)
    print(data[x].values)


# In[10]:


df = data.copy()


# In[11]:


df = df.drop(columns='index')


# In[12]:


df.isnull().sum()


# In[13]:


df= df.dropna(subset=['CustomerID'])


# In[14]:


df.isnull().sum()


# In[15]:


print(df.duplicated().sum())


# In[16]:


df = df.drop_duplicates()


# In[17]:


print(df.duplicated().sum())


# In[18]:


df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# In[19]:


df.dtypes


# In[20]:


df.describe()


# In[21]:


df.shape


# In[22]:


df[df['Quantity']<0]


# In[23]:


df[df['StockCode'].str.len() == 1][['StockCode','Description']].value_counts()


# In[24]:


df[(df['Quantity']<0) & (df['InvoiceNo'] == 'C536379')]


# In[25]:


df[(df['Quantity']<0) & (df['Description'] == 'Discount')][['StockCode','Description']].value_counts()


# In[26]:


df[df['UnitPrice']==0]


# In[27]:


df = df[(df['Quantity']>0) & (df['UnitPrice']>0)]


# In[28]:


df.shape


# In[30]:


df.to_csv('/Users/yoonchanghoon/Desktop/online_retail_preprocessed (1).csv', index=False)


# In[31]:


df.describe()


# In[32]:


df['CustomerID']


# In[33]:


print(f"Number of unique invoices: {df['InvoiceNo'].nunique()}")
print(f"Number of unique products: {df['StockCode'].nunique()}")
print(f"Number of unique descriptions: {df['Description'].nunique()}")


# In[34]:


print(f"Most common products:\n{df['Description'].value_counts().head(10)}")


# In[35]:


# Define custom bin ranges
bin_ranges = list(range(0, 1001, 50))  # Bins from 0 to 1000 in increments of 50

# Plot histogram with custom bins
plt.figure(figsize=(11, 5))
sns.histplot(df['Quantity'], bins=bin_ranges, kde=False)
plt.title('Quantity Distribution (Custom Intervals)')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()


# In[36]:


# Apply log transformation to Quantity
df['Log_Quantity'] = df['Quantity'].apply(lambda x: np.log1p(x) if x > 0 else 0)

# Plot histogram of log-transformed data
plt.figure(figsize=(10, 6))
sns.histplot(df['Log_Quantity'], bins=50, kde=True)
plt.title('Log-Transformed Quantity Distribution')
plt.xlabel('Log(Quantity)')
plt.ylabel('Frequency')
plt.show()


# In[37]:


df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')

plt.figure(figsize=(14, 5))
df['YearMonth'].value_counts().sort_index().plot(kind='bar')
plt.title('Temporal Trend of Transactions')
plt.xlabel('Year-Month')
plt.ylabel('Number of Transactions')
plt.show()


# In[38]:


print("\nCountry Analysis:")
print(f"Number of unique countries: {df['Country'].nunique()}")
print(f"Top 10 countries by transactions:\n{df['Country'].value_counts().head(10)}")

# Transactions by country
plt.figure(figsize=(15, 8))
df['Country'].value_counts().plot(kind='bar')
plt.title('Country-wise Transaction Breakdown')
plt.ylabel('Number of Transactions')
plt.show()


# In[39]:


df1 = df.copy()


# In[40]:


### Get Month
df1['InvoiceMonth'] = df1['InvoiceDate'].dt.date


# In[41]:


df1.tail()


# In[42]:


### Get Cohort Month
df1['FirstOrderDate'] = df1.groupby('CustomerID')['InvoiceMonth'].transform('min')


# In[43]:


df1.head()


# In[44]:


df1.tail()


# In[45]:


## Check value 
df1[df1['CustomerID'] == '12680'][['InvoiceMonth', 'FirstOrderDate']].value_counts()


# In[46]:


df1[df1['CustomerID'] == '17850'][['InvoiceMonth', 'FirstOrderDate']].value_counts()


# In[47]:


df1.dtypes


# In[48]:


### Change FirstOrderDate to dataframe dtype
df1['InvoiceMonth'] = pd.to_datetime(df1['InvoiceMonth'])
df1['FirstOrderDate'] = pd.to_datetime(df1['FirstOrderDate'])
# 월의 첫째날 가져오기 
df1['InvoiceMonth'] = df1['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
df1['FirstOrderDate'] = df1['FirstOrderDate'].dt.to_period('M').dt.to_timestamp()


# In[49]:


df1.dtypes


# In[50]:


df1['FirstOrderDate'].value_counts()


# In[51]:


df1[df1['FirstOrderDate'] == '2011-01-01'][['FirstOrderDate','InvoiceMonth']].value_counts()


# In[52]:


df1['CohortIndex'] = (
    (df1['InvoiceMonth'].dt.year - df1['FirstOrderDate'].dt.year) * 12 +
    (df1['InvoiceMonth'].dt.month - df1['FirstOrderDate'].dt.month) #+ 1
)


# In[53]:


df1.head()


# In[54]:


df1[df1['CustomerID'] == '12680'][['InvoiceMonth', 'FirstOrderDate','CohortIndex']]


# In[55]:


df1.describe()


# In[56]:


cohort_counts = (
    df1.groupby(['FirstOrderDate', 'CohortIndex'])['CustomerID']
    .nunique() #각 그룹의 고유 고객 수(CustomerID)를 직접 계산)
    .unstack() #Reshapes the grouped data into a pivot table
)
cohort_counts


# In[57]:


plt.figure(figsize=(14, 7))
plt.title('Customer Cohort Analysis_ Number of Customers')
sns.heatmap(data=cohort_counts, annot=True, vmin=0.0, fmt=".0f", cmap="YlGnBu")
plt.show()


# In[58]:


cohort_size = cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_size,axis=0) #axis=0 to ensure the divide along the row axis 
retention.round(3) * 100 #백분율 표시


# In[59]:


cohort_size


# In[60]:


324/885


# In[61]:


286.0/885


# In[62]:


plt.figure(figsize=(14, 7))
plt.title('Customer Retention Analysis')
sns.heatmap(data=retention, annot=True, fmt='.1%', vmin=0.0, vmax=0.5, cmap="Blues")
plt.show()


# In[63]:


average_quantity = (
    df1.groupby(['FirstOrderDate', 'CohortIndex'])['Quantity']
    .mean()
    .unstack()
    .round(1)
)


# In[64]:


average_quantity


# In[65]:


plt.figure(figsize=(14, 7))
plt.title('Average Quantity by Cohort Group')
sns.heatmap(data=average_quantity, annot=True, vmin=0.0, vmax=20, cmap="OrRd")
plt.show()


# In[66]:


df1['Sales'] = df1['Quantity'] * df1['UnitPrice']


# In[67]:


cohort_revenue = (
    df1.groupby(['FirstOrderDate', 'CohortIndex'])['Sales'].sum()
    .unstack() #Reshapes the grouped data into a pivot table
)
cohort_revenue


# In[68]:


plt.figure(figsize=(14, 7))
plt.title('Net Revenue by Cohort Group')
sns.heatmap(data=cohort_revenue, annot=True, vmin=0.0, fmt=".0f", cmap="YlOrBr")
plt.show()


# In[69]:


cohort_cumulative_revenue = (
    df1.groupby(['FirstOrderDate', 'CohortIndex'])['Sales'].sum()
    .unstack()  # Reshapes the grouped data into a pivot table
    .cumsum(axis=1)  # Calculates the cumulative sum across columns (CohortIndex)
)
cohort_cumulative_revenue


# In[70]:


plt.figure(figsize=(14, 7))
plt.title('Cumulative Lifetime Revenue by Cohort Group')
sns.heatmap(data=cohort_cumulative_revenue, annot=True, vmin=0.0, fmt=".0f", cmap="GnBu")
plt.show()


# In[ ]:




