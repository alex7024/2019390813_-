#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.stats.api as sms
from scipy.stats import shapiro, levene, mannwhitneyu


# In[2]:


#### Load dataset
data = pd.read_csv('/Users/yoonchanghoon/Desktop/ab_data.csv')
data.head()


# In[3]:


#### Check format
data.info()


# In[4]:


#### Check values 
for x in data.columns: 
    print(x)
    print(data[x].values)


# In[5]:


#### Check unique values
data.apply(lambda x: x.nunique())


# In[6]:


#### Check null
data.isnull().sum()


# In[7]:


#### Remove duplicates
print(data.shape)
df = data.drop_duplicates(subset= 'user_id', keep= False)
print(df.shape)


# In[8]:


df[['group', 'landing_page']].value_counts()


# In[9]:


#### Check Mismatch - group & landing page
df_mismatch = df[(df["group"]=="treatment")&(df["landing_page"]=="old_page")
                |(df["group"]=="control")&(df["landing_page"]=="new_page")]
n_mismatch = df_mismatch.shape[0]
print(f"The number of mismatched rows:{n_mismatch} rows" )
print("Percent of mismatched rows:%.2f%%" % (n_mismatch/df.shape[0]*100))


# In[10]:


import pandas as pd

# Function to convert MM:SS.S to total seconds
def convert_to_seconds(timestamp):
    if pd.isnull(timestamp):  # Handle missing values
        return None
    minutes, seconds = map(float, timestamp.split(':'))
    return minutes * 60 + seconds

df['total_seconds'] = df['timestamp'].apply(convert_to_seconds)

print(df)


# In[11]:


# Missing value
df['total_seconds'] = df['total_seconds'].fillna(0)


# In[12]:


df.columns


# In[13]:


df.describe()


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt

# Plot the distribution of total seconds
plt.figure(figsize=(10, 6))
plt.hist(df['total_seconds'], bins=25, alpha=0.8, edgecolor='black')
plt.title('Frequency Distribution of Total Seconds')
plt.xlabel('Total Seconds')
plt.ylabel('Count')
plt.ylim(bottom=0)  # Change starting point
plt.grid(axis='y', linestyle='-', alpha=0.6)
plt.show()


# In[15]:


import pandas as pd
group_counts = df['group'].value_counts()
group_ratios = group_counts / group_counts.sum()
print(group_ratios)


# In[16]:


df.groupby(['group','landing_page']).agg({'landing_page': lambda x: x.value_counts()})


# In[17]:


df.groupby(['group','landing_page']).agg({'landing_page': lambda x: x.value_counts()/group_counts.sum()})


# In[19]:


import seaborn as sns 

page_conversion = df.groupby(['landing_page', 'group'])['converted'].mean().reset_index()

# Plot

sns.barplot(data=page_conversion, x='landing_page', y='converted', hue='group', palette='coolwarm')
plt.title('Conversion Rate by Landing Page')
plt.ylabel('Conversion Rate')
plt.xlabel('Landing Page')
plt.legend(title='Group')
plt.show()


# In[20]:


df.groupby(['group','landing_page']).agg({'converted': 'mean'})


# In[21]:


conversion_summary = df.groupby('group')['converted'].mean().reset_index()
conversion_summary.columns = ['Group', 'Conversion Rate']

print(conversion_summary)


# In[22]:


import seaborn as sns  

sns.barplot(data=conversion_summary, x='Group', y='Conversion Rate', palette='viridis')
plt.title('Conversion Rate by Group')
plt.ylabel('Conversion Rate')
plt.xlabel('Group')
plt.show()


# In[23]:


conversion_rates = df.groupby('group')['converted'].mean()
print(conversion_rates)


# In[24]:


time_metrics = df.groupby('group')['total_seconds'].mean()
print(time_metrics)


# In[25]:


from statsmodels.stats.proportion import proportions_ztest

control = df[df['group'] == 'control']['converted']
treatment = df[df['group'] == 'treatment']['converted']

control_converted = control.sum()
treatment_converted = treatment.sum()

#Size of group 
n_control = len(control)
n_treatment = len(treatment)

stat, p_value = proportions_ztest([control_converted, treatment_converted],
                                   [n_control, n_treatment])
print(f"Z-test Statistic: {stat}, p-value: {p_value:.6f}")


# In[26]:


from scipy.stats import ttest_ind

control_time = df[df['group'] == 'control']['total_seconds']
treatment_time = df[df['group'] == 'treatment']['total_seconds']

t_stat, p_value = ttest_ind(control_time, treatment_time)
print(f"T-test Statistic: {t_stat}, p-value: {p_value:.6f}")


# In[ ]:




