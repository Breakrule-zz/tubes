
# coding: utf-8

# ### 1. IMPORT LIBRARY

# In[30]:


# Import the necessary packages used in this notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import IPython.display as display
url = 'https://raw.githubusercontent.com/Breakrule/database/master/dataFifa.csv'
# Any results you write to the current directory are saved as output.


# **Read the dataset and save it to variable 'datafr'

# In[31]:


#datafr = pd.read_csv("../input/dataFifa.csv", error_bad_lines=False)
datafr = pd.read_csv(url, error_bad_lines=False)


# **Display the structure of the dataset

# In[32]:


print(datafr.head(10))


# **Identifying the shape of the dataset

# In[33]:


# Dimension of the datatset
print("Dimension of the dataset is: ",datafr.shape)


# **Missing values for each attribute

# In[34]:


# Check the missing values in the column
datafr.isnull().sum().sort_values(ascending=False)


# **Above stats shows that there are a lot of empty values and this large number can not be filled, so dropping the most empty columns woud help**

# In[35]:


datafr.drop(['Loaned From'], 1, inplace=True) #axis=1 represents the column
# datafr.drop('Loaned From', axis=1)
# Dropped the column 'Loaned From' as it has 93% of missing data, thus not suitable for our evaluation**


# ### 2. Visualization
# **A. Relationship b/w Jersey Number and Overall 

# In[62]:


plt.figure(figsize=(12,10))
ax = sns.scatterplot(x="Jersey Number", y="Overall", hue ="Overall", size= "Overall", data=datafr)
ax.set_title('Scatter plot of Jersey Number vs Overall', fontsize=16)
sns.set_context("paper", font_scale=1.4)
plt.show()

# This relationship is not clear, as player's jersey number is associated more with size of club 
# and their importance of role or position. 
# Thus we won't be using Jersey Number as our feature.**


# **B. Ratio of players by Nationality

# In[63]:


datafr['Nationality'].value_counts()[:10]


# In[64]:


# Pie plot showing the overall ratio in the dataset
# Data to plot

England = len(datafr[datafr['Nationality'] == 'England'])
Germany = len(datafr[datafr['Nationality'] == 'Germany'])
Spain = len(datafr[datafr['Nationality'] == 'Spain'])
Argentina = len(datafr[datafr['Nationality'] == 'Argentina'])
France = len(datafr[datafr['Nationality'] == 'France'])
Brazil = len(datafr[datafr['Nationality'] == 'Brazil'])
Italy = len(datafr[datafr['Nationality'] == 'Italy'])
Colombia = len(datafr[datafr['Nationality'] == 'Colombia'])
Japan = len(datafr[datafr['Nationality'] == 'Japan'])
Netherlands = len(datafr[datafr['Nationality'] == 'Netherlands'])

labels = 'England','Germany','Spain','Argentina','France','Brazil','Italy','Colombia','Japan','Netherlands'
sizes = [England,Germany,Spain,Argentina,France,Brazil,Italy,Colombia,Japan,Netherlands]
plt.figure(figsize=(6,6))

# Plot
plt.pie(sizes, explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05), labels=labels, colors=sns.color_palette("Purples"),
autopct='%1.1f%%', shadow=True, startangle=90)
sns.set_context("paper", font_scale=1.2)
plt.title('Ratio of players by different Nationality', fontsize=16)
plt.show()


# **In our given dataset, more than 50% of players come from popular countries like England, Germany, Spain, Argentina and France. This could be explained by the popularity and sizes of domestic leagues within these nations.**

# In[65]:


smart_buy = datafr[(datafr['Contract Valid Until']=='2019') & (datafr['Overall']>=75)]
young_buy = smart_buy[smart_buy['Potential']>smart_buy['Overall']]
experience_buy = smart_buy[smart_buy['Age']>30]


# In[66]:


plt.figure(figsize=(12,10))
ax = sns.scatterplot(x="Age", y="Potential", hue ="Overall", size="Overall", data=young_buy)
ax.set_title('Scatter plot of Age vs Potential for Smart Trade of Young players', fontsize=16)
sns.set_context("paper", font_scale=1.4)


# **From the above scatter plot, we get to know that there are few players that a club manager could pursue since their contract is running only till 2019 so either club managers could  negotiate less or directly negotiate with player in January transfer window. We might target those players represented by Purples dots in top left quadrant as their current Overall Rating and future Potential rating is high.**

# In[67]:


plt.figure(figsize=(12,10))
ax = sns.scatterplot(x="Overall", y="Potential", hue ="Age", data=experience_buy)
ax.set_title('Scatter plot of Overall vs Potential for Experience Smart Trade', fontsize=16)
sns.set_context("paper", font_scale=1.4)


# **Drop unnecessary columns

# In[68]:


## lets drop unnecessary columns
datafr.drop(['ID','Unnamed: 0','Weak Foot','Release Clause','Wage','Photo', 'Nationality', 'Flag', 'Club Logo', 'International Reputation', 'Body Type', 'Real Face','Jersey Number', 'Joined','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF',  'RW','LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM','CDM', 'RDM', 'RWB', 'LB', 'LCB','CB', 'RCB', 'RB'],
          axis=1,inplace=True)
datafr.shape
datafr.head()


# **C. Count of players by position & Distribution of players by overall

# In[69]:


sns.set(style="darkgrid")
fig, axs = plt.subplots(nrows=2, figsize=(16, 20))
sns.countplot(datafr['Position'], palette="RdPu", ax=axs[0])
axs[0].set_title('Number of players per position', fontsize=16)
sns.distplot(datafr['Overall'],color="Purple", ax=axs[1])
axs[1].set_title('Distribution of players by Overall', fontsize=16)


# **D. Youth prospects with high potential growth
# **Create a dataset for young prospects

# In[70]:


youth_special = datafr[(datafr['Overall']>75) & (datafr['Potential'] - datafr['Overall']>=10)].sort_values(by='Overall',ascending=False)
cols = ['Name','Club','Age','Overall','Potential','Position','Value']
youth_special[cols]


# In[71]:


sns.set(style="darkgrid")
fig, axs = plt.subplots(nrows=2, figsize=(16, 20))
sns.countplot(youth_special['Position'], palette="PuRd", ax=axs[0])
axs[0].set_title('Number of young prospects per position', fontsize=16)
sns.distplot(youth_special['Overall'],color="Red", ax=axs[1])
axs[1].set_title('Distribution of young prospects by Overall', fontsize=16)


# **E. Ratio of youth prospects by Position

# In[72]:


youth_special['Position'].unique()


# In[73]:


# Data to plot
GK = len(youth_special[youth_special['Position'] == 'GK'])
LM = len(youth_special[youth_special['Position'] == 'LM'])
RB = len(youth_special[youth_special['Position'] == 'RB'])
CB = len(youth_special[youth_special['Position'] == 'CB'])
LW = len(youth_special[youth_special['Position'] == 'LW'])
RCM = len(youth_special[youth_special['Position'] == 'RCM'])
CM = len(youth_special[youth_special['Position'] == 'CM'])
LCB = len(youth_special[youth_special['Position'] == 'LCB'])
RS = len(youth_special[youth_special['Position'] == 'RS'])
RM = len(youth_special[youth_special['Position'] == 'RM'])
ST = len(youth_special[youth_special['Position'] == 'ST'])
CDM = len(youth_special[youth_special['Position'] == 'CDM'])
LB = len(youth_special[youth_special['Position'] == 'LB'])
CAM = len(youth_special[youth_special['Position'] == 'CAM'])

labels = 'GK', 'LM', 'RB', 'CB', 'LW', 'RCM', 'CM', 'LCB', 'RS', 'RM', 'ST', 'CDM', 'LB', 'CAM'
sizes = [GK,LM,RB,CB,LW,RCM,CM,LCB,RS,RM,ST,CDM,LB,CAM]
plt.figure(figsize=(6,6))

# Plot
plt.pie(sizes, explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05), labels=labels, colors=sns.color_palette("Purples"),
autopct='%1.1f%%', shadow=True, startangle=90)
sns.set_context("paper", font_scale=1.2)
plt.title('Ratio of young prospects by different Positions', fontsize=16)
plt.show()


# # 3. Predictions

# In[74]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# In[75]:


datafr.columns


# **Selecting columns to find similarity among players

# In[76]:


attributes = datafr.iloc[:, 14:]
attributes['Skill Moves'] = datafr['Skill Moves']
attributes['Age'] = datafr['Age']
workrate = datafr['Work Rate'].str.get_dummies(sep='/ ')
attributes = pd.concat([attributes, workrate], axis=1)
df = attributes
attributes = attributes.dropna()
df['Name'] = datafr['Name']
df['Position'] = datafr['Position']
df = df.dropna()
print(attributes.columns)


# **Displaying our attribute set

# In[77]:


attributes.head()


# **Correlation Matrix based on attribute set

# In[78]:


plt.figure(figsize=(9,9))

# Compute the correlation matrix
corr = attributes.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap="RdBu", vmax=.3, center=0,
            square=True, linewidths=.7, cbar_kws={"shrink": .7})


# **From the above correlation chart, we can see a lot of Goalkeepers attributes have a negative correlation with the attributes possessed by a Forward, Midfielder and Defender.**
# 
# **Standardize the dataset

# In[79]:


scaled = StandardScaler()
X = scaled.fit_transform(attributes)


# **Create recommendations using NearestNeighbors ML

# In[80]:


recommendations = NearestNeighbors(n_neighbors=5,algorithm='kd_tree')
recommendations.fit(X)


# **Get Similar Players index

# In[81]:


player_index = recommendations.kneighbors(X)[1]


# In[82]:


player_index


# **Define a recommend function to display results

# In[83]:


def get_index(x):
    return df[df['Name']==x].index.tolist()[0]

def recommend_similar(player):
    print("These are 4 players similar to {} : ".format(player))
    index=  get_index(player)
    for i in player_index[index][1:]:
        print("Name: {0}\nPosition: {1}\n".format(df.iloc[i]['Name'],df.iloc[i]['Position']))


# **Test 1 : Eden Hazard**

# In[84]:


recommend_similar('E. Hazard')


# **Test 2 : Mohamed Salah**

# In[85]:


recommend_similar(player='M. Salah')


# **Test 3 : Manuel Neuer**

# In[86]:


recommend_similar('M. Neuer')


# **Test 4: Joe Gomez (Young Prospect)**

# In[87]:


recommend_similar('J. Gomez')

