#!/usr/bin/env python
# coding: utf-8

# ## Hypothesis
# 
# #### 1. Average abv of the beers in the world is 5%. This sample dataset represents population
# #### 2. The overall characteristics and your personal experience of the beer is strongly influenced by aroma and taste
# #### 3. Stronger beers (above 10% abv) are rated 5 for overall charecteristics
# #### 4. More the review time, better is the rating i.e beers rated 4, 5 for overall charecteristics has taken more time for review 
# #### 5. Atleast 80% of the beers have abv between 4% - 7%
# 

# # 

# ## Importing Libraries & Data

# In[1]:


import pandas as pd 
import numpy as np #For Mathematic calulcation
import seaborn as sns #For data visualization
import matplotlib.pyplot as plt #For plotting graph
import os #For changing directory

import sklearn #Scikit-Learn for Model Building
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import statsmodels.api as sm

def summary_stats(col):
    """
    Computes summary statistics (min, max, quartiles) of a column in a database.
    Args: col (pandas.Series): The column of the database to analyze.
    Returns: dict: A dictionary containing the summary statistics.
    """
    
    # Compute summary statistics
    min_val = col.min()
    max_val = col.max()
    quartiles = np.percentile(col, [25, 50, 75, 90, 95, 99, 10])
    
    # Create dictionary of summary statistics
    stats_dict = {
        'Min': min_val,
        '10%ile': quartiles[6],
        'Q1': quartiles[0],
        'Q2 (Median)': quartiles[1],
        'Q3': quartiles[2],
        '90%ile': quartiles[3],
        '95%ile': quartiles[4],
        '99%ile': quartiles[5],
        'Max': max_val
    }
    
    return stats_dict


# In[2]:


path1="......../Python_Beer_Review" #Insert folder where csv file is stored
os.chdir(path1) #Directory path1 is set


# In[3]:


db = pd.read_csv("beer_reviews.csv")


# ## Understanding the data

# In[4]:


db.columns


# In[5]:


db.shape


# In[6]:


db.dtypes


# In[7]:


print(db.head(5))


# ### Beers are rated for aroma, appearance, taste, palate and overall experiences
# ### These rating must be 1 - 5

# In[8]:


print("Unique (Brewery) = ", db["brewery_name"].nunique())
print("Unique (Beer) = ", db["beer_name"].nunique())
print("Unique (Beer_Style) = ", db["beer_style"].nunique())
print("Unique (Beer_Abv) = ", db["beer_abv"].nunique())
print("Unique (review_profilename) = ", db["review_profilename"].nunique())

print("Unique (review_overall) = ", db["review_overall"].nunique())
print("Unique (review_aroma) = ", db["review_aroma"].nunique())
print("Unique (review_appearance) = ", db["review_appearance"].nunique())
print("Unique (review_palate) = ", db["review_palate"].nunique())
print("Unique (review_taste) = ", db["review_taste"].nunique())


# ### review_overall & review_appearance seems to have a outlier
# 
# 
# # EDA
# # Missing Values & Outliers treatment

# In[9]:


#Finding percent missing values in each column
missing_counts = db.isna().sum() #Count of missing values
missing_percentages = (missing_counts / len(db)) * 100 #Percentage of missing values

missing_data = pd.concat([missing_counts, missing_percentages], axis=1) #Combining both
missing_data.columns = ['missing_count', 'missing_percentage'] #Labeling

print(missing_data)


# In[10]:


#Since the missing values are not much, filling categorical & ordinal with MODE
# NOTE: We use [0] with mode() because mode() returns a series of value and we take the first one with [0]
db["brewery_name"].fillna(db["brewery_name"].mode()[0], inplace = True)
db["review_profilename"].fillna(db["review_profilename"].mode()[0], inplace = True)

#Filling numerical with MEDIAN
db["beer_abv"].fillna(db["beer_abv"].median(), inplace = True)

db.isnull().sum()


# In[11]:


print(db["review_overall"].value_counts())

db = db[db['review_overall'] != 0]
print(db["review_overall"].value_counts())
print(db["review_overall"].value_counts(normalize=True))

ax = db["review_overall"].value_counts(normalize=True).plot.bar(title = "review_overall")
ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
plt.show()


# In[12]:


print(db["review_appearance"].value_counts(normalize=True))

ax = db["review_appearance"].value_counts(normalize=True).plot.bar(title = "review_appearance")
ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
plt.show()


# In[13]:


print(db["review_aroma"].value_counts(normalize=True))

ax = db["review_aroma"].value_counts(normalize=True).plot.bar(title = "review_aroma")
ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
plt.show()


# In[14]:


print(db["review_palate"].value_counts(normalize=True))

ax = db["review_palate"].value_counts(normalize=True).plot.bar(title = "review_palate")
ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
plt.show()


# In[15]:


print(db["review_taste"].value_counts(normalize=True))

ax = db["review_taste"].value_counts(normalize=True).plot.bar(title = "review_taste")
ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
plt.show()


# ## Findings
# 
# #### taste: 85% of the rating is between 3 - 4.5
# #### palate: 85% of the rating is between 3 - 4.5
# #### aroma: 85% of the rating is between 3 - 4.5
# #### appearance: 90% of the rating is between 3 - 4.5
# #### overall: 85% of the rating is between 3 - 4.5

# # Hypothesis Testing

# ### Hypothesis 01: Average abv of the beers in the world is 5%. This sample dataset represents population

# In[16]:


summary_stats(db["beer_abv"])


# In[17]:


sns.displot(db["beer_abv"])


# ### Doing 2 tail test to check the hypothesis
# #### Null Hypothesis: Avg of sample is 5%
# #### Alternate Hypothesis: Avg os sample is not 5%
# #### Significance level (alpha) = 5%
# 
# #### The abv seems to be normally distributed as seen from the histogram plot above, applying two tailed z-test
# #### Also, since the sample size >>>>> 30, we can assume it is normally distributed

# In[18]:


abv_mean = db["beer_abv"].mean()
abv_std = db["beer_abv"].std()
pop_mean = 5
sample_size = len(db)

z_score = (abv_mean - pop_mean) / (abv_std / np.sqrt(sample_size))
p_value = stats.norm.sf(abs(z_score))*2

if p_value < 0.025 and p_value > 0.0975:
    print("The sample mean is significantly different from the population mean.")
else:
    print("The sample mean is not significantly different from the population mean.")


# #### The sample is not significantly different from the population mean. Thus, it represents the population.

# ### Hypothesis 02: The overall characteristics and your personal experience of the beer is strongly influenced by aroma and taste

# In[19]:


import statsmodels.api as sm

x = db[['review_aroma', 'review_appearance', 'review_taste', 'review_palate']]
y = db[['review_overall']]

model_results  = sm.OLS(y, x).fit()
y_pred = model_results.predict(x)

# find p-values of the coefficients
p_values = model_results.pvalues

# find confidence intervals of the coefficients
conf_intervals = model_results.conf_int()

# create a summary table of the regression results including p-values and
# confidence intervals
summary_table = pd.concat([model_results.params,
                           conf_intervals,
                           p_values],
                          axis=1, keys=['Coefficients', 'Confidence Interval', 'P-Value'])
print(summary_table)
print("\n r2:", r2_score(y, y_pred))
print("\n mse:", mean_squared_error(y, y_pred))


# #### All variables have p-values=0, indicating that they are statistically significant predictors of overall
# #### The value of r2> 0.6, we can consider this model to be a good fit for the data.
# #### Looking at the coefficient of the parameters of the model, palate & taste has strong relationship with overall quality
# #### Thus, hypothesis 02 is partially correct - Taste influemces the overall but aroma doesn't

# ### Hypothesis 03: Stronger beers (above 10% abv) are rated 5 for overall charecteristics

# In[20]:


strong_beer = db[db['beer_abv'] > 10]
print("ABV: \n", summary_stats(strong_beer['beer_abv']))
print("\n Overall: \n", summary_stats(strong_beer['review_overall']))


# #### The hypothesis is not true as 90% of the data has rating below 5 for overall quality

# ### Hypothesis 04: More the review time, better is the rating i.e beers rated 4, 5 for overall charecteristics has taken more time for review

# In[21]:


plt.figure(figsize=(12,10))
sns.scatterplot(x="review_overall", y="review_time", data=db)
plt.title("Overall vs Time", fontsize=12)
plt.show()


# #### There is no relationship between the time taken for review and the overall quality. Hypothesis 04 is not true.

# ### Hypothesis 05: Atleast 80% of the beers have abv between 4% - 7%

# In[22]:


summary_stats(db["beer_abv"])


# #### Less than 75% of the beer has abv < 7%, thus, hypothesis 5 rejected

# # Feature Engineering - Adding a new rating system
# 
# #### The new rating system is dervied from all 5 ratings - aroma, appearance, taste, palate and overall. It is taken as weighted sum of each variable. 
# 
# | Previous rating | Weights |
# | --------------- | ------- |
# | Appearance      | 10%     |
# | Palate          | 10%     |
# | Aroma           | 20%     |
# | Taste           | 20%     |
# | Overall         | 40%     |
# 
# #### Source: https://www.ratebeer.com/our-scores

# In[23]:


db['rating'] = round((0.1 * db['review_appearance']) + (0.2 * db['review_aroma']) + (0.2 * db['review_taste']) + (0.1 * db['review_palate']) + (0.4 * db['review_overall']),2)
sns.displot(db['rating'])
print("Summary of rating: \n", summary_stats(db['rating']))


# In[24]:


finest = db[db['rating'] == 5]
print('\n No. of 5* Beer:',len(finest))
print('\n % of 5* Beer:',round(len(finest)/len(db)*100,2),"%")
print('\n Mean of abv for 5* Beer:',round(finest["beer_abv"].mean(),2))
print("\n Summary of ABV: \n", summary_stats(finest["beer_abv"]))
finest


# In[25]:


sns.displot(finest["beer_abv"])


# ## 3 Wierd Beer

# In[26]:


#Beer with maximum alcohol content
a = finest[finest['beer_abv'] == 41]
print(a[["brewery_name", "beer_style", "beer_name", "beer_abv"]])

#Beer with minimum alcohol content
b = finest[finest['beer_abv'] == 0.5]
print(b[["brewery_name", "beer_style", "beer_name", "beer_abv"]])

#Beer with mean of the abv
c = finest[finest['beer_abv'] == 21]
print(c[["brewery_name", "beer_style", "beer_name", "beer_abv"]])


# ## Beer I would recommend

# In[27]:


style = finest["beer_style"].value_counts()
style[style == 1]


# In[28]:


style1 = finest[finest['beer_style'] == 'Kristalweizen']
print(style1[["brewery_name", "beer_style", "beer_name", "beer_abv"]])

style2 = finest[finest['beer_style'] == 'English Pale Mild Ale']
print(style2[["brewery_name", "beer_style", "beer_name", "beer_abv"]])

style3 = finest[finest['beer_style'] == 'English Stout']
print(style3[["brewery_name", "beer_style", "beer_name", "beer_abv"]])

style4 = finest[finest['beer_style'] == 'Keller Bier / Zwickel Bier']
print(style4[["brewery_name", "beer_style", "beer_name", "beer_abv"]])


# # Brewery with Best Beer & KPI 1

# In[29]:


c1 = finest["brewery_name"].value_counts()
c2 = pd.DataFrame({"brewery_name": c1.index, "fin_counts": c1.values})

c3 = db["brewery_name"].value_counts()
c4 = pd.DataFrame({"brewery_name": c3.index, "db_counts": c3.values})

merged = pd.merge(c2, c4, on="brewery_name", how="left")
merged = pd.merge(merged[['brewery_name', 'fin_counts',  'db_counts']], finest[['brewery_name', 'beer_abv']], on="brewery_name", how="left")
print(merged.head(5))

summary_stats(merged["db_counts"])


# In[30]:


merged['KPI_1'] = merged["fin_counts"]/merged["db_counts"]*100 
merged = merged.sort_values(by=["fin_counts", "db_counts", "KPI_1"], ascending=False)
merged.to_excel('merged.xlsx')

#Best Brewery is Brouwerij Westvleteren (Sint-Sixtusabdij van Westvleteren) has the highest % of 5* beers (7%) 
#with considerable number of total beer reviews (171 out of 2378)

summary_stats(merged["KPI_1"])


# ## KPI 2

# In[31]:


s1 = finest["beer_style"].value_counts()
s2 = pd.DataFrame({"beer_style": s1.index, "fin_counts": s1.values})

s3 = db["beer_style"].value_counts()
s4 = pd.DataFrame({"beer_style": s3.index, "db_counts": s3.values})

KPI2 = pd.merge(s2, s4, on="beer_style", how="left")
KPI2 = pd.merge(KPI2[['beer_style', 'fin_counts',  'db_counts']], finest[['beer_style', 'beer_abv']], on="beer_style", how="left")
print(KPI2.head(5))

summary_stats(KPI2["db_counts"])


# In[32]:


KPI2['KPI_2'] = KPI2["fin_counts"]/KPI2["db_counts"]*100 
KPI2 = KPI2.sort_values(by=["fin_counts", "db_counts", "KPI_2"], ascending=False)
print(KPI2)

summary_stats(KPI2["KPI_2"])

