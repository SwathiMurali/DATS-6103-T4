# This will be our main file for our Anime project for 6103.

#%%
#Loading libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import r2_score

#%%
#Loading and Understanding the Data
df = pd.read_csv('Anime.csv')
corr_df=df
# %%
df.head()
# %%
df.tail()
# %%
df.shape
# %%
df.columns
for column in df.columns:
    print(column)
# %%
df.dtypes
# %%
df.info()
# %%
df.describe()
# %%
df.isnull().sum().sort_values(ascending = False)
# %%
df_new = df.dropna()
# %%
df_new.isnull().sum()
# %%
df_new.info()
# %%
df_new.shape  # having too much small rows. so for the prediction the testing and training we get bad scores
# %%
# Identify numeric columns
numeric_columns = df.select_dtypes(include='number').columns

# Filter the DataFrame to include only numeric columns
numeric_df = df[numeric_columns]

# Filter rows where 'Release_year' is 2020.0 and then sum the values
result = numeric_df[df['Release_year'] == 2020.0].sum()

# Print the result or use it as needed
print(result)
# %%
df[df['Release_season'] == 'Winter'].select_dtypes(include='number').sum()
# %%
