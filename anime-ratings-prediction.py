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
df_new.shape  

# not enough rows, so for the prediction the testing and training we get bad scores.
# go back to the main dataset

# ----------- Is any of the following section necessary for anything? -------------

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
# Numeric columns in winter
df[df['Release_season'] == 'Winter'].select_dtypes(include='number').sum()
# %%
# List of movies
df[df['Type'] == 'Movie'].head()

# %%
# List of those with rating > 4.5
df[df['Rating'] > 4.50].head()
# %%
df1 = pd.DataFrame(df['Rating'])
df1.sum()

# -------------- Feature Engineering ----------------    
# %%
# create dataframe with cleaned data
df_clean = df.copy()

# %%
# removes spaces in the type column
df_clean.Type = df_clean.Type.apply(lambda x: x.strip())

# new columns with count of related anime, manga, voice actors, staff, and tags
def isNaN(num):
    return num != num

#df_clean['warnings_count'] = df_clean.Content_Warning.apply(lambda x: len(x.split(',,')) if not isNaN(x) else 0)
df_clean['rel_anim_count'] = df_clean.Related_anime.apply(lambda x: len(x.split(',')) if not isNaN(x) else 0)
df_clean['rel_mang_count'] = df_clean.Related_Mange.apply(lambda x: len(x.split(',')) if not isNaN(x) else 0)
df_clean['voice_act_count'] = df_clean.Voice_actors.apply(lambda x: len(x.split(',')) if not isNaN(x) else 0)
df_clean['staff_count'] = df_clean.staff.apply(lambda x: len(x.split(',')) if not isNaN(x) else 0)
df_clean['tags_count'] = df_clean.Tags.apply(lambda x: len(x.split(',')) if not isNaN(x) else 0)
df_clean['rel_media_count'] = df_clean.rel_anim_count + df_clean.rel_mang_count

# splits tags by comma
df_clean.Tags = df_clean.Tags.apply(lambda x: x.split(',') if not isNaN(x) else np.nan)

# -------------- Data Cleaning ----------------    
# %%
# dropped the columns that are not needed
df_clean.drop(['Japanese_name','End_year','Content_Warning','Related_Mange','Related_anime', 'Voice_actors', 'staff'],axis=1,inplace=True)

df_clean.head()

# %%
# check dataset for null values
df_clean.isna().sum()

# %%
df_clean.shape

# -------------- Visualization ---------------- 
# %%
# Histogram of ratings
sns.histplot(df_clean["Rating"].dropna(),kde=True)
plt.title("Anime by Ratings")
#Most of the anime had ratings between 3 and 4, suggesting that most anime had middling ratings. This also shows a approximately normal distribution.

# %%
# Histogram of release year
sns.histplot(df_clean["Release_year"].dropna(),kde=False)
plt.title("Anime by Release Year")
#Most of the anime were released in the past 10-15 years. This makes sense since there was a boom in the manga and anime industry in the late 2010's.

# %%
sns.countplot(x="Type",data=df_clean)
plt.title("Anime by Type")
#Most anime are released as TV shows.

# %%
most_anime_studios = df_clean["Studio"].value_counts()[:15]
sns.barplot(y=most_anime_studios.index,x=most_anime_studios)
plt.title("Top 15 Anime Studios")
#The top 15 anime studios by number of anime produced. Toei Animation is the most prolific studio, followed by Sunrise and J.C. Staff.

# %%
sns.boxplot(x="Type",y="Rating",data=df_clean)
#TV Sp, even though there are less anime hosted on it, it tends to get a higher average rating.

# %%
# Correlation matrix
correlation=df_clean.corr(numeric_only=True)
sns.heatmap(correlation,annot=True)
# Rankings are created based on ratings so ignore that correlation. Also ignore the correlation between related media and related anime and manga because 
# those fields make up related media.
#There is an interestingly strong negative correlation between the number of voice actors and rank. The count of staff and voice actors have a relatively 
# high correlation with each other. This makes sense since the more voice actors you have, the more staff you need.

# %%
sns.histplot(x=df_clean["Episodes"],data=df,bins=500,color='green')
plt.xlim(0,55)
plt.title("Anime Count by Number of Episodes")

# %%
# Top 10 anime by number of episodes
df_rank = df_clean[(df_clean.Rank < 11)]
sns.barplot(x='Name',y='Episodes',data=df_rank)
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.title("Top 10 Anime by Number of Episodes")

# %%
# Anime Ratings by release season
plt.title("Anime Ratings by Release Season", size=10)

sns.boxplot(x='Release_season', y='Rating', data=df_clean , palette='rainbow')
# there's not much difference in the ratings by season

# %%
# Let's try by TV and Web
plt.title("Anime Ratings by Release Season", size=10)

data_for_box = df_clean[df_clean.Type.isin(['TV', 'Web'])]

sns.boxplot(x='Type', y='Rating', hue='Release_season', data=data_for_box , palette='rainbow')

# %%
# Anime types by top 10 studios
anime_counts=df_clean.groupby(['Studio','Release_year']).size().reset_index(name='Anime_produced')
studio_total=anime_counts.groupby('Studio')['Anime_produced'].sum().reset_index(name='Total anime')
top_studio=studio_total.nlargest(10,'Total anime')

top_10_studio=df_clean[df_clean['Studio'].isin(top_studio['Studio'])]

type_counts=top_10_studio.groupby(['Studio','Type']).size().reset_index(name='Count')

fig,ax=plt.subplots(figsize=(5,5))
sns.scatterplot(x='Studio',y='Count',hue='Type',data=type_counts,ax=ax)
plt.xticks(rotation=90)
plt.title('Distribution of anime types by each studio')
plt.legend(bbox_to_anchor=(1,1))

# They mostly produce TV shows and movies.

# -------------- SMART Question ----------------

# How does the type of anime (e.g., TV series, movie, OVA) relate to the average user ratings on Anime Planet? 
# Are there significant differences in ratings based on the type of anime?

# %%
mean_type = df_clean.groupby('Type')['Rating'].mean().sort_values(ascending= False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Rating', y='Type', data=df_clean, order=mean_type.index, errorbar=None)
plt.title('Mean Ratings by Type')
plt.xlabel('Mean Rating')
plt.ylabel('Type')
plt.show()

# DVD S: This type has the highest mean rating of 3.541850.
# TV: The mean rating for TV shows is 3.476540, making it the second-highest.
# Movie: Movies have a mean rating of 3.469146.
# TV Sp: TV specials have a mean rating of 3.447844.
# Web: Web-based content has a mean rating of 3.353540.
# Music: The mean rating for music-related content is 3.302360.
# OVA: Original Video Animations have a mean rating of 3.285362.
# Other: The 'Other' category has the lowest mean rating among the listed types at 3.261346.
# This output provides insights into the average ratings for different types of content, sorted from highest to lowest mean rating.

# -------------- Modeling ----------------

# Can we predict the user ratings of anime based on the available information such as the studio, release season, and tags? 
# What factors have the most significant impact on an anime's rating, and can we build a predictive model for anime popularity?