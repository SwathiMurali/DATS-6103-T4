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

# testing out dropping all missing values
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
df_clean['rel_anime_count'] = df_clean.Related_anime.apply(lambda x: len(x.split(',')) if not isNaN(x) else 0)
df_clean['rel_mang_count'] = df_clean.Related_Mange.apply(lambda x: len(x.split(',')) if not isNaN(x) else 0)
df_clean['voice_actors_count'] = df_clean.Voice_actors.apply(lambda x: len(x.split(',')) if not isNaN(x) else 0)
df_clean['staff_count'] = df_clean.staff.apply(lambda x: len(x.split(',')) if not isNaN(x) else 0)
df_clean['tags_count'] = df_clean.Tags.apply(lambda x: len(x.split(',')) if not isNaN(x) else 0)
df_clean['rel_media_count'] = df_clean.rel_anime_count + df_clean.rel_mang_count

# splits tags by comma
# df_clean.Tags = df_clean.Tags.apply(lambda x: x.split(',') if not isNaN(x) else np.nan)

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
#TV has the most anime shows available for viewing.

# %%
most_anime_studios = df_clean["Studio"].value_counts()[:15]
sns.barplot(y=most_anime_studios.index,x=most_anime_studios)
plt.title("Top 15 Anime Studios")
#The top 15 anime studios by number of anime shows produced. Interestingly there's a Chinese animation studio in the top 15. Chinese anime, also known as donghua, has also become popular in recent years.

# %%
sns.boxplot(x="Type",y="Rating",data=df_clean)
plt.title("Anime Ratings by Type")
#TV Sp, even though there are less anime hosted on it, it tends to get a higher average rating.

# %%
# Correlation matrix
correlation=df_clean.corr(numeric_only=True)
sns.heatmap(correlation,annot=True)
# Rankings are created based on ratings so ignore that correlation. Also ignore the correlation between related media and related anime and manga because 
# those fields make up related media.
# There is an interestingly strong negative correlation between the number of voice actors and rank. The count of staff and voice actors have a relatively 
# high correlation with each other. This makes sense since the more voice actors you have, the more staff you need.
# Thus, we can drop one of these variables during the model-building process.

# %%
sns.histplot(x=df_clean["Episodes"],data=df,bins=500,color='green')
plt.xlim(0,55)
plt.title("Anime Count by Number of Episodes")

# Most anime with only one episode can be considered as movies since they typically have one episode. The next most common episode count is around 12, which is the average length for TV shows or Original Video Animations (OVA).

# %%
# Top 10 anime by number of episodes
df_rank = df_clean[(df_clean.Rank < 11)]
sns.barplot(x='Episodes',y='Name',data=df_rank)
plt.title("Top 10 Anime Number of Episodes")

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
sns.scatterplot(x='Count',y='Studio',hue='Type',data=type_counts,ax=ax)
plt.title('Distribution of anime types by top studios')
plt.legend(bbox_to_anchor=(1,1))

# They mostly produce TV shows and movies.

# -------------- SMART Question 1 ----------------

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

# -------------- Unsure about this section? ----------------
# %%
df_clean["Description"]

# %%
df_clean[df_clean['Description'] == "No synopsis yet - check back soon!"]

# %%
df_clean.dropna(inplace=True,subset=["Description"])
filtering = df_clean[df_clean['Description'] == "No synopsis yet - check back soon!"].index
filtering2 = df_clean[df_clean['Description'] == "'No synopsis yet - check back soon!'"].index
df_clean.drop(filtering,inplace=True)
df_clean.drop(filtering2,inplace=True)
df_clean.reset_index()

# %%
df_clean_values = pd.DataFrame(df_clean['Description'])
df_clean_values['Description'] = df_clean_values['Description'].str.replace(r'[^\w\s]', "")
df_clean

# %%
# Check if the 'punkt' resource is available, and download it if not
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
df_clean_values['Description'] = df_clean_values['Description'].apply(word_tokenize)

# %%
try:
    nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
eng_stopwords = stopwords.words('english') 
df_clean_values['Description'] = df_clean_values['Description'].dropna().apply(lambda words: [word.lower() for word in words if word.lower() not in eng_stopwords])
df_clean_values["Description"]

# %%
df_clean_values["Description"].explode().value_counts()[:30]

# %%
v = TfidfVectorizer(max_features=150,stop_words='english',ngram_range=(1,3))
x = v.fit_transform(df_clean['Description'])
print(x.toarray()[0])

# %%
tfidf_clean_values = pd.DataFrame(x.toarray(), columns=v.get_feature_names_out())
print(tfidf_clean_values)

# %%
df_clean = pd.concat([df_clean.reset_index(),tfidf_clean_values],axis=1)
df_clean

# %%
df_clean.drop(['Description'],axis=1,inplace=True)


#--------------- DATA CLEANING AND TRANSFORMATION ----------------

# %% [markdown]
# Imputing missing values.
# 

# %% [markdown]
# **1) Studio**

# %% [markdown]
# Substituting missing values with a randomly selected studio is done, considering the likelihood of a studio producing an anime.
# %%
temp=df_clean["Studio"].value_counts()
temp

# %%
p = temp.values 
p = p/sum(p)

# %%
size = df_clean["Studio"].isna().sum()

# %%
df_clean.loc[df_clean["Studio"].isna(),"Studio"] = np.random.choice(temp,size,p=p)

# %% [markdown]
# **2) Episodes**

# %% [markdown]
# In this context, opting for either the median or mean would yield negligible variations. However, for avid anime enthusiasts, it's widely acknowledged that anime seasons typically consist of 12 episodes each.
# %%
df_clean["Episodes"].fillna(df_clean["Episodes"].median(),inplace=True)

# %% [markdown]
# **3) Release Year**

# %% [markdown]
# Imputing it with the median.

# %%
df_clean["Release_year"].value_counts()
df_clean["Release_year"].fillna(df_clean["Release_year"].median(),inplace=True)

# %% [markdown]
# # Converting Categorical Variables

# %% [markdown]
# Next, we will transform the tags column into a list of tags to facilitate binary classification through one-hot encoding.
# %%
df_clean["Tags"]=df_clean["Tags"].apply(lambda x:x.split(',') if pd.isna(x)!=True else "")

# %%
mlb = MultiLabelBinarizer()

df_clean = pd.concat([df_clean,pd.DataFrame(mlb.fit_transform(df_clean["Tags"]),columns=mlb.classes_, index=df_clean.index)],axis=1)

# %%
df_clean

# %% [markdown]
# We also create binary columns for Type and Studio, and release season

# %%
df_clean = pd.get_dummies(df_clean,columns = ['Type', 'Studio','Release_season'])
df_clean

# %%
test = pd.Series([x for item in df_clean["Tags"] for x in item]).value_counts()
test.head(20)

# %% [markdown]
# Scaling the columns to avoid adverse effects.

# %%
scaler = preprocessing.MinMaxScaler()
df_clean[["Release_year"]] = scaler.fit_transform(df_clean[["Release_year"]])
df_clean[["Episodes"]] = scaler.fit_transform(df_clean[["Episodes"]])
df_clean[["staff_count"]] = scaler.fit_transform(df_clean[["staff_count"]])
df_clean[["voice_actors_count"]] = scaler.fit_transform(df_clean[["voice_actors_count"]])
df_clean[["rel_anime_count"]] = scaler.fit_transform(df_clean[["rel_anime_count"]])


# -------------- SMART Question 2: Modeling ----------------

# Can we predict the user ratings of anime based on the available information such as the studio, release season, and tags? 
# What factors have the most significant impact on an anime's rating, and can we build a predictive model for anime popularity?

# %% [markdown]
# We'll initially create a DataFrame comprising rows with unavailable ratings. This dataset will be utilized in the final stage to predict the ratings for these anime entries.
# %%
predicting_df_clean = df_clean[df_clean.isnull().any(axis=1)]
predicting_df_clean.drop(["Rating","Name","Rank","Tags","index"],axis=1,inplace=True)
predicting_df_clean.shape

#%%
predicting_df_clean.head()
# %% [markdown]
# Next we split our data for training and testing.

# %%
df_clean.dropna(subset=['Rating'], inplace=True)

X = df_clean.drop(["Rating","Name","Rank","Tags","index"],axis=1)
y=df_clean["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

# %%
X_train

# %% [markdown]
# Building our model and comparing the results

# %%
df_clean.columns[:20]

# %%
model = XGBRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print('Mean Absolute Error:',mae)

# %%
mse = mean_squared_error(preds, y_test)
mse

# %% 
r2 = r2_score(preds,y_test)
print(r2)

# %% 
for i in range(5):
    print('Predicted: {:.2f};\n     Real: {};\n'.format(preds[i], y_test.iloc[i]))

# %% [markdown]
# Our model appears to be performing well. Next, we'll forecast the ratings for the anime entries that were previously unavailable.
# %%
predictions = pd.DataFrame(model.predict(predicting_df_clean))
predictions.columns=["Rating"]
predictions

# We could successfully predict rating of the anime as seen in the above table

# %%
# let's compare the actual and predicted values
preds_df = pd.DataFrame(preds)

# change index
compare = y_test.copy()
compare = pd.DataFrame(compare)
compare.index = range(len(compare))
compare.columns = ['Actual']
compare['Predicted'] = preds_df
compare['Difference'] = compare['Actual'] - compare['Predicted']
compare

# %%
# let's plot the histogram of difference
sns.histplot(compare['Difference'], kde=True)
plt.title("Histogram of Difference between Actual and Predicted Ratings")

# %%
