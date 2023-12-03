# %%
import numpy as np
import pandas as pd 
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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import r2_score
# %% [markdown]
# # Loading and Understanding the Data

df_anime = pd.read_csv('Anime.csv')
df_anime.head()

# %%
df_anime.shape

# %%
df_anime.info()

# %%
df_anime.isnull().sum()

# %%
df_anime.describe()

#%%
df_anime.isnull().sum().sort_values(ascending = False)
# %%
df_anime_new = df_anime.dropna()
# %%
df_anime_new.isnull().sum()
# %%
df_anime_new.info()
# %%
df_anime_new.shape  
# not enough rows, so for the prediction the testing and training we get bad scores.
# go back to the main dataset

# %% [markdown]
# # Feature engineering

# %% [markdown]
# Removing unnecessary spaces in the type column

# %%
df_anime.Type = df_anime.Type.apply(lambda x: x.strip())

# %%
temp = df_anime.isnull().mean()*100
temp

# %% [markdown]
# We can observe that there are several columns have missing values exceeding 50%, and so we will remove them.
# %%
df_anime.drop(['Japanese_name','Release_season','End_year','Content_Warning','Related_Mange'],axis=1,inplace=True)

# %% [markdown]
#Now, let's consider the number of voice actors and staff members. 
# 
# Typically, a higher count of these indicates a larger budget for the anime, increasing the likelihood of it becoming popular and receiving a favorable rating. 
# 
# Additionally, examining the count of related anime is valuable since more seasons usually correlate with higher overall popularity.
# %%
df_anime['voice_actors_count'] = df_anime.Voice_actors.apply(lambda x: len(x.split(',')) if pd.isna(x)!=True else 0)

df_anime['staff_count'] = df_anime.staff.apply(lambda x: len(x.split(',')) if pd.isna(x)!=True else 0)

df_anime['rel_anime_count'] = df_anime.Related_anime.apply(lambda x: len(x.split(',')) if pd.isna(x)!=True else 0)

# %%
df_anime.drop(['Voice_actors','staff','Related_anime'],axis=1,inplace=True)
# %% [markdown]
# # Exploratory Data Analysis

# %%
sns.histplot(df_anime["Rating"].dropna(),color='darkred',kde=True)

# %% [markdown]
# - The majority of anime tend to fall within the rating range of 3 to 4, indicating that only a small number of anime received exceptionally high ratings.
# %%
sns.histplot(df_anime["Release_year"].dropna(),color='darkred',kde=False)

# %% [markdown]
# - Many of the anime were released within the last 10 to 15 years, which aligns with the significant growth observed in the manga and anime industry during the late 2010s.
# %%
sns.countplot(x="Type",data=df_anime)

# %% [markdown]
# - It looks like TV has the most anime shows available for viewing.

# %%
most_anime_studios = df_anime["Studio"].value_counts()[:15]
sns.barplot(y=most_anime_studios.index,x=most_anime_studios)

# %% [markdown]
# - We can notice that there's a Chinese animation studio producing more than 700 animated shows.
# %%
sns.boxplot(x="Type",y="Rating",data=df_anime)

# %% [markdown]
# - Not much we can infer from this except tha despite hosting fewer anime shows, the TV special category tends to receive higher average ratings.
# %%
sns.histplot(x=df_anime["Episodes"],data=df_anime,bins=500,color='darkred')
plt.xlim(0,55)

# %% [markdown]
# - Most anime with only one episode can be considered as movies since they typically have one episode. The next most common episode count is around 12, which is the average length for TV shows or Original Video Animations (OVA).
# %%
correlation=df_anime.corr(numeric_only=True)
sns.heatmap(correlation,annot=True)

# %% [markdown]
# - Rankings are created based on ratings so ignore that correlation. Also ignore the correlation between related media and related anime and manga because those fields make up related media.
# - There is an interestingly strong negative correlation between the number of voice actors and rank. The count of staff and voice actors have a relatively high correlation with each other. 
# - This makes sense since the more voice actors you have, the more staff you need.
# - Thus, we can drop one of these variables during the model-building process.

#%%
# Top 10 anime by number of episodes
df_rank = df_anime[(df_anime.Rank < 11)]
sns.barplot(x='Name',y='Episodes',data=df_rank)
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.title("Top 10 Anime by Number of Episodes")

#%%
# Anime types by top 10 studios
anime_counts=df_anime.groupby(['Studio','Release_year']).size().reset_index(name='Anime_produced')
studio_total=anime_counts.groupby('Studio')['Anime_produced'].sum().reset_index(name='Total anime')
top_studio=studio_total.nlargest(10,'Total anime')

top_10_studio=df_anime[df_anime['Studio'].isin(top_studio['Studio'])]

type_counts=top_10_studio.groupby(['Studio','Type']).size().reset_index(name='Count')

fig,ax=plt.subplots(figsize=(5,5))
sns.scatterplot(x='Studio',y='Count',hue='Type',data=type_counts,ax=ax)
plt.xticks(rotation=90)
plt.title('Distribution of anime types by each studio')
plt.legend(bbox_to_anchor=(1,1))

#%%[markdown]
# They mostly produce TV shows and movies.

#### SMART Question:

# How does the type of anime (e.g., TV series, movie, OVA) relate to the average user ratings on Anime Planet?
#
# Are there significant differences in ratings based on the type of anime?

#%%
mean_type = df_anime.groupby('Type')['Rating'].mean().sort_values(ascending= False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Rating', y='Type', data=df_anime, order=mean_type.index, errorbar=None)
plt.title('Mean Ratings by Type')
plt.xlabel('Mean Rating')
plt.ylabel('Type')
plt.show()

#%%[markdown]
# - DVD S: This type has the highest mean rating of 3.541850.
# - TV: The mean rating for TV shows is 3.476540, making it the second-highest.
# - Movie: Movies have a mean rating of 3.469146.
# - TV Sp: TV specials have a mean rating of 3.447844.
# - Web: Web-based content has a mean rating of 3.353540.
# - Music: The mean rating for music-related content is 3.302360.
# - OVA: Original Video Animations have a mean rating of 3.285362.
# - Other: The 'Other' category has the lowest mean rating among the listed types at 3.261346.
# - This output provides insights into the average ratings for different types of content, sorted from highest to lowest mean rating.
# %%
df_anime["Description"]

# %%
df_anime[df_anime['Description'] == "No synopsis yet - check back soon!"]

# %%
df_anime.dropna(inplace=True,subset=["Description"])
filtering = df_anime[df_anime['Description'] == "No synopsis yet - check back soon!"].index
filtering2 = df_anime[df_anime['Description'] == "'No synopsis yet - check back soon!'"].index
df_anime.drop(filtering,inplace=True)
df_anime.drop(filtering2,inplace=True)
df_anime.reset_index()

# %%
df_anime_values = pd.DataFrame(df_anime['Description'])
df_anime_values['Description'] = df_anime_values['Description'].str.replace(r'[^\w\s]', "")
df_anime

# %%
# Check if the 'punkt' resource is available, and download it if not
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
df_anime_values['Description'] = df_anime_values['Description'].apply(word_tokenize)

# %%
try:
    nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
eng_stopwords = stopwords.words('english') 
df_anime_values['Description'] = df_anime_values['Description'].dropna().apply(lambda words: [word.lower() for word in words if word.lower() not in eng_stopwords])
df_anime_values["Description"]

# %%
df_anime_values["Description"].explode().value_counts()[:30]

# %%
v = TfidfVectorizer(max_features=150,stop_words='english',ngram_range=(1,3))
x = v.fit_transform(df_anime['Description'])
print(x.toarray()[0])

# %%
tfidf_anime_values = pd.DataFrame(x.toarray(), columns=v.get_feature_names_out())
print(tfidf_anime_values)

# %%
df_anime = pd.concat([df_anime.reset_index(),tfidf_anime_values],axis=1)
df_anime

# %%
df_anime.drop(['Description'],axis=1,inplace=True)

# %% [markdown]
# # DATA CLEANING AND TRANSFORMATION

# %% [markdown]
# Imputing missing values.
# 

# %% [markdown]
# **1) Studio**

# %% [markdown]
# Substituting missing values with a randomly selected studio is done, considering the likelihood of a studio producing an anime.
# %%
temp=df_anime["Studio"].value_counts()
temp

# %%
p = temp.values 
p = p/sum(p)

# %%
size = df_anime["Studio"].isna().sum()

# %%
df_anime.loc[df_anime["Studio"].isna(),"Studio"] = np.random.choice(temp,size,p=p)

# %% [markdown]
# **2) Episodes**

# %% [markdown]
# In this context, opting for either the median or mean would yield negligible variations. However, for avid anime enthusiasts, it's widely acknowledged that anime seasons typically consist of 12 episodes each.
# %%
df_anime["Episodes"].fillna(df_anime["Episodes"].median(),inplace=True)

# %% [markdown]
# **3) Release Year**

# %% [markdown]
# Imputing it with the median.

# %%
df_anime["Release_year"].value_counts()
df_anime["Release_year"].fillna(df_anime["Release_year"].median(),inplace=True)

# %% [markdown]
# # Converting Categorical Variables

# %% [markdown]
# Next, we will transform the tags column into a list of tags to facilitate binary classification through one-hot encoding.
# %%
df_anime["Tags"]=df_anime["Tags"].apply(lambda x:x.split(',') if pd.isna(x)!=True else "")

# %%
mlb = MultiLabelBinarizer()

df_anime = pd.concat([df_anime,pd.DataFrame(mlb.fit_transform(df_anime["Tags"]),columns=mlb.classes_, index=df_anime.index)],axis=1)

# %%
df_anime

# %% [markdown]
# We also create binary columns for Type and Studio

# %%
df_anime = pd.get_dummies(df_anime,columns = ['Type', 'Studio'])
df_anime

# %%
test = pd.Series([x for item in df_anime["Tags"] for x in item]).value_counts()
test.head(20)

# %% [markdown]
# Scaling the columns to avoid adverse effects.

# %%
scaler = preprocessing.MinMaxScaler()
df_anime[["Release_year"]] = scaler.fit_transform(df_anime[["Release_year"]])
df_anime[["Episodes"]] = scaler.fit_transform(df_anime[["Episodes"]])
df_anime[["staff_count"]] = scaler.fit_transform(df_anime[["staff_count"]])
df_anime[["voice_actors_count"]] = scaler.fit_transform(df_anime[["voice_actors_count"]])
df_anime[["rel_anime_count"]] = scaler.fit_transform(df_anime[["rel_anime_count"]])

# %% [markdown]
# # Building a Model
#### SMART Questions:
#
# Can we predict the user ratings of anime based on the available information such as the studio, release season, and tags? 
#
# What factors have the most significant impact on an anime's rating, and can we build a predictive model for anime popularity?
# %% [markdown]
# We'll initially create a DataFrame comprising rows with unavailable ratings. This dataset will be utilized in the final stage to predict the ratings for these anime entries.
# %%
predicting_df_anime = df_anime[df_anime.isnull().any(axis=1)]
predicting_df_anime.drop(["Rating","Name","Rank","Tags","index"],axis=1,inplace=True)
predicting_df_anime.shape

#%%
predicting_df_anime.head()
# %% [markdown]
# Next we split our data for training and testing.

# %%
df_anime.dropna(subset=['Rating'], inplace=True)

X = df_anime.drop(["Rating","Name","Rank","Tags","index"],axis=1)
y=df_anime["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

# %%
X_train

# %% [markdown]
# Building our model and comparing the results

# %%
df_anime.columns[:20]

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
predictions = pd.DataFrame(model.predict(predicting_df_anime))
predictions.columns=["Rating"]
predictions
# %% [markdown]
# We could successfully predict rating of the anime as seen in the above table
# %%
