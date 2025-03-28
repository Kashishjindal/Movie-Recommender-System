#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import sklearn
import ast
import nltk


# In[10]:


print("numpy",np.__version__)
print("pandas",pd.__version__)
print("sklearn",sklearn.__version__)
print("nltk",nltk.__version__)


# In[4]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[5]:


movies.head(4)


# In[6]:


credits.tail(5)


# In[7]:


credits['cast'].head(1).values


# In[8]:


credits['crew'].head(1).values


# In[9]:


movies_df = movies.merge(credits ,on='title')


# In[10]:


movies.shape


# In[11]:


credits.shape


# In[12]:


movies_df.shape


# In[13]:


movies_df.head(1)


# In[14]:


#drop columns which are not needed
#genres
#id
#keywords
#title
#overview
#cast
#crew

movies_df = movies_df[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[15]:


movies['original_language'].value_counts()#most of movies are in english language


# In[16]:


movies_df.info()


# In[17]:


movies_df.head()


# In[18]:


movies_df.isnull().sum()


# In[19]:


movies_df.dropna(inplace=True)


# In[20]:


movies_df.isnull().sum()


# In[21]:


movies.duplicated().sum()


# In[22]:


movies.iloc[0].genres


# In[23]:


#change format of genres to['Action','Adventure,'Fantasy','Science Fiction']
def convert(obj):
    L = []
    for i in obj:
        L.append(i['name'])
    return L        


# In[24]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[25]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[26]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    


# In[27]:


movies_df['genres'] = movies_df['genres'].apply(convert)


# In[28]:


movies_df.head()


# In[43]:


movies_df['keywords'] = movies_df['keywords'].apply(convert)


# In[29]:


movies_df['cast'][0]#need first 3 dictionary only 


# In[30]:


def convert2(obj):
    L = []
    counter = 0 
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter +=1
        else:
            break
    return L  


# In[31]:


movies_df['cast'] = movies_df['cast'].apply(convert2)


# In[32]:


movies_df.head(2)


# In[33]:


#crew
movies_df['crew'][0] #neede only those dict having job value as director


# In[38]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[40]:


movies_df['crew'] = movies_df['crew'].apply(fetch_director)


# In[44]:


movies_df.sample(5)


# In[45]:


movies['overview'][0]#string ->convert to list


# In[47]:


movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split())


# In[49]:


movies_df.head()


# In[ ]:


#concat overview,genres,keywords,cast,crew ->into one column tag
#one more problem 'Sam Worthington' -> 'SamWorthington'


# In[52]:


movies_df['genres'] = movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['crew'] = movies_df['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[53]:


movies_df.sample(3)


# In[54]:


movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['crew']


# In[55]:


movies_df.head()


# In[88]:


new_df = movies_df[['movie_id','title','tags']]


# In[89]:


new_df


# In[90]:


#conver tags datatype to string
new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))


# In[91]:


new_df['tags'][0]#overview,genre,cast(top 3 actors),crew(director)


# In[92]:


new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# In[93]:


new_df.head()


# In[94]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[95]:


def stem(text):#convert words to root words
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[96]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d jamescameron')


# In[97]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[98]:


new_df['tags'][1300]


# In[ ]:


#now comes how can we find the similarirt between two movies .


# In[63]:


new_df['tags'][0]


# In[64]:


new_df['tags'][1]


# In[ ]:


#how can we find the similarirt score between these twwo paragraphs


# # VECTORIZATION
# - we will apply vectorization so that each movie behave as vector and for recommendation we will recommnd closest movies to that vector.
# - Text ->vectors(technique use ->Bag of words)
# - bag of worgs is a technique that combibe all the words and then acc to frequency found similarity between two vectors

# In[99]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words='english')#max no.of words =5000,stopwords not included


# In[100]:


vectors = cv.fit_transform(new_df['tags']).toarray()#by default cv retrun sparse matrix will conver it to array.


# In[101]:


vectors.shape


# In[102]:


vectors


# In[103]:


vectors[0]


# In[104]:


#we can check what are most frequent words 
cv.get_feature_names_out()


# In[ ]:


#apply stemming 
['loved','loving','love']
['love','love','love']


# In[79]:


#import nltk
get_ipython().system('pip install nltk')


# In[80]:





# In[105]:


#now we find distance between two movies 
#instead of eucledian distance we will find cosine(angular)distance
#in higher dimensions eucledian distance is not a good measure to use


# In[106]:


from sklearn.metrics.pairwise import cosine_similarity


# In[109]:


similarity = cosine_similarity(vectors)


# In[112]:


similarity[0].shape


# In[113]:


similarity[0]#1st movie similarity with itself =1


# In[120]:


sorted(similarity[0],reverse=True)#problem is that we losses index of the movie 


# In[127]:


sorted(list(enumerate(similarity[0])),reverse = True,key = lambda x:x[1])[1:6]


# In[137]:


#make recommendation function
def recommend(movie): 
    movie_index = new_df[new_df['title'] == movie].index[0]#fetch index of that movie
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True,key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        
    


# In[117]:


new_df[new_df['title'] == 'Avatar'].index[0]


# In[118]:


new_df[new_df['title'] == 'Batman Begins'].index[0]


# In[138]:


recommend('Avatar')


# In[133]:


new_df.iloc[539].title


# In[ ]:





# In[139]:


import pickle
pickle.dump(new_df,open('movies.pkl','wb'))


# In[140]:


pickle.dump(similarity,open('similarity.pkl',"wb"))

