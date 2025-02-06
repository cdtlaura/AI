#!/usr/bin/env python
# coding: utf-8



# In[2]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[5]:


import certifi
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# In[6]:


url = 'https://raw.githubusercontent.com/fenago/datasets/refs/heads/main/sample-data.csv'
ds = pd.read_csv(url)


# In[8]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['description'])


# In[9]:


cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[10]:


results = {}

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

    results[row['id']] = similar_items[1:]


# In[11]:


def item(id):
    return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]


# In[12]:


def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")


# In[13]:


recommend(item_id=5, num=3)


# Recommending 3 products similar to Alpine guide pants...
# -------
# Recommended: Active sport boxer briefs (score:0.2748)
# Recommended: Active classic boxers (score:0.2639)
# Recommended: Active sport briefs (score:0.2103)

# In[14]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Load the dataset
url = 'https://raw.githubusercontent.com/fenago/datasets/refs/heads/main/sample-data.csv'
ds = pd.read_csv(url)

# TF-IDF Vectorization
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['description'])

# Calculate cosine similarities
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Store results in a dictionary
results = {}
for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]
    results[row['id']] = similar_items[1:]

# Helper function to get item description
def item(id):
    return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]

# Recommendation function
def recommend(item_id, num_recommendations):
    if item_id not in results:
        return "Product not found."
    
    recs = results[item_id][:num_recommendations]
    recommendations_text = f"Recommending {num_recommendations} products similar to '{item(item_id)}':\n\n"
    for rec in recs:
        recommendations_text += f"- {item(rec[1])} (score: {rec[0]:.4f})\n"
    return recommendations_text

# Streamlit interface
st.title("Product Recommendation System")

# Product ID input
item_id = st.number_input("Enter Product ID", min_value=int(ds['id'].min()), max_value=int(ds['id'].max()), value=int(ds['id'].min()))
num_recommendations = st.number_input("Number of Recommendations", min_value=1, max_value=10, value=5)

# Show recommendations when button is pressed
if st.button("Recommend"):
    recommendations = recommend(item_id, num_recommendations)
    st.text(recommendations)

