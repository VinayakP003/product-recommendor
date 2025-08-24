import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# ------------------------
# 1. Load Data
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("books.csv")  # 'book_id','title','authors','average_rating'
    ratings = pd.read_csv("ratings.csv")  # 'user_id','book_id','rating'
    return df, ratings

df, ratings = load_data()

# ------------------------
# 2. Content-Based Filtering
# ------------------------
tfidf = TfidfVectorizer(stop_words='english')
df["title"] = df["title"].fillna("")
tfidf_matrix = tfidf.fit_transform(df["title"] + " " + df["authors"])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend_content(title, top_n=5):
    if title not in indices:
        return ["Title not found in dataset"]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[book_indices].tolist()

# ------------------------
# 3. Collaborative Filtering
# ------------------------
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id','book_id','rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

def recommend_collaborative(user_id, top_n=5):
    book_ids = df['book_id'].unique()
    predictions = [model.predict(user_id, bid) for bid in book_ids]
    predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    top_books = [df[df['book_id'] == pred.iid]['title'].values[0] for pred in predictions[:top_n]]
    return top_books

# ------------------------
# 4. Hybrid Recommendation
# ------------------------
def hybrid_recommend(user_id, title, top_n=5):
    content_recs = recommend_content(title, top_n)
    collab_recs = recommend_collaborative(user_id, top_n)
    return content_recs, collab_recs

# ------------------------
# 5. Streamlit App UI
# ------------------------
st.title("📚 E-commerce Product Recommendation System")

st.sidebar.header("User Input")
user_id = st.sidebar.number_input("Enter User ID", min_value=1, max_value=int(ratings['user_id'].max()), value=1)
book_title = st.sidebar.text_input("Enter a Book Title", "The Hobbit")

if st.sidebar.button("Recommend"):
    st.subheader(f"Recommendations for User {user_id} and Book '{book_title}':")
    content_recs, collab_recs = hybrid_recommend(user_id, book_title, top_n=5)

    st.write("### Content-Based Recommendations:")
    for rec in content_recs:
        st.write(f"- {rec}")

    st.write("### Collaborative Recommendations:")
    for rec in collab_recs:
        st.write(f"- {rec}")
