# app.py — Streamlit Product Recommender (Surprise if available, sklearn fallback)
import streamlit as st
import pandas as pd
import numpy as np

# Try Surprise first; fallback to sklearn if not available (e.g., Streamlit Cloud)
USE_SURPRISE = True
try:
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split as sp_train_test_split
except Exception:
    USE_SURPRISE = False

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

st.set_page_config(page_title="📚 E-commerce Product Recommendation System", page_icon="🛒", layout="wide")

# ------------------------
# 1) Load Data
# ------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("books.csv")       # columns: book_id, title, authors
    ratings = pd.read_csv("ratings.csv")  # columns: user_id, book_id, rating

    # Clean types & align IDs
    df["book_id"] = pd.to_numeric(df["book_id"], errors="coerce")
    df = df.dropna(subset=["book_id", "title"]).copy()
    df["book_id"] = df["book_id"].astype(int)

    for c in ["user_id", "book_id", "rating"]:
        ratings[c] = pd.to_numeric(ratings[c], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "book_id", "rating"]).copy()
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["book_id"] = ratings["book_id"].astype(int)
    ratings["rating"]  = ratings["rating"].astype(float)

    # Keep only items present in df to avoid lookup errors
    valid = set(df["book_id"])
    ratings = ratings[ratings["book_id"].isin(valid)]

    # Popularity fallback
    pop_df = (
        ratings.groupby("book_id")
        .agg(cnt=("rating", "count"), mean=("rating", "mean"))
        .sort_values(["cnt", "mean"], ascending=[False, False])
        .reset_index()
    )
    return df.reset_index(drop=True), ratings.reset_index(drop=True), pop_df

df, ratings, pop_df = load_data()
books_idx = df.set_index("book_id")

if df.empty or ratings.empty:
    st.error("books.csv or ratings.csv is empty/invalid. Ensure required columns exist.")
    st.stop()

# ------------------------
# 2) Content-Based Filtering
# ------------------------
@st.cache_resource(show_spinner=False)
def build_content_index(df_books: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words='english', max_features=100_000)
    corpus = (df_books["title"].fillna("") + " " + df_books["authors"].fillna("")).values
    tfidf_matrix = tfidf.fit_transform(corpus)  # sparse
    title_to_idx = pd.Series(df_books.index, index=df_books['title'].astype(str)).drop_duplicates()
    return tfidf, tfidf_matrix, title_to_idx

tfidf, tfidf_matrix, indices = build_content_index(df)

def recommend_content(title, top_n=5):
    title = str(title)
    if title not in indices:
        return []
    idx = int(indices[title])
    # Use cosine_similarity; request dense output for a flat array
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix, dense_output=True).ravel()
    sims[idx] = -1.0  # exclude itself
    k = min(top_n, len(sims)-1)
    cand = np.argpartition(-sims, k)[:k]
    cand = cand[np.argsort(-sims[cand])]
    out = df.iloc[cand][["book_id", "title", "authors"]].copy()
    return out["title"].tolist()

# ------------------------
# 3) Collaborative Filtering (Surprise or sklearn fallback)
# ------------------------
@st.cache_resource(show_spinner=True)
def train_collab(ratings_df: pd.DataFrame,
                 n_factors: int = 100, n_epochs: int = 20, lr_all: float = 0.005, reg_all: float = 0.02):
    if USE_SURPRISE:
        reader = Reader(rating_scale=(float(ratings_df["rating"].min()), float(ratings_df["rating"].max())))
        data = Dataset.load_from_df(ratings_df[["user_id", "book_id", "rating"]], reader)
        trainset, _ = sp_train_test_split(data, test_size=0.2, random_state=42)
        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, random_state=42)
        algo.fit(trainset)
        user_known = ratings_df.groupby("user_id")["book_id"].apply(set)
        all_items = df["book_id"].tolist()
        return {"type": "surprise", "algo": algo, "user_known": user_known, "all_items": all_items}

    # sklearn fallback (no compilation issues)
    u_codes, u_vals = pd.factorize(ratings_df["user_id"], sort=True)
    i_codes, i_vals = pd.factorize(ratings_df["book_id"], sort=True)
    R = csr_matrix((ratings_df["rating"].values, (u_codes, i_codes)),
                   shape=(len(u_vals), len(i_vals)))
    k = min(100, max(1, min(R.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=42)
    U = svd.fit_transform(R)
    V = svd.components_.T
    user_to_idx = {int(u): i for i, u in enumerate(u_vals)}
    idx_to_item = {i: int(it) for it, i in {int(v): i for i, v in enumerate(i_vals)}.items()}
    user_known = ratings_df.groupby("user_id")["book_id"].apply(set)
    return {"type": "sklearn", "U": U, "V": V, "user_to_idx": user_to_idx, "idx_to_item": idx_to_item, "user_known": user_known}

model = train_collab(ratings)

def recommend_collaborative(user_id: int, top_n=5):
    known = model["user_known"].get(user_id, set())

    # Popularity fallback
    def popularity_fallback():
        top_ids = [bid for bid in pop_df["book_id"].tolist() if bid not in known][:top_n]
        out = books_idx.reindex(top_ids)[["title","authors"]].reset_index().dropna(subset=["book_id"])
        return out["title"].tolist()

    if model["type"] == "surprise":
        algo = model["algo"]
        preds = []
        for iid in model["all_items"]:
            if iid in known:
                continue
            est = algo.predict(uid=user_id, iid=iid, verbose=False).est
            preds.append((iid, est))
        if not preds:
            return popularity_fallback()
        preds.sort(key=lambda x: -x[1])
        top_ids = [iid for iid,_ in preds[:top_n]]
        out = books_idx.reindex(top_ids)[["title","authors"]].reset_index().dropna(subset=["book_id"])
        return out["title"].tolist()

    # sklearn path
    if user_id not in model["user_to_idx"]:
        return popularity_fallback()
    uidx = model["user_to_idx"][user_id]
    scores = model["V"].dot(model["U"][uidx])
    cand = []
    for i, s in enumerate(scores):
        iid = model["idx_to_item"].get(i)
        if iid is not None and iid not in known:
            cand.append((iid, float(s)))
    if not cand:
        return popularity_fallback()
    cand.sort(key=lambda x: -x[1])
    top_ids = [iid for iid,_ in cand[:top_n]]
    out = books_idx.reindex(top_ids)[["title","authors"]].reset_index().dropna(subset=["book_id"])
    return out["title"].tolist()

# ------------------------
# 4) Hybrid Recommendation
# ------------------------
def hybrid_recommend(user_id, title, top_n=5):
    return recommend_content(title, top_n), recommend_collaborative(user_id, top_n)

# ------------------------
# 5) Streamlit App UI
# ------------------------
st.title("📚 E-commerce Product Recommendation System")
if not USE_SURPRISE:
    st.info("Running without scikit-surprise (not available). Using a fast sklearn SVD fallback.")

st.sidebar.header("User Input")
user_id = st.sidebar.number_input("Enter User ID", min_value=int(ratings['user_id'].min()),
                                  max_value=int(ratings['user_id'].max()), value=int(ratings['user_id'].min()))
book_title = st.sidebar.text_input("Enter a Book Title", str(df['title'].iloc[0]))

if st.sidebar.button("Recommend"):
    st.subheader(f"Recommendations for User {user_id} and Book '{book_title}':")
    content_recs, collab_recs = hybrid_recommend(int(user_id), book_title, top_n=5)

    st.write("### Content-Based Recommendations:")
    if content_recs:
        for rec in content_recs:
            st.write(f"- {rec}")
    else:
        st.write("- No similar titles found.")

    st.write("### Collaborative Recommendations:")
    if collab_recs:
        for rec in collab_recs:
            st.write(f"- {rec}")
    else:
        st.write("- No recommendations (user may have rated all items); falling back to popularity might help.")
