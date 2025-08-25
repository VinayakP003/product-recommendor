# app.py — Streamlit Product Recommender (sklearn SVD only, Cloud-safe)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="📚 E-commerce Product Recommendation System", page_icon="🛒", layout="wide")

# ------------------------ Data Loading & Validation ------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if not Path("books.csv").exists():
        raise FileNotFoundError("books.csv not found next to app.py")
    if not Path("ratings.csv").exists():
        raise FileNotFoundError("ratings.csv not found next to app.py")

    df = pd.read_csv("books.csv")        # expects: book_id, title, authors
    ratings = pd.read_csv("ratings.csv") # expects: user_id, book_id, rating

    # Clean types
    df["book_id"] = pd.to_numeric(df["book_id"], errors="coerce")
    df = df.dropna(subset=["book_id", "title"]).copy()
    df["book_id"] = df["book_id"].astype(int)
    df["authors"] = df["authors"].fillna("")

    for c in ["user_id","book_id","rating"]:
        ratings[c] = pd.to_numeric(ratings[c], errors="coerce")
    ratings = ratings.dropna(subset=["user_id","book_id","rating"]).copy()
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["book_id"] = ratings["book_id"].astype(int)
    ratings["rating"]  = ratings["rating"].astype(float)

    # Align ratings to valid items
    valid = set(df["book_id"])
    ratings = ratings[ratings["book_id"].isin(valid)]

    # Popularity fallback
    pop_df = (ratings.groupby("book_id")
              .agg(cnt=("rating","count"), mean=("rating","mean"))
              .sort_values(["cnt","mean"], ascending=[False, False])
              .reset_index())

    if df.empty or ratings.empty:
        raise ValueError("books.csv or ratings.csv ended up empty after cleaning/alignment.")

    return df.reset_index(drop=True), ratings.reset_index(drop=True), pop_df

df, ratings, pop_df = load_data()
books_idx = df.set_index("book_id")

# ------------------------ Content-Based (TF-IDF) ------------------------
@st.cache_resource(show_spinner=False)
def build_content_index(df_books: pd.DataFrame):
    text = (df_books["title"].fillna("") + " " + df_books["authors"].fillna("")).values
    tfidf = TfidfVectorizer(stop_words="english", max_features=100_000)
    X = tfidf.fit_transform(text)  # sparse CSR
    title_to_idx = pd.Series(df_books.index, index=df_books["title"].astype(str)).drop_duplicates()
    return tfidf, X, title_to_idx

tfidf, X_content, title_to_idx = build_content_index(df)

def recommend_content(title: str, top_n: int = 5):
    title = str(title)
    if title not in title_to_idx:
        return []
    idx = int(title_to_idx[title])
    sims = cosine_similarity(X_content[idx], X_content, dense_output=True).ravel()
    sims[idx] = -1.0
    k = min(top_n, len(sims)-1)
    cand = np.argpartition(-sims, k)[:k]
    cand = cand[np.argsort(-sims[cand])]
    out = df.iloc[cand][["book_id","title","authors"]]
    return out["title"].tolist()

# ------------------------ Collaborative (TruncatedSVD) ------------------------
@st.cache_resource(show_spinner=True)
def train_collab(ratings_df: pd.DataFrame):
    # Factorize ids
    u_codes, u_vals = pd.factorize(ratings_df["user_id"], sort=True)
    i_codes, i_vals = pd.factorize(ratings_df["book_id"], sort=True)

    # User–item sparse matrix
    R = csr_matrix((ratings_df["rating"].values, (u_codes, i_codes)),
                   shape=(len(u_vals), len(i_vals)))

    k = min(100, max(1, min(R.shape)-1))
    svd = TruncatedSVD(n_components=k, random_state=42)
    U = svd.fit_transform(R)  # (n_users, k)
    V = svd.components_.T     # (n_items, k)

    user_to_idx = {int(u): i for i, u in enumerate(u_vals)}
    idx_to_item = {i: int(it) for it, i in {int(v): i for i, v in enumerate(i_vals)}.items()}
    user_known = ratings_df.groupby("user_id")["book_id"].apply(set)

    return {"U": U, "V": V, "user_to_idx": user_to_idx, "idx_to_item": idx_to_item, "user_known": user_known}

model = train_collab(ratings)

def recommend_collaborative(user_id: int, top_n: int = 5):
    known = model["user_known"].get(user_id, set())

    # Popularity fallback
    def pop_fallback():
        top_ids = [bid for bid in pop_df["book_id"].tolist() if bid not in known][:top_n]
        out = books_idx.reindex(top_ids)[["title","authors"]].reset_index().dropna(subset=["book_id"])
        return out["title"].tolist()

    if user_id not in model["user_to_idx"]:
        return pop_fallback()

    uidx = model["user_to_idx"][user_id]
    scores = model["V"].dot(model["U"][uidx])
    cand = []
    for i, s in enumerate(scores):
        iid = model["idx_to_item"].get(i)
        if iid is not None and iid not in known:
            cand.append((iid, float(s)))

    if not cand:
        return pop_fallback()

    cand.sort(key=lambda x: -x[1])
    top_ids = [iid for iid,_ in cand[:top_n]]
    out = books_idx.reindex(top_ids)[["title","authors"]].reset_index().dropna(subset=["book_id"])
    return out["title"].tolist()

# ------------------------ UI ------------------------
st.title("📚 E-commerce Product Recommendation System")

st.sidebar.header("User Input")
u_min, u_max = int(ratings["user_id"].min()), int(ratings["user_id"].max())
user_id = st.sidebar.number_input("Enter User ID", min_value=u_min, max_value=u_max, value=u_min, step=1)
book_title = st.sidebar.text_input("Enter a Book Title", str(df["title"].iloc[0]))
top_n = st.sidebar.slider("How many recommendations?", 3, 20, 5)

if st.sidebar.button("Recommend", type="primary"):
    st.subheader(f"Recommendations for User {user_id} and Book '{book_title}':")
    # Content
    c_recs = recommend_content(book_title, top_n)
    st.write("### Content-Based Recommendations:")
    st.write("- No similar titles found." if not c_recs else "")
    for r in c_recs:
        st.write(f"- {r}")
    # Collaborative
    k_recs = recommend_collaborative(int(user_id), top_n)
    st.write("### Collaborative Recommendations:")
    st.write("- No collaborative recommendations (user may have rated all items)." if not k_recs else "")
    for r in k_recs:
        st.write(f"- {r}")
