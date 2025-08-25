# 📚 Product Recommendation System (Goodbooks-10k) — Streamlit App

A **hybrid book recommender** built on the [Goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k) dataset.  
It combines **content-based filtering** (TF-IDF on title + authors) and **collaborative filtering** (matrix factorization with SVD).  
Designed to **run reliably on Streamlit Cloud** (no tricky native builds) and locally.

> 🔗 **Live demo :** https://share.streamlit.io/user/vinayakp003

---

## ✨ Features
- **Content-based**: TF-IDF + cosine similarity over `title` and `authors`.
- **Collaborative**: Matrix factorization (SVD).  
  - Cloud-safe: uses **scikit-learn’s TruncatedSVD** by default.  
  - Optional local **Surprise SVD** if you install `scikit-surprise`.
- **Hybrid view**: Content and collaborative recommendations side-by-side.
- **Cold-start fallback**: Popularity-based recommendations when needed.
- **Robust**: Handles missing files/columns and ID mismatches gracefully.

---

## 📁 Project Structure
product-recommender/
├─ app.py # Streamlit app (sklearn fallback; Surprise optional locally)
├─ books.csv # Goodbooks-10k metadata (book_id, title, authors, ...)
├─ ratings.csv # Goodbooks-10k ratings (user_id, book_id, rating) — consider a sample
├─ requirements.txt # Cloud-safe dependencies
└─ README.md # This file
