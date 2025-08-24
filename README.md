# 📚 E-commerce Product Recommendation System

A hybrid recommendation system that suggests products (books in this case) using:
- **Content-Based Filtering** (TF-IDF + cosine similarity)
- **Collaborative Filtering** (SVD matrix factorization)

Built with Python, Scikit-learn, Surprise, and Streamlit.

---

## 🚀 Features
- Enter a **User ID** and a **Book Title**
- Get **Content-based recommendations** (similar books)
- Get **Collaborative recommendations** (based on similar users)
- Hybrid recommendation interface

---

## 🛠️ Tech Stack
- Python 3.9+
- Pandas, NumPy
- Scikit-learn
- Surprise (Collaborative Filtering)
- Streamlit (UI & Deployment)

---

## 📂 Project Structure
```
├── app.py              # Main Streamlit app
├── books.csv           # Dataset (books metadata)
├── ratings.csv         # Dataset (user ratings)
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```
