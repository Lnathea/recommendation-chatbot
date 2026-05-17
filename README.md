# 🎬🎵📚 RecBot — Bilingual Recommendation Chatbot

> NLP-powered chatbot yang bisa rekomendasiin film, musik, dan buku — dalam Bahasa Indonesia maupun English

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20ML-orange)
![Bilingual](https://img.shields.io/badge/Language-ID%20%2B%20EN-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lnathea/recommendation-chatbot/blob/main/chatbot_complete.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-url.streamlit.app)

---

## 📌 Tentang Project

**RecBot** adalah chatbot berbasis NLP yang mengenali *intent* dari kalimat user dan memberikan rekomendasi film, musik, atau buku yang relevan — dengan dukungan Bahasa Indonesia dan Inggris secara bersamaan.

Motivasi utama: *"Bisakah kita membangun chatbot rekomendasi yang benar-benar memahami maksud user — bukan sekadar mencocokkan kata kunci?"*

---

## ✨ Fitur Utama

| Fitur | Detail |
|---|---|
| 🇮🇩🇬🇧 **Bilingual** | Otomatis deteksi Bahasa Indonesia & Inggris |
| 🎯 **Intent Classification** | 9 intent dikenali: greet, recommend, filter, ask_more, dll |
| 🔍 **Entity Extraction** | Ekstrak genre & mood dari kalimat user |
| 🧠 **Context Aware** | Ingat genre & mood dari percakapan sebelumnya |
| 🎬🎵📚 **Multi-domain** | Film, musik, dan buku dalam satu chatbot |
| 💬 **Natural Response** | Template respons bervariasi agar tidak monoton |

---

## 🧠 Arsitektur NLP

```
User Input
    ↓
Language Detection (langdetect)
    ↓
Text Preprocessing
  → Lowercase → Remove punctuation
  → Tokenize (NLTK)
  → Stopword Removal (ID: NLTK | EN: NLTK)
  → Stemming (ID: PySastrawi | EN: PorterStemmer)
    ↓
TF-IDF Vectorization (ngram 1-2, 500 fitur)
    ↓
Intent Classifier (ML terbaik dari 3 model)
    ↓
Entity Extractor (genre + mood + content type)
    ↓
Recommendation Engine (filter database)
    ↓
Bilingual Response Generator
```

---

## 🎯 Intent yang Dikenali

| Intent | Contoh (ID) | Contoh (EN) |
|---|---|---|
| `greet` | "Halo!" | "Hey there!" |
| `recommend_movie` | "Rekomendasiin film dong" | "Suggest me a movie" |
| `recommend_music` | "Kasih playlist" | "Recommend some music" |
| `recommend_book` | "Buku apa yang bagus?" | "Good books to read?" |
| `filter_genre` | "Yang horror" | "Something sci-fi" |
| `filter_mood` | "Lagi sedih nih" | "I'm feeling happy" |
| `ask_more` | "Ada lagi?" | "Give me more" |
| `thanks` | "Makasih!" | "Thank you!" |
| `goodbye` | "Sampai jumpa" | "Goodbye!" |

---

## 🗂️ Struktur Project

```
recommendation-chatbot/
│
├── 📓 chatbot_complete.ipynb    ← Notebook utama (Part 1 + 2 + 3)
├── 🌐 app.py                    ← Streamlit web app
│
├── 📦 intent_pipeline.pkl       ← Model classifier (TF-IDF + ML)
├── 📄 label_mapping.json        ← Mapping index → intent label
├── 📄 chatbot_config.json       ← Response templates + entity maps
│
├── 📄 db_movies.json            ← Database 23 film
├── 📄 db_music.json             ← Database 22 lagu
├── 📄 db_books.json             ← Database 18 buku
│
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🔬 Alur Notebook

### Part 1 — Data & Intent Design
- Merancang 9 intent dengan 140+ kalimat training (bilingual)
- Preprocessing teks: tokenisasi, stopword removal, stemming
- Membangun database rekomendasi film, musik, dan buku dengan genre & mood tag

### Part 2 — NLP Model Training
| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 56.00% | 0.5480 |
| **SVM (LinearSVC)** ⭐ | 60.00% | 0.5916 |
| Random Forest | 52.00% | 0.5272 |

- TF-IDF Vectorizer (unigram + bigram, 500 fitur)
- 5-Fold Cross Validation untuk validasi stabilitas model
- Pipeline TF-IDF + Classifier disimpan sebagai `.pkl`

### Part 3 — Chatbot Engine
- Language Detector (langdetect)
- Entity Extractor (keyword-based genre & mood detection)
- Recommendation Engine (filter + random sampling)
- Response Generator (bilingual templates dengan variasi)
- RecBot Class — menggabungkan semua komponen
- Demo percakapan: Bahasa Indonesia, English, dan code-switching

---

## 💡 Contoh Percakapan

```
👤 User  : rekomendasiin film action dong
🤖 RecBot: Nih beberapa film yang mungkin kamu suka 🎬
           🎬 Top Gun: Maverick (2022) ⭐8.3 — Elite pilot faces new mission
           🎬 Mad Max: Fury Road (2015) ⭐8.1 — Post-apocalyptic chase thriller

👤 User  : I'm feeling sad, any music?
🤖 RecBot: Check these out, you might love them! 🎵
           🎵 The Night We Met — Lord Huron (2014)
           🎵 someone like you — Adele (2011)

👤 User  : buku motivasi yang bagus
🤖 RecBot: Buku-buku ini kayaknya bakal kamu suka 📚
           📚 Atomic Habits — James Clear ⭐4.4 — Build good habits, break bad ones
           📚 The Alchemist — Paulo Coelho ⭐4.2 — Follow your dreams journey
```

---

## 🛠️ Tech Stack

| Library | Kegunaan |
|---|---|
| `nltk` | Tokenisasi, stopword removal, stemming (EN) |
| `PySastrawi` | Stemming Bahasa Indonesia |
| `langdetect` | Deteksi bahasa otomatis |
| `scikit-learn` | TF-IDF + ML classifier |
| `streamlit` | Web app deployment |
| `pandas` | Manajemen database rekomendasi |

---

## 🚀 Cara Menjalankan

**Opsi 1 — Streamlit Web App (Live Demo)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://recommendation-chatbot-gfcfyhm74n4h7miiz4kvc6.streamlit.app/)

**Opsi 2 — Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lnathea/recommendation-chatbot/blob/main/chatbot_complete.ipynb)

**Opsi 3 — Lokal**
```bash
git clone https://github.com/Lnathea/recommendation-chatbot.git
cd recommendation-chatbot
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Muhammad Afriza Hidayat**
Mahasiswa Teknologi Informasi | Data & AI Enthusiast | Telkom University Jakarta

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/afriza)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Lnathea)
