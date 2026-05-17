import streamlit as st
import json
import pickle
import random
import re
import os
import nltk
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RecBot — Bilingual Recommendation Chatbot",
    page_icon="🎬",
    layout="wide"
)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from langdetect import detect, LangDetectException

# ── Load Aset ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    with open('intent_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('label_mapping.json', 'r') as f:
        mapping   = json.load(f)
        idx2label = {int(k): v for k, v in mapping['idx2label'].items()}
    with open('chatbot_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open('db_movies.json', 'r') as f:
        movies = json.load(f)
    with open('db_music.json', 'r') as f:
        music  = json.load(f)
    with open('db_books.json', 'r') as f:
        books  = json.load(f)
    return pipeline, idx2label, config, movies, music, books

pipeline, idx2label, config, movies, music, books = load_assets()
GENRE_MAP        = config['genre_map']
MOOD_MAP         = config['mood_map']
CONTENT_TYPE_MAP = config['content_type_map']
RESPONSES        = config['responses']

# ── Preprocessing ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_stemmers():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer_en = PorterStemmer()
    stemmer_id = StemmerFactory().create_stemmer()
    stop_en    = set(stopwords.words('english'))
    stop_id    = set(stopwords.words('indonesian'))
    stop_id.update(['dong', 'nih', 'ya', 'deh', 'sih', 'yuk', 'lah', 'banget', 'aja'])
    stop_en.update(['please', 'me', 'give', 'i', 'want', 'need'])
    return stemmer_en, stemmer_id, stop_en, stop_id

stemmer_en, stemmer_id, stop_en, stop_id = load_stemmers()

# ── Helper Functions ──────────────────────────────────────────────────────────
def detect_language(text):
    try:
        lang = detect(text)
        return 'id' if lang == 'id' else 'en'
    except LangDetectException:
        return 'en'

def extract_entities(text):
    text_lower = text.lower()
    words      = re.findall(r'\w+', text_lower)
    genre = mood = ctype = None
    for word in words:
        if word in GENRE_MAP and not genre:
            genre = GENRE_MAP[word]
        if word in MOOD_MAP and not mood:
            mood = MOOD_MAP[word]
        if word in CONTENT_TYPE_MAP and not ctype:
            ctype = CONTENT_TYPE_MAP[word]
    return {'genre': genre, 'mood': mood, 'content_type': ctype}

def get_recommendations(content_type, genre=None, mood=None, n=3):
    db       = {'movie': movies, 'music': music, 'book': books}.get(content_type, movies)
    filtered = db
    if genre:
        g = [i for i in filtered if i.get('genre') == genre]
        if g: filtered = g
    if mood:
        m = [i for i in filtered if mood in i.get('mood', [])]
        if m: filtered = m
    if not filtered:
        filtered = db
    return random.sample(filtered, min(n, len(filtered)))

def get_response_text(key, lang):
    resp = RESPONSES.get(key, RESPONSES['fallback'])
    text = resp.get(lang, resp.get('en', ''))
    if isinstance(text, list):
        return random.choice(text)
    return text

def format_rec_cards(items, content_type):
    """Format rekomendasi sebagai card HTML."""
    cards = []
    for item in items:
        if content_type == 'movie':
            cards.append(f"""
            <div style='background:#1a1a2e;border-radius:10px;padding:12px;
                        margin:6px 0;border-left:3px solid #0052D9'>
                <b>🎬 {item['title']}</b> ({item.get('year','')}) ⭐ {item.get('rating','')}
                <br><small style='color:#aaa'>{item.get('desc','')}</small>
                <br><small style='color:#4FC3F7'>Genre: {item.get('genre','')}</small>
            </div>""")
        elif content_type == 'music':
            cards.append(f"""
            <div style='background:#1a2e1a;border-radius:10px;padding:12px;
                        margin:6px 0;border-left:3px solid #1DB954'>
                <b>🎵 {item['title']}</b> — {item['artist']}
                <br><small style='color:#aaa'>{item.get('year','')}</small>
                <small style='color:#69db7c'> | Genre: {item.get('genre','')}</small>
            </div>""")
        else:
            cards.append(f"""
            <div style='background:#2e1a1a;border-radius:10px;padding:12px;
                        margin:6px 0;border-left:3px solid #FF6B35'>
                <b>📚 {item['title']}</b> — {item['author']} ⭐ {item.get('rating','')}
                <br><small style='color:#aaa'>{item.get('desc','')}</small>
                <br><small style='color:#ffa94d'>Genre: {item.get('genre','')}</small>
            </div>""")
    return ''.join(cards)

def chatbot_respond(user_input, context):
    """Proses input dan return (response_text, html_cards, updated_context)."""
    lang     = detect_language(user_input)
    pred_idx = pipeline.predict([user_input])[0]
    intent   = idx2label[pred_idx]
    entities = extract_entities(user_input)

    genre = entities['genre'] or context.get('last_genre')
    mood  = entities['mood']  or context.get('last_mood')
    ctype = entities['content_type']

    if entities['genre']: context['last_genre'] = entities['genre']
    if entities['mood']:  context['last_mood']  = entities['mood']
    if ctype:             context['last_content_type'] = ctype

    context['last_lang'] = lang
    html_cards = ''

    if intent == 'greet':
        text = get_response_text('greet', lang)

    elif intent in ['recommend_movie', 'recommend_music', 'recommend_book']:
        ctype_map = {'recommend_movie': 'movie',
                     'recommend_music': 'music',
                     'recommend_book' : 'book'}
        ctype = ctype_map[intent]
        context['last_content_type'] = ctype
        recs       = get_recommendations(ctype, entities['genre'], entities['mood'])
        text       = get_response_text(intent, lang)
        html_cards = format_rec_cards(recs, ctype)

    elif intent in ['filter_genre', 'filter_mood']:
        ctype = context.get('last_content_type', 'movie')
        recs  = get_recommendations(ctype, genre, mood)
        intent_key = f'recommend_{ctype}'
        text       = get_response_text(intent_key, lang)
        html_cards = format_rec_cards(recs, ctype)

    elif intent == 'ask_more':
        ctype = context.get('last_content_type', 'movie')
        recs  = get_recommendations(ctype,
                                    context.get('last_genre'),
                                    context.get('last_mood'))
        text       = get_response_text('ask_more', lang)
        html_cards = format_rec_cards(recs, ctype)

    elif intent == 'thanks':
        text = get_response_text('thanks', lang)

    elif intent == 'goodbye':
        text = get_response_text('goodbye', lang)

    else:
        text = get_response_text('fallback', lang)

    lang_flag = '🇮🇩' if lang == 'id' else '🇬🇧'
    debug_info = f"<small style='color:#666'>{lang_flag} {lang.upper()} | intent: <b>{intent}</b></small>"

    return text, html_cards, debug_info, context

# ── Init Session State ────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'context' not in st.session_state:
    st.session_state.context = {
        'last_content_type': None,
        'last_genre'       : None,
        'last_mood'        : None,
        'last_lang'        : 'id'
    }
if 'started' not in st.session_state:
    st.session_state.started = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎬🎵📚 RecBot")
st.sidebar.markdown("**Bilingual Recommendation Chatbot**")
st.sidebar.markdown("Bisa bahasa Indonesia maupun English!")
st.sidebar.divider()

st.sidebar.markdown("### 💡 Contoh Pertanyaan")
examples_id = [
    "rekomendasiin film action dong",
    "lagi sedih, musik apa yang cocok?",
    "buku motivasi yang bagus",
    "film horror yang menegangkan",
    "ada rekomendasi lagi?",
]
examples_en = [
    "suggest me a relaxing book",
    "what movie should I watch?",
    "I'm feeling happy, any music?",
    "something sci-fi please",
    "give me more recommendations",
]

st.sidebar.markdown("**🇮🇩 Bahasa Indonesia:**")
for ex in examples_id:
    st.sidebar.caption(f"• {ex}")

st.sidebar.markdown("**🇬🇧 English:**")
for ex in examples_en:
    st.sidebar.caption(f"• {ex}")

st.sidebar.divider()
if st.sidebar.button("🗑️ Reset Percakapan"):
    st.session_state.messages = []
    st.session_state.context  = {
        'last_content_type': None,
        'last_genre': None,
        'last_mood': None,
        'last_lang': 'id'
    }
    st.session_state.started = False
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("**Muhammad Afriza Hidayat**")
st.sidebar.markdown("Mahasiswa Teknik Informatika | Data & AI Enthusiast")
st.sidebar.markdown(
    "[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/afriza) "
    "[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/Lnathea)"
)

# ── Main Page ─────────────────────────────────────────────────────────────────
st.title("🤖 RecBot — Bilingual Recommendation Chatbot")
st.markdown("Chatbot pintar yang bisa rekomendasiin **film 🎬, musik 🎵, dan buku 📚** — dalam Bahasa Indonesia maupun English!")

# Greeting otomatis
if not st.session_state.started:
    greeting = "Halo! Aku RecBot 🤖 Aku bisa rekomendasiin film 🎬, musik 🎵, atau buku 📚 buat kamu. Mau coba? Ketik aja apa yang kamu inginkan — bisa bahasa Indonesia atau English!"
    st.session_state.messages.append({
        "role": "assistant",
        "text": greeting,
        "cards": "",
        "debug": ""
    })
    st.session_state.started = True

# Tampilkan chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["text"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["text"])
            if msg.get("debug"):
                st.markdown(msg["debug"], unsafe_allow_html=True)
            if msg.get("cards"):
                st.markdown(msg["cards"], unsafe_allow_html=True)

# Input user
user_input = st.chat_input("Ketik pesan kamu di sini... (ID/EN)")

if user_input:
    # Tampilkan pesan user
    st.session_state.messages.append({
        "role": "user",
        "text": user_input,
        "cards": "",
        "debug": ""
    })

    # Generate respons
    response_text, html_cards, debug_info, updated_context = chatbot_respond(
        user_input, st.session_state.context
    )
    st.session_state.context = updated_context

    # Simpan respons bot
    st.session_state.messages.append({
        "role"  : "assistant",
        "text"  : response_text,
        "cards" : html_cards,
        "debug" : debug_info
    })

    st.rerun()
