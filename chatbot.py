import os
import json
import datetime
import nltk
import ssl
import streamlit as st
import random
import csv
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from streamlit_option_menu import option_menu

# --- Load environment variables ---
load_dotenv()
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')

# --- NLTK setup ---
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# --- Load intents from the JSON file ---
file_path = os.path.abspath("./intents.json")
with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# --- Preprocess data ---
patterns = []
tags = []
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# --- Train vectorizer and classifier ---
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
X = vectorizer.fit_transform(patterns)
y = tags
clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X, y)

# --- Chatbot function ---
def chatbot(user_input):
    input_vec = vectorizer.transform([user_input])
    predicted_tag = clf.predict(input_vec)[0]
    for intent in intents:
        if intent['tag'] == predicted_tag:
            responses = intent.get('responses', [])
            if responses:
                return random.choice(responses)
            else:
                return "Sorry, I don't have a response for that."
    return "Sorry, I don't understand."

# --- Sentiment analysis function ---
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return '😊', 'Positive'
    elif polarity < -0.2:
        return '😞', 'Negative'
    else:
        return '😐', 'Neutral'

# --- Persistent chat history ---
CHAT_LOG = 'chat_log.csv'
def load_chat_history():
    messages = []
    if os.path.exists(CHAT_LOG):
        with open(CHAT_LOG, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader, None)
            for row in csv_reader:
                if len(row) >= 4:
                    messages.append({
                        "role": row[0],
                        "content": row[1],
                        "timestamp": row[2],
                        "sentiment": row[3]
                    })
    return messages

def save_message(role, content, timestamp, sentiment):
    with open(CHAT_LOG, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([role, content, timestamp, sentiment])

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="NLP Chatbot", page_icon="🤖", layout="wide")
    # Theme CSS
    theme = st.session_state.get('theme', 'Dark')
    if theme == "Light":
        st.markdown("""
            <style>
            body {background: #fff !important; color: #222 !important;}
            .stChatMessage {background: #f5f6fa; color: #222;}
            .user-bubble {background: #e0e0e0; color: #222;}
            .bot-bubble {background: #f3e6ff; color: #222;}
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            body {background: #000 !important; color: #fff !important;}
            .stChatMessage {background: #22223b; color: #fff;}
            .user-bubble {background: linear-gradient(90deg, #00fff7 60%, #000 100%); color: #fff;}
            .bot-bubble {background: linear-gradient(90deg, #ff00cc 60%, #000 100%); color: #fff;}
            </style>
        """, unsafe_allow_html=True)

    # --- Sidebar: Profile, Theme, Analytics ---
    with st.sidebar:
        st.title("👤 User Profile")
        avatar_file = st.file_uploader("Upload Avatar", type=["png", "jpg", "jpeg"])
        if avatar_file:
            st.session_state['avatar'] = avatar_file.getvalue()
        if st.session_state.get('avatar'):
            st.image(st.session_state['avatar'], width=100)
        user_name = st.text_input("Your Name", value=st.session_state.get('user_name', 'User'))
        st.session_state['user_name'] = user_name

        st.markdown("---")
        theme = st.radio("Theme", ["Dark", "Light"], index=0 if st.session_state.get('theme', 'Dark') == 'Dark' else 1)
        st.session_state['theme'] = theme

        st.markdown("---")
        st.markdown("<div class='sidebar-stats'>", unsafe_allow_html=True)
        st.markdown(f"**Total Messages:** {len(st.session_state.get('messages', []))}")
        if 'start_time' not in st.session_state:
            st.session_state['start_time'] = datetime.datetime.now()
        duration = (datetime.datetime.now() - st.session_state['start_time']).seconds
        st.markdown(f"**Session Duration:** {duration} sec")
        st.markdown("</div>", unsafe_allow_html=True)

        # Sentiment chart
        st.markdown("**Sentiment Analysis**")
        sentiments = [m['sentiment'] for m in st.session_state.get('messages', []) if m['role'] == 'user']
        pos = sentiments.count('Positive')
        neu = sentiments.count('Neutral')
        neg = sentiments.count('Negative')
        st.bar_chart({"Positive": [pos], "Neutral": [neu], "Negative": [neg]})

        if st.button("🧹 Clear Chat History"):
            st.session_state['messages'] = []
            st.session_state['start_time'] = datetime.datetime.now()
            if os.path.exists(CHAT_LOG):
                os.remove(CHAT_LOG)
            st.success("Chat history cleared!")

        # Download conversation as CSV
        if st.session_state.get('messages', []):
            csv_data = 'role,content,timestamp,sentiment\n' + '\n'.join([
                f"{m['role']},{m['content']},{m['timestamp']},{m['sentiment']}"
                for m in st.session_state['messages']
            ])
            st.download_button(
                label="⬇️ Download Chat as CSV",
                data=csv_data,
                file_name="chat_history.csv",
                mime="text/csv"
            )

    # --- Option Menu ---
    selected = option_menu(
        menu_title=None,
        options=["Home", "Conversation History", "About", "Feedback"],
        icons=["house", "clock-history", "info-circle", "chat-dots"],
        orientation="horizontal"
    )

    # --- Load persistent chat history ---
    if 'messages' not in st.session_state:
        st.session_state['messages'] = load_chat_history()

    if selected == "Home":
        st.title("🤖 Neon Chatbot UI & Feedback")
        # Show avatar and name at top of main area
        col1, col2 = st.columns([1, 6])
        with col1:
            if st.session_state.get('avatar'):
                st.image(st.session_state['avatar'], width=70)
            else:
                st.markdown("<div style='font-size:3em;'>🧑</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h3 style='margin-top: 20px;'>{st.session_state['user_name']}</h3>", unsafe_allow_html=True)
        st.write(f"Welcome, {st.session_state['user_name']}! Type a message below and press Enter to chat.")

        # Chat input and display
        if prompt := st.chat_input("Type your message..."):
            emoji, sentiment = get_sentiment(prompt)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state['messages'].append({
                "role": "user",
                "content": prompt,
                "timestamp": timestamp,
                "sentiment": sentiment
            })
            save_message("user", prompt, timestamp, sentiment)
            response = chatbot(prompt)
            bot_emoji, _ = get_sentiment(response)
            bot_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state['messages'].append({
                "role": "assistant",
                "content": response,
                "timestamp": bot_time,
                "sentiment": "Neutral"
            })
            save_message("assistant", response, bot_time, "Neutral")

        # Display chat history as bubbles with avatars and sentiment
        for i, message in enumerate(st.session_state['messages']):
            avatar = "🧑" if message["role"] == "user" else "🤖"
            bubble_class = "user-bubble" if message["role"] == "user" else "bot-bubble"
            sentiment_emoji, _ = get_sentiment(message["content"]) if message["role"] == "user" else ("🤖", "Neutral")
            st.markdown(f"<div class='stChatMessage {bubble_class}'><span class='chat-avatar'>{avatar}</span> <b>{message['role'].capitalize()}:</b> {message['content']} <span style='font-size:1.2em;'>{sentiment_emoji}</span><br><span style='font-size:0.8em;color:#00fff7'>{message['timestamp']}</span></div>", unsafe_allow_html=True)

        # Goodbye message
        if st.session_state['messages']:
            last_bot = st.session_state['messages'][-1]
            if last_bot["role"] == "assistant" and last_bot["content"].lower() in ['goodbye', 'bye']:
                st.info("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif selected == "Conversation History":
        st.header("Conversation History")
        if os.path.exists(CHAT_LOG):
            with open(CHAT_LOG, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader, None)
                for row in csv_reader:
                    if len(row) >= 4:
                        st.markdown(f"<div class='stChatMessage user-bubble'><b>User:</b> {row[1]} <span style='font-size:1.2em;'>{get_sentiment(row[1])[0]}</span></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='stChatMessage bot-bubble'><b>Chatbot:</b> {row[1]}</div>", unsafe_allow_html=True)
                        st.caption(f"Timestamp: {row[2]}")
                        st.markdown("---")
        else:
            st.info("No conversation history found.")

    elif selected == "About":
        st.title("About the Chatbot")
        st.write("""
        This project demonstrates a chatbot built using NLP and Logistic Regression, with a modern Streamlit interface. It supports intent recognition, chat history, feedback collection, user profile, theme switching, and sentiment analysis.
        """)
        st.subheader("Project Overview:")
        st.write("""
        1. NLP techniques and Logistic Regression algorithm are used to train the chatbot on labeled intents and entities.
        2. Streamlit web framework provides a user-friendly chat interface.
        3. Sentiment analysis and user profile features enhance the experience.
        """)
        st.subheader("Dataset:")
        st.write("""
        The dataset is a collection of labelled intents and entities stored in a JSON file.
        """)
        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface uses chat bubbles, avatars, session state, and analytics for a smooth experience.")
        st.subheader("Conclusion:")
        st.write("This chatbot can be extended with more data, advanced NLP, or deep learning models.")

    elif selected == "Feedback":
        st.header("Feedback")
        feedback = st.text_area("Please provide your feedback here:")
        user_email = st.text_input("Your Email Address:")
        if st.button("Submit Feedback"):
            if feedback and user_email:
                st.success("Thank you for your feedback! We will review it shortly.")
            else:
                st.error("Please fill in all the fields before submitting.")

if __name__ == '__main__':
    # Create chat_log.csv with header if not exists
    if not os.path.exists(CHAT_LOG):
        with open(CHAT_LOG, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['role', 'content', 'timestamp', 'sentiment'])
    main()
