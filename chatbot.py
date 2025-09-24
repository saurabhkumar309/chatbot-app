import os
import json
import datetime
import nltk
import ssl
import streamlit as st
import random
import csv
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

# --- Feedback email function ---
def send_feedback_email(user_email, feedback):
    sender_email = EMAIL_USER
    sender_password = EMAIL_PASS
    receiver_email = EMAIL_USER  # Send to yourself
    subject = "Chatbot Feedback"
    body = f"Feedback from: {user_email}\n\n{feedback}"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True, None
    except Exception as e:
        return False, str(e)

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="NLP Chatbot", page_icon="🤖", layout="centered")
    st.markdown("""
        <style>
        .stChatMessage {background: #f0f4fc; border-radius: 16px; margin-bottom: 10px; padding: 12px;}
        .user-bubble {background: linear-gradient(90deg, #d1e7dd 60%, #b6e2d3 100%);}
        .bot-bubble {background: linear-gradient(90deg, #f8d7da 60%, #f3b6c2 100%);}
        .chat-avatar {font-size: 1.7em; margin-right: 8px;}
        .sidebar-stats {background: #e3e3e3; border-radius: 10px; padding: 10px; margin-bottom: 10px;}
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Menu")
    menu = ["Home", "Conversation History", "About", "Feedback"]
    choice = st.sidebar.radio("Go to", menu)

    # Initialize chat history and stats in session state
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = datetime.datetime.now()

    # Sidebar stats and clear button
    st.sidebar.markdown("<div class='sidebar-stats'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Total Messages:** {len(st.session_state['messages'])}")
    duration = (datetime.datetime.now() - st.session_state['start_time']).seconds
    st.sidebar.markdown(f"**Session Duration:** {duration} sec")
    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state['messages'] = []
        st.session_state['start_time'] = datetime.datetime.now()
        st.success("Chat history cleared!")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # Download conversation as CSV
    if st.session_state['messages']:
        csv_data = 'role,content,timestamp\n' + '\n'.join([f"{m['role']},{m['content']},{m['timestamp']}" for m in st.session_state['messages']])
        st.sidebar.download_button(
            label="⬇️ Download Chat as CSV",
            data=csv_data,
            file_name="chat_history.csv",
            mime="text/csv"
        )

    if choice == "Home":
        st.title("🤖 Chatbot with Modern UI")
        st.write("Welcome! Type a message below and press Enter to chat.")

        # Chat input and display
        if prompt := st.chat_input("Type your message..."):
            st.session_state['messages'].append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            response = chatbot(prompt)
            st.session_state['messages'].append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            # Save to CSV
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([prompt, response, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        # Display chat history as bubbles with avatars
        for message in st.session_state['messages']:
            avatar = "🧑" if message["role"] == "user" else "🤖"
            bubble_class = "user-bubble" if message["role"] == "user" else "bot-bubble"
            st.markdown(f"<div class='stChatMessage {bubble_class}'><span class='chat-avatar'>{avatar}</span> <b>{message['role'].capitalize()}:</b> {message['content']}<br><span style='font-size:0.8em;color:gray'>{message['timestamp']}</span></div>", unsafe_allow_html=True)

        # Goodbye message
        if st.session_state['messages']:
            last_bot = st.session_state['messages'][-1]
            if last_bot["role"] == "assistant" and last_bot["content"].lower() in ['goodbye', 'bye']:
                st.info("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    st.markdown(f"<div class='stChatMessage user-bubble'><b>User:</b> {row[0]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='stChatMessage bot-bubble'><b>Chatbot:</b> {row[1]}</div>", unsafe_allow_html=True)
                    st.caption(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.info("No conversation history found.")

    elif choice == "About":
        st.title("About the Chatbot")
        st.write("""
        This project demonstrates a chatbot built using NLP and Logistic Regression, with a modern Streamlit interface. It supports intent recognition, chat history, and feedback collection.
        """)
        st.subheader("Project Overview:")
        st.write("""
        1. NLP techniques and Logistic Regression algorithm are used to train the chatbot on labeled intents and entities.
        2. Streamlit web framework provides a user-friendly chat interface.
        """)
        st.subheader("Dataset:")
        st.write("""
        The dataset is a collection of labelled intents and entities stored in a JSON file.
        """)
        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface uses chat bubbles, avatars, and session state for a smooth experience.")
        st.subheader("Conclusion:")
        st.write("This chatbot can be extended with more data, advanced NLP, or deep learning models.")

    elif choice == "Feedback":
        st.header("Feedback")
        feedback = st.text_area("Please provide your feedback here:")
        user_email = st.text_input("Your Email Address:")
        if st.button("Submit Feedback"):
            if feedback and user_email:
                with st.spinner("Sending feedback..."):
                    success, error_msg = send_feedback_email(user_email, feedback)
                if success:
                    st.success("Thank you for your feedback! We will review it shortly.")
                else:
                    st.error(f"There was an error sending your feedback. Please try again.\nError details: {error_msg}")
            else:
                st.error("Please fill in all the fields before submitting.")

if __name__ == '__main__':
    # Create chat_log.csv with header if not exists
    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
    main()
