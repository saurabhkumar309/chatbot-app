
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
from PIL import Image
import base64
import io
import plotly.express as px
import pandas as pd


# --- Load environment variables ---
load_dotenv()
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')


# --- NLTK setup ---
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
try:
    nltk.download('punkt', quiet=True)
except:
    pass


# --- Load intents from the JSON file ---
file_path = os.path.abspath("./intents.json")
try:
    with open(file_path, "r", encoding="utf-8") as file:
        intents = json.load(file)
except FileNotFoundError:
    # Default intents if file doesn't exist
    intents = [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
            "responses": ["Hi there!", "Hello!", "Hey! How can I help you?", "I'm doing great! How can I assist you today?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you later", "Take care"],
            "responses": ["Goodbye! Have a great day!", "See you later!", "Take care!", "Thanks for chatting!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thank you", "Thanks", "Thank you so much", "I appreciate it"],
            "responses": ["You're welcome!", "Happy to help!", "Anytime!", "Glad I could assist!"]
        }
    ]


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


# --- Enhanced chatbot function with image handling ---
def chatbot(user_input, image_uploaded=False):
    if image_uploaded:
        return "I can see you've shared an image! While I can't analyze the image content yet, I appreciate you sharing it with me. How can I help you today?"

    input_vec = vectorizer.transform([user_input])
    predicted_tag = clf.predict(input_vec)[0]
    confidence = max(clf.predict_proba(input_vec)[0])

    # If confidence is too low, provide a generic response
    if confidence < 0.3:
        return "I'm not quite sure about that. Could you please rephrase your question or ask something else?"

    for intent in intents:
        if intent['tag'] == predicted_tag:
            responses = intent.get('responses', [])
            if responses:
                return random.choice(responses)
            else:
                return "Sorry, I don't have a response for that."
    return "Sorry, I don't understand."


# --- Feedback email function (unchanged as requested) ---
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


# --- Image processing functions ---
def process_image(uploaded_file):
    """Process uploaded image and return base64 string for display"""
    image = Image.open(uploaded_file)
    # Resize image if too large
    max_size = (800, 600)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Convert to base64 for storage in session state
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str, image


# --- Analytics functions ---
def get_chat_analytics():
    """Generate analytics from chat history"""
    if not st.session_state['messages']:
        return None

    df = pd.DataFrame(st.session_state['messages'])

    # Message count by role
    role_counts = df['role'].value_counts()

    # Messages over time (by hour)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    hourly_counts = df.groupby('hour').size().reset_index(name='count')

    return {
        'role_counts': role_counts,
        'hourly_counts': hourly_counts,
        'total_messages': len(df),
        'session_duration': (datetime.datetime.now() - st.session_state['start_time']).seconds
    }


# --- Enhanced Streamlit App ---
def main():
    st.set_page_config(
        page_title="AI Chatbot Pro", 
        page_icon="🤖", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for modern UI
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .main-header {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            padding: 1rem;
            border-radius: 15px;
            text-align: center;
            color: white;
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        .chat-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 20px 20px 5px 20px;
            margin: 10px 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
        }
        .bot-message {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 20px 20px 20px 5px;
            margin: 10px 0;
            max-width: 80%;
            box-shadow: 0 4px 15px 0 rgba(240, 147, 251, 0.4);
        }
        .sidebar-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
        }
        .stat-card {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        .feature-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            transition: transform 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
        }
        .upload-area {
            border: 2px dashed #4facfe;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            background: rgba(255, 255, 255, 0.1);
            margin: 10px 0;
        }
        .image-preview {
            border-radius: 15px;
            margin: 10px 0;
            max-width: 100%;
            box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar with modern design
    with st.sidebar:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.title("🤖 AI Chatbot Pro")
        st.markdown("Your intelligent conversation partner")
        st.markdown('</div>', unsafe_allow_html=True)

        menu = ["🏠 Home", "💬 Chat History", "📊 Analytics", "ℹ️ About", "💌 Feedback", "⚙️ Settings"]
        choice = st.selectbox("Navigate to:", menu)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = datetime.datetime.now()
    if 'user_name' not in st.session_state:
        st.session_state['user_name'] = ""
    if 'theme' not in st.session_state:
        st.session_state['theme'] = "modern"

    # Sidebar stats
    with st.sidebar:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Session Stats")
        total_messages = len(st.session_state['messages'])
        duration = (datetime.datetime.now() - st.session_state['start_time']).seconds

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", total_messages)
        with col2:
            st.metric("Duration", f"{duration}s")

        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state['messages'] = []
            st.session_state['start_time'] = datetime.datetime.now()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Download chat history
        if st.session_state['messages']:
            csv_data = 'role,content,timestamp,type\n' + '\n'.join([
                f"{m['role']},{m['content'].replace(',', ';')},{m['timestamp']},{m.get('type', 'text')}" 
                for m in st.session_state['messages']
            ])
            st.download_button(
                label="📥 Download Chat",
                data=csv_data,
                file_name=f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Main content area
    if choice == "🏠 Home":
        st.markdown('<div class="main-header">🤖 AI Chatbot Pro - Enhanced Experience</div>', unsafe_allow_html=True)

        # Welcome message and user input
        if not st.session_state['user_name']:
            with st.container():
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                st.markdown("### 👋 Welcome! What should I call you?")
                user_name = st.text_input("Enter your name:", key="name_input")
                if st.button("Start Chatting") and user_name:
                    st.session_state['user_name'] = user_name
                    st.session_state['messages'].append({
                        "role": "assistant",
                        "content": f"Nice to meet you, {user_name}! I'm your AI assistant. I can chat with you and even view images you share. How can I help you today?",
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "text"
                    })
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Main chat interface
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)

                # Display chat messages
                for message in st.session_state['messages']:
                    if message["role"] == "user":
                        if message.get('type') == 'image':
                            st.markdown(f'<div class="user-message">🖼️ {st.session_state["user_name"]} shared an image</div>', unsafe_allow_html=True)
                            if 'image_data' in message:
                                st.image(base64.b64decode(message['image_data']), caption="Shared image", use_column_width=True)
                        else:
                            st.markdown(f'<div class="user-message">👤 {st.session_state["user_name"]}: {message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="bot-message">🤖 Bot: {message["content"]}</div>', unsafe_allow_html=True)

                    # Timestamp
                    st.caption(f"⏰ {message['timestamp']}")
                    st.write("")

                st.markdown('</div>', unsafe_allow_html=True)

                # Input area with image upload
                st.markdown("### 💬 Send a message")

                # File upload for images
                uploaded_file = st.file_uploader(
                    "📷 Upload an image (optional)", 
                    type=['png', 'jpg', 'jpeg', 'gif'],
                    help="Share an image with the chatbot"
                )

                # Text input
                if prompt := st.chat_input("Type your message here..."):
                    # Handle image upload
                    image_uploaded = False
                    if uploaded_file is not None:
                        try:
                            img_str, processed_image = process_image(uploaded_file)
                            st.session_state['messages'].append({
                                "role": "user",
                                "content": f"[Image uploaded: {uploaded_file.name}]",
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "type": "image",
                                "image_data": img_str
                            })
                            image_uploaded = True
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")

                    # Add user message
                    if prompt.strip():
                        st.session_state['messages'].append({
                            "role": "user",
                            "content": prompt,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": "text"
                        })

                    # Get bot response
                    response = chatbot(prompt, image_uploaded)
                    st.session_state['messages'].append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "text"
                    })

                    # Save to CSV
                    try:
                        with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([
                                prompt, 
                                response, 
                                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "image" if image_uploaded else "text"
                            ])
                    except:
                        pass

                    st.rerun()

            with col2:
                # Quick actions and features
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown("### 🚀 Quick Actions")

                if st.button("👋 Say Hello", use_container_width=True):
                    greeting_msg = f"Hello {st.session_state['user_name']}! How are you doing today?"
                    st.session_state['messages'].append({
                        "role": "assistant",
                        "content": greeting_msg,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "text"
                    })
                    st.rerun()

                if st.button("🎲 Random Fact", use_container_width=True):
                    facts = [
                        "Did you know? The first computer virus was created in 1971!",
                        "Fun fact: Honey never spoils!",
                        "Amazing: A group of flamingos is called a 'flamboyance'!",
                        "Cool fact: Octopuses have three hearts!",
                        "Interesting: The shortest war in history lasted only 38 minutes!"
                    ]
                    fact = random.choice(facts)
                    st.session_state['messages'].append({
                        "role": "assistant",
                        "content": fact,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "text"
                    })
                    st.rerun()

                if st.button("💡 Get Tips", use_container_width=True):
                    tips = [
                        "💡 Tip: You can upload images and I'll acknowledge them!",
                        "💡 Tip: Check the Analytics tab to see your chat patterns!",
                        "💡 Tip: Use the download button to save your chat history!",
                        "💡 Tip: Try asking me different types of questions!",
                        "💡 Tip: The sidebar shows your session statistics in real-time!"
                    ]
                    tip = random.choice(tips)
                    st.session_state['messages'].append({
                        "role": "assistant",
                        "content": tip,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "text"
                    })
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

    elif choice == "💬 Chat History":
        st.markdown('<div class="main-header">💬 Conversation History</div>', unsafe_allow_html=True)

        if os.path.exists('chat_log.csv') or st.session_state['messages']:
            # Display session history
            if st.session_state['messages']:
                st.markdown("### Current Session")
                for i, message in enumerate(st.session_state['messages']):
                    with st.expander(f"Message {i+1} - {message['role'].title()} ({message['timestamp']})"):
                        if message.get('type') == 'image':
                            st.write("🖼️ Image message:")
                            if 'image_data' in message:
                                st.image(base64.b64decode(message['image_data']), use_column_width=True)
                        st.write(f"**Content:** {message['content']}")
                        st.write(f"**Type:** {message.get('type', 'text')}")

            # Display saved history
            if os.path.exists('chat_log.csv'):
                st.markdown("### Saved Chat History")
                try:
                    with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                        csv_reader = csv.reader(csvfile)
                        rows = list(csv_reader)
                        if len(rows) > 1:  # Skip header
                            for i, row in enumerate(rows[1:], 1):
                                if len(row) >= 3:
                                    with st.expander(f"Chat {i} - {row[2]}"):
                                        st.write(f"**User:** {row[0]}")
                                        st.write(f"**Bot:** {row[1]}")
                                        if len(row) > 3:
                                            st.write(f"**Type:** {row[3]}")
                except Exception as e:
                    st.error(f"Error reading chat history: {str(e)}")
        else:
            st.info("No conversation history found. Start chatting to see your messages here!")

    elif choice == "📊 Analytics":
        st.markdown('<div class="main-header">📊 Chat Analytics Dashboard</div>', unsafe_allow_html=True)

        analytics = get_chat_analytics()
        if analytics:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Total Messages", analytics['total_messages'])
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Session Duration", f"{analytics['session_duration']}s")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                user_msgs = analytics['role_counts'].get('user', 0)
                bot_msgs = analytics['role_counts'].get('assistant', 0)
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("User/Bot Ratio", f"{user_msgs}:{bot_msgs}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Message Distribution")
                if not analytics['role_counts'].empty:
                    fig_pie = px.pie(
                        values=analytics['role_counts'].values,
                        names=analytics['role_counts'].index,
                        color_discrete_sequence=['#667eea', '#764ba2']
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.markdown("### Messages by Hour")
                if not analytics['hourly_counts'].empty:
                    fig_bar = px.bar(
                        analytics['hourly_counts'],
                        x='hour',
                        y='count',
                        color_discrete_sequence=['#4facfe']
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No chat data available for analytics. Start chatting to see statistics!")

    elif choice == "ℹ️ About":
        st.markdown('<div class="main-header">ℹ️ About AI Chatbot Pro</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🤖 Project Overview")
            st.write("""
            AI Chatbot Pro is an enhanced conversational AI built with modern web technologies and machine learning. 
            This project demonstrates advanced NLP techniques, interactive UI design, and real-time analytics.
            """)

            st.markdown("### ✨ Key Features")
            st.write("""
            - 🖼️ **Image Upload Support** - Share images with the chatbot
            - 📊 **Real-time Analytics** - Track your conversation patterns
            - 💾 **Chat History** - Save and review past conversations
            - 🎨 **Modern UI** - Beautiful, responsive interface with animations
            - 📱 **Mobile Friendly** - Works seamlessly on all devices
            - 📥 **Export Data** - Download chat history as CSV
            - ⚡ **Fast Response** - Powered by scikit-learn ML models
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🔧 Technology Stack")
            st.write("""
            - **Frontend**: Streamlit with custom CSS
            - **ML Model**: Logistic Regression with TF-IDF
            - **NLP**: NLTK for text processing
            - **Analytics**: Plotly for interactive charts
            - **Image Processing**: PIL (Pillow)
            - **Data Storage**: CSV files for chat history
            """)

            st.markdown("### 🎯 Use Cases")
            st.write("""
            - Customer service automation
            - Educational assistance
            - Entertainment and casual conversation
            - Data collection and user feedback
            - Prototype for larger chatbot systems
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Future Enhancements")
        st.write("""
        - 🧠 Integration with advanced language models (GPT, BERT)
        - 🔍 Image content analysis and description
        - 🗣️ Voice input and text-to-speech output
        - 🌐 Multi-language support
        - 🔐 User authentication and personalization
        - ☁️ Cloud deployment and scaling
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    elif choice == "💌 Feedback":
        st.markdown('<div class="main-header">💌 Share Your Feedback</div>', unsafe_allow_html=True)

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown("### We'd love to hear from you!")
        st.write("Your feedback helps us improve the chatbot experience.")

        col1, col2 = st.columns([2, 1])

        with col1:
            feedback = st.text_area(
                "Please provide your feedback here:",
                height=150,
                help="Tell us what you liked, what could be improved, or suggest new features"
            )
            user_email = st.text_input(
                "Your Email Address:",
                help="We'll use this to follow up on your feedback if needed"
            )

            feedback_type = st.selectbox(
                "Feedback Type:",
                ["General", "Bug Report", "Feature Request", "Compliment", "Complaint"]
            )

            rating = st.slider("Rate your experience:", 1, 5, 3)

        with col2:
            st.markdown("### 📋 Feedback Guidelines")
            st.info("""
            **Good feedback includes:**
            - Specific examples
            - Clear descriptions
            - Suggestions for improvement
            - Steps to reproduce issues
            """)

        if st.button("Submit Feedback", type="primary", use_container_width=True):
            if feedback and user_email:
                with st.spinner("Sending feedback..."):
                    detailed_feedback = f"""
                    Feedback Type: {feedback_type}
                    Rating: {rating}/5 stars
                    User: {st.session_state.get('user_name', 'Anonymous')}

                    Feedback:
                    {feedback}
                    """
                    success, error_msg = send_feedback_email(user_email, detailed_feedback)

                if success:
                    st.success("🎉 Thank you for your feedback! We appreciate your input and will review it shortly.")
                    st.balloons()
                else:
                    st.error(f"❌ There was an error sending your feedback. Please try again.\nError details: {error_msg}")
            else:
                st.error("⚠️ Please fill in all the required fields before submitting.")

        st.markdown('</div>', unsafe_allow_html=True)

    elif choice == "⚙️ Settings":
        st.markdown('<div class="main-header">⚙️ Settings & Preferences</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 👤 Profile Settings")

            new_name = st.text_input("Display Name:", value=st.session_state.get('user_name', ''))
            if st.button("Update Name") and new_name:
                st.session_state['user_name'] = new_name
                st.success("Name updated successfully!")

            st.markdown("### 🎨 Appearance")
            theme_option = st.selectbox("Theme:", ["Modern", "Classic", "Dark"])

            st.markdown("### 💾 Data Management")
            if st.button("Clear All Data", type="secondary"):
                if st.checkbox("I understand this will delete all chat history"):
                    st.session_state['messages'] = []
                    try:
                        if os.path.exists('chat_log.csv'):
                            os.remove('chat_log.csv')
                    except:
                        pass
                    st.success("All data cleared successfully!")
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🔧 System Information")

            st.info(f"""
            **Session Started:** {st.session_state['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
            **Total Messages:** {len(st.session_state['messages'])}
            **Current User:** {st.session_state.get('user_name', 'Not set')}
            **Version:** 2.0 Pro
            """)

            st.markdown("### 📊 Storage Usage")
            try:
                if os.path.exists('chat_log.csv'):
                    file_size = os.path.getsize('chat_log.csv')
                    st.write(f"Chat log size: {file_size} bytes")
                else:
                    st.write("No chat log file found")
            except:
                st.write("Unable to calculate storage usage")

            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    # Create chat_log.csv with header if not exists
    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp', 'Type'])

    main()
