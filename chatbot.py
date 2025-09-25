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
from sklearn.metrics import classification_report
from PIL import Image
import base64
import io
import plotly.express as px
import pandas as pd
import numpy as np

# --- Load environment variables ---
load_dotenv()
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')

# --- NLTK setup ---
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# --- Enhanced Intent Loading with Better Error Handling ---
def load_intents():
    """Load intents from JSON file with comprehensive error handling"""
    file_path = os.path.abspath("./intents.json")
    
    try:
        print(f"Looking for intents file at: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as file:
            intents_data = json.load(file)
            
        # Handle different JSON structures
        if isinstance(intents_data, dict) and 'intents' in intents_data:
            intents = intents_data['intents']
        else:
            intents = intents_data
        
        # Validate and clean intents
        cleaned_intents = []
        for i, intent in enumerate(intents):
            try:
                # Check required fields
                if not all(key in intent for key in ['tag', 'patterns', 'responses']):
                    print(f"Warning: Intent {i} missing required fields, skipping...")
                    continue
                
                # Ensure patterns and responses are lists and not empty
                patterns = intent['patterns']
                responses = intent['responses']
                
                if not isinstance(patterns, list):
                    patterns = [str(patterns)] if patterns else []
                if not isinstance(responses, list):
                    responses = [str(responses)] if responses else []
                
                # Skip empty intents
                if not patterns or not responses:
                    print(f"Warning: Intent {i} has empty patterns or responses, skipping...")
                    continue
                
                # Clean and validate each pattern and response
                clean_patterns = [str(p).strip() for p in patterns if str(p).strip()]
                clean_responses = [str(r).strip() for r in responses if str(r).strip()]
                
                if clean_patterns and clean_responses:
                    cleaned_intents.append({
                        'tag': str(intent['tag']).strip(),
                        'patterns': clean_patterns,
                        'responses': clean_responses
                    })
                    
            except Exception as e:
                print(f"Error processing intent {i}: {e}")
                continue
        
        print(f"Successfully loaded {len(cleaned_intents)} valid intents from JSON file")
        
        # Add some statistics
        total_patterns = sum(len(intent['patterns']) for intent in cleaned_intents)
        unique_tags = set(intent['tag'] for intent in cleaned_intents)
        print(f"Total patterns: {total_patterns}")
        print(f"Unique tags: {len(unique_tags)}")
        
        return cleaned_intents
        
    except FileNotFoundError:
        print("intents.json file not found. Using comprehensive default intents.")
        return get_default_intents()
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        print("Using comprehensive default intents.")
        return get_default_intents()
        
    except Exception as e:
        print(f"Unexpected error loading intents: {e}")
        return get_default_intents()

def get_default_intents():
    """Return comprehensive default intents if JSON file fails"""
    return [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up", "Good morning", "Good afternoon", "Good evening", "Greetings", "Howdy"],
            "responses": ["Hi there!", "Hello!", "Hey! How can I help you?", "I'm doing great! How can I assist you today?", "Hello! Nice to meet you!", "Hi! What can I do for you?"]
        },
        {
            "tag": "goodbye", 
            "patterns": ["Bye", "Goodbye", "See you later", "Take care", "Farewell", "See you", "Until next time", "Catch you later"],
            "responses": ["Goodbye! Have a great day!", "See you later!", "Take care!", "Thanks for chatting!", "Farewell!", "Until next time!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thank you", "Thanks", "Thank you so much", "I appreciate it", "Thanks a lot", "Much appreciated", "Grateful"],
            "responses": ["You're welcome!", "Happy to help!", "Anytime!", "Glad I could assist!", "My pleasure!", "No problem at all!"]
        },
        {
            "tag": "help",
            "patterns": ["Help", "I need help", "Can you help me", "What should I do", "Assist me", "Support", "I'm stuck"],
            "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?", "What can I help you with today?"]
        },
        {
            "tag": "about",
            "patterns": ["What can you do", "Who are you", "What are you", "Tell me about yourself", "Your capabilities", "What is your purpose"],
            "responses": ["I am a chatbot designed to help and assist you!", "My purpose is to provide helpful responses to your questions.", "I can answer questions and provide assistance on various topics.", "I'm here to help you with information and support!"]
        },
        {
            "tag": "age",
            "patterns": ["What is your age", "How old are you", "When were you born", "Your age"],
            "responses": ["I am a bot, I don't have an age.", "Age is just a number for me!", "I exist in digital time, so age doesn't apply to me!"]
        },
        {
            "tag": "weather",
            "patterns": ["What's the weather like", "How's the weather today", "Weather forecast", "Is it raining", "Temperature outside"],
            "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website.", "For current weather, I recommend checking your local weather service."]
        },
        {
            "tag": "name",
            "patterns": ["What's your name", "Who are you", "Your name", "What should I call you"],
            "responses": ["I'm an AI chatbot!", "You can call me Assistant!", "I'm your friendly chatbot helper!", "I'm here to help - you can call me Bot!"]
        },
        {
            "tag": "capabilities",
            "patterns": ["What can you do", "Your features", "How can you help", "What are you capable of"],
            "responses": ["I can chat with you, answer questions, and provide assistance!", "I'm designed to be helpful, informative, and conversational!", "I can help with various topics and provide information!"]
        },
        {
            "tag": "small_talk",
            "patterns": ["How was your day", "What's new", "Tell me something interesting", "Random fact", "Entertain me"],
            "responses": ["Every day is a new adventure in the digital world!", "Did you know that honey never spoils?", "Here's something cool - octopuses have three hearts!", "I find every conversation interesting!"]
        }
    ]

# --- Load intents ---
intents = load_intents()

# --- Advanced preprocessing ---
def preprocess_data(intents):
    """Enhanced data preprocessing with better text normalization"""
    patterns = []
    tags = []
    
    for intent in intents:
        for pattern in intent['patterns']:
            # Basic text cleaning
            clean_pattern = str(pattern).strip().lower()
            if clean_pattern:  # Only add non-empty patterns
                patterns.append(clean_pattern)
                tags.append(intent['tag'])
    
    print(f"Preprocessed {len(patterns)} patterns for training")
    return patterns, tags

# Preprocess data
patterns, tags = preprocess_data(intents)

# --- Enhanced model training ---
def train_model(patterns, tags):
    """Train the chatbot model with optimized parameters"""
    try:
        # Use enhanced TF-IDF parameters
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Reduced from (1,4) to (1,3) for better performance
            lowercase=True,
            stop_words='english',
            max_features=5000,  # Limit features to prevent overfitting
            min_df=1,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )
        
        X = vectorizer.fit_transform(patterns)
        y = tags
        
        # Use optimized LogisticRegression parameters
        clf = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,  # Regularization parameter
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        clf.fit(X, y)
        
        print("Model trained successfully!")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of classes: {len(set(y))}")
        
        return vectorizer, clf
        
    except Exception as e:
        print(f"Error training model: {e}")
        raise

# Train the model
try:
    vectorizer, clf = train_model(patterns, tags)
    model_loaded = True
except Exception as e:
    print(f"Failed to train model: {e}")
    model_loaded = False

# --- Enhanced chatbot function ---
def chatbot(user_input, image_uploaded=False):
    """Enhanced chatbot with better confidence handling and response selection"""
    try:
        if not model_loaded:
            return "Sorry, the chatbot model is not available right now. Please try again later."
            
        if image_uploaded:
            return "I can see you've shared an image! While I can't analyze the image content yet, I appreciate you sharing it with me. How can I help you today?"

        # Clean and validate input
        user_input = str(user_input).strip()
        if not user_input:
            return "I didn't receive any input. Could you please say something?"

        # Transform input
        input_vec = vectorizer.transform([user_input.lower()])
        
        # Get prediction and confidence scores
        predicted_tag = clf.predict(input_vec)[0]
        confidence_scores = clf.predict_proba(input_vec)[0]
        max_confidence = max(confidence_scores)
        
        # Get top predictions for better fallback
        top_classes = clf.classes_[np.argsort(confidence_scores)[::-1][:3]]
        top_scores = sorted(confidence_scores, reverse=True)[:3]
        
        print(f"Input: '{user_input}' -> Top predictions:")
        for i, (cls, score) in enumerate(zip(top_classes, top_scores)):
            print(f"  {i+1}. {cls}: {score:.3f}")

        # Dynamic confidence threshold based on input length and complexity
        base_threshold = 0.1  # Lower base threshold
        
        # Adjust threshold based on input characteristics
        if len(user_input.split()) <= 2:  # Short inputs
            confidence_threshold = base_threshold * 0.8
        elif any(word in user_input.lower() for word in ['help', 'what', 'how', 'who', 'when', 'where', 'why']):
            confidence_threshold = base_threshold * 0.9  # Question words
        else:
            confidence_threshold = base_threshold

        # If confidence is too low, try alternative approaches
        if max_confidence < confidence_threshold:
            # Check if input matches any pattern partially
            best_match_score = 0
            best_match_intent = None
            
            user_words = set(user_input.lower().split())
            
            for intent in intents:
                for pattern in intent['patterns']:
                    pattern_words = set(pattern.lower().split())
                    # Calculate Jaccard similarity
                    intersection = len(user_words.intersection(pattern_words))
                    union = len(user_words.union(pattern_words))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > best_match_score and similarity > 0.3:
                        best_match_score = similarity
                        best_match_intent = intent
            
            if best_match_intent:
                response = random.choice(best_match_intent['responses'])
                print(f"Using pattern matching fallback (similarity: {best_match_score:.3f})")
                return response
            
            # Final fallback responses
            fallback_responses = [
                "I'm not quite sure about that. Could you please rephrase your question or ask something else?",
                "I didn't fully understand that. Can you try asking in a different way?",
                "Hmm, I'm not certain about that topic. Is there something else I can help you with?",
                "I'm still learning! Could you rephrase that or ask about something else?",
                "That's not something I'm familiar with. What else would you like to know?",
                "I'm having trouble understanding that. Could you be more specific or ask something else?"
            ]
            return random.choice(fallback_responses)

        # Find the intent and return response
        for intent in intents:
            if intent['tag'] == predicted_tag:
                responses = intent.get('responses', [])
                if responses:
                    response = random.choice(responses)
                    print(f"Returning response from tag '{predicted_tag}' (confidence: {max_confidence:.3f})")
                    return response
        
        return "Sorry, I don't have a response for that topic."
        
    except Exception as e:
        print(f"Error in chatbot function: {e}")
        return "Sorry, I encountered an error while processing your message. Please try again."

# --- Test function for debugging ---
def test_chatbot_responses():
    """Test the chatbot with various inputs"""
    test_inputs = [
        "Hello",
        "Hi there", 
        "How are you",
        "What's your name",
        "Help me",
        "Thank you",
        "Goodbye",
        "What can you do",
        "Tell me about yourself",
        "Random question that might not match"
    ]
    
    print("=== CHATBOT TEST RESULTS ===")
    for test_input in test_inputs:
        response = chatbot(test_input)
        print(f"Input: '{test_input}' -> Response: '{response}'")
        print("-" * 50)

# --- Rest of your Streamlit application code remains the same ---
def send_feedback_email(user_email, feedback):
    try:
        sender_email = EMAIL_USER
        sender_password = EMAIL_PASS
        
        if not sender_email or not sender_password:
            return False, "Email configuration not found"
            
        receiver_email = EMAIL_USER  # Send to yourself
        subject = "Chatbot Feedback"
        body = f"Feedback from: {user_email}\n\n{feedback}"
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True, None
    except Exception as e:
        return False, str(e)

def process_image(uploaded_file):
    """Process uploaded image and return base64 string for display"""
    try:
        image = Image.open(uploaded_file)
        # Resize image if too large
        max_size = (800, 600)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Convert to base64 for storage in session state
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str, image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def get_chat_analytics():
    """Generate analytics from chat history"""
    if not st.session_state['messages']:
        return None

    try:
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
    except Exception as e:
        print(f"Error generating analytics: {e}")
        return None

def main():
    st.set_page_config(
        page_title="AI Chatbot Pro - Enhanced", 
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
        .success-indicator {
            color: #28a745;
        }
        .error-indicator {
            color: #dc3545;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar with modern design
    with st.sidebar:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.title("🤖 AI Chatbot Pro")
        st.markdown("Your enhanced conversation partner")
        
        # Model status indicator
        if model_loaded:
            st.markdown('<p class="success-indicator">✅ Model Status: Ready</p>', unsafe_allow_html=True)
            st.markdown(f"**Training Data:** {len(patterns)} patterns, {len(set(tags))} categories")
        else:
            st.markdown('<p class="error-indicator">❌ Model Status: Error</p>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

        menu = ["🏠 Home", "💬 Chat History", "📊 Analytics", "ℹ️ About", "💌 Feedback", "⚙️ Settings", "🧪 Debug"]
        choice = st.selectbox("Navigate to:", menu)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = datetime.datetime.now()
    if 'user_name' not in st.session_state:
        st.session_state['user_name'] = ""

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
        st.markdown('<div class="main-header">🤖 AI Chatbot Pro - Enhanced & Fixed</div>', unsafe_allow_html=True)

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
                        "content": f"Nice to meet you, {user_name}! I'm your enhanced AI assistant with improved understanding. I can chat about many topics and even acknowledge images you share. How can I help you today?",
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
                                try:
                                    st.image(base64.b64decode(message['image_data']), caption="Shared image", use_column_width=True)
                                except:
                                    st.write("Image data could not be displayed")
                        else:
                            st.markdown(f'<div class="user-message">👤 {st.session_state["user_name"]}: {message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="bot-message">🤖 Bot: {message["content"]}</div>', unsafe_allow_html=True)

                    # Timestamp
                    st.caption(f"⏰ {message['timestamp']}")
                    st.write("")

                st.markdown('</div>', unsafe_allow_html=True)

                # Input area
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
                        img_str, processed_image = process_image(uploaded_file)
                        if img_str:
                            st.session_state['messages'].append({
                                "role": "user",
                                "content": f"[Image uploaded: {uploaded_file.name}]",
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "type": "image",
                                "image_data": img_str
                            })
                            image_uploaded = True

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
                    except Exception as e:
                        print(f"Error saving to CSV: {e}")

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

    elif choice == "🧪 Debug":
        st.markdown('<div class="main-header">🧪 Debug & Testing</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Model Information")
            
            if model_loaded:
                st.success("✅ Model successfully loaded and trained!")
                st.write(f"**Total intents loaded:** {len(intents)}")
                st.write(f"**Training patterns:** {len(patterns)}")
                st.write(f"**Unique categories:** {len(set(tags))}")
                
                # Show sample categories
                st.markdown("**Sample categories:**")
                sample_tags = list(set(tags))[:10]
                st.write(", ".join(sample_tags))
                
                if hasattr(vectorizer, 'get_feature_names_out'):
                    st.write(f"**Vocabulary size:** {len(vectorizer.get_feature_names_out())}")
            else:
                st.error("❌ Model failed to load!")

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🧪 Test Chatbot")
            
            test_input = st.text_input("Test message:", key="debug_test")
            
            if st.button("Test Response") and test_input:
                with st.spinner("Processing..."):
                    response = chatbot(test_input)
                    st.write(f"**Input:** {test_input}")
                    st.write(f"**Response:** {response}")
            
            st.markdown("### 🎯 Quick Tests")
            
            if st.button("Run Comprehensive Test", use_container_width=True):
                with st.spinner("Running tests..."):
                    # This will print to console, but we'll also show some results
                    test_responses = []
                    test_inputs = ["Hello", "Help me", "Thank you", "Goodbye", "What's your name"]
                    
                    for test_input in test_inputs:
                        response = chatbot(test_input)
                        test_responses.append((test_input, response))
                    
                    st.write("**Test Results:**")
                    for inp, resp in test_responses:
                        st.write(f"• '{inp}' → '{resp[:50]}{'...' if len(resp) > 50 else ''}'")
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Intent Explorer
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📚 Intent Explorer")
        
        if intents:
            selected_intent = st.selectbox(
                "Select an intent to explore:",
                options=range(len(intents)),
                format_func=lambda x: f"{intents[x]['tag']} ({len(intents[x]['patterns'])} patterns)"
            )
            
            intent = intents[selected_intent]
            st.write(f"**Tag:** {intent['tag']}")
            st.write(f"**Sample Patterns:** {', '.join(intent['patterns'][:3])}")
            st.write(f"**Sample Responses:** {', '.join(intent['responses'][:2])}")
            
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
        st.markdown('<div class="main-header">ℹ️ About AI Chatbot Pro - Enhanced</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🚀 Latest Improvements")
            st.write("""
            **Enhanced Features in this version:**
            - 🎯 **Smarter Confidence Handling** - Dynamic thresholds based on input type
            - 🧠 **Pattern Matching Fallback** - Secondary matching when ML confidence is low  
            - 📊 **Better Training Data Processing** - Comprehensive JSON validation and cleaning
            - 🔍 **Advanced Preprocessing** - Improved text normalization and feature extraction
            - 🎨 **Debug Interface** - Built-in testing and model exploration tools
            - ⚡ **Performance Optimization** - Faster response times with balanced accuracy
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🔧 Technical Specifications")
            st.write("""
            - **ML Algorithm:** Logistic Regression with L2 regularization
            - **Text Vectorization:** TF-IDF with 1-3 n-grams
            - **Features:** Up to 5000 with balanced class weights
            - **Fallback Strategy:** Jaccard similarity pattern matching
            - **Confidence Adaptation:** Dynamic threshold based on input type
            - **Data Validation:** Multi-level JSON structure verification
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
        try:
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp', 'Type'])
        except Exception as e:
            print(f"Error creating chat log file: {e}")

    # Run tests if this is the main module
    if len(intents) > 0:
        print("Running initial chatbot tests...")
        test_chatbot_responses()
        print("Tests completed. Starting Streamlit app...")

    main()