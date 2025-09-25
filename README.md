# 🤖 Intelligent Chatbot Application

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Status](https://img.shields.io/badge/Status-Live-brightgreen.svg)](https://chatbot-app-9y5f3djomdsjgyzqee8guw.streamlit.app)

> An advanced conversational AI chatbot built with natural language processing capabilities, featuring intent recognition and contextual responses.

## ✨ Features

### 🔥 Core Features
- 💬 Interactive Chat Interface - Modern, responsive chat UI with animated bubbles
- 🖼️ Image Upload Support - Share images with the chatbot (acknowledgment feature)
- 📊 Real-time Analytics - Track conversation patterns and statistics
- 💾 Chat History Management - Save, view, and export conversation history
- 📱 Mobile Responsive - Works seamlessly on all devices

### 🎨 Enhanced UI/UX
- 🌈 Modern Gradient Design - Beautiful color schemes and animations
- 🎯 Interactive Dashboard - Multiple tabs with different functionalities
- ⚡ Quick Action Buttons - Fast access to common features
- 📈 Visual Analytics - Charts and graphs for conversation insights
- 🎭 User Personalization - Custom user names and preferences

### 🛠️ Technical Features
- 🧠 Machine Learning - TF-IDF vectorization with Logistic Regression
- 📝 Natural Language Processing - NLTK for text processing
- 💌 Email Integration - Feedback system with email notifications
- 📊 Data Visualization - Interactive charts with Plotly
- 🔐 Environment Variables - Secure configuration management

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
- If you have git installed
- git clone <your-repo-url>
- cd ai-chatbot-pro

- Or download and extract the files to a folder


### Step 2: Install Dependencies
- pip install -r requirements.txt


### Step 3: Set Up Environment (Optional)
For email functionality, create a `.env` file:

- Copy the example file
- cp .env.example .env

- Edit .env with your credentials
- EMAIL_USER=your_email@gmail.com
- EMAIL_PASS=your_app_password


### Step 4: Run the Application

- streamlit run enhanced_chatbot.py

The application will open in your default web browser at http://localhost:8501

## 📚 Usage Guide

### 🏠 Home Tab
- Chat Interface: Type messages in the chat input
- Image Upload: Click the file uploader to share images
- Quick Actions: Use buttons for quick interactions
- User Setup: Enter your name when first starting

### 📊 Analytics Tab
- Session Statistics: View message counts and duration
- Interactive Charts: See conversation patterns
- Export Data: Download chat history as CSV

### 💬 Chat History Tab
- Current Session: View ongoing conversation
- Saved History: Browse past conversations
- Expandable Messages: Click to see full message details

### ⚙️ Settings Tab
- Profile Management: Update display name
- Data Management: Clear chat history
- System Information: View app details

### 💌 Feedback Tab
- Feedback Form: Share your experience
- Rating System: Rate the chatbot (1-5 stars)
- Email Integration: Automatic email notifications

## 🔧 Configuration

### Intents Configuration
Edit `intents.json` to customize chatbot responses:

The application will open in your default web browser at http://localhost:8501

## 📚 Usage Guide

### 🏠 Home Tab
- Chat Interface: Type messages in the chat input
- Image Upload: Click the file uploader to share images
- Quick Actions: Use buttons for quick interactions
- User Setup: Enter your name when first starting

### 📊 Analytics Tab
- Session Statistics: View message counts and duration
- Interactive Charts: See conversation patterns
- Export Data: Download chat history as CSV

### 💬 Chat History Tab
- Current Session: View ongoing conversation
- Saved History: Browse past conversations
- Expandable Messages: Click to see full message details

### ⚙️ Settings Tab
- Profile Management: Update display name
- Data Management: Clear chat history
- System Information: View app details

### 💌 Feedback Tab
- Feedback Form: Share your experience
- Rating System: Rate the chatbot (1-5 stars)
- Email Integration: Automatic email notifications

## 🔧 Configuration

### Intents Configuration
Edit `intents.json` to customize chatbot responses:

[
{
"tag": "greeting",
"patterns": ["Hi", "Hello", "Hey"],
"responses": ["Hello! How can I help you today?"]
}
]



### Email Setup (Gmail)
- Enable 2-Factor Authentication on Gmail
- Generate an App Password
- Use App Password in `.env` file (not your regular password)

### Customizing UI
Modify the CSS styles in the `st.markdown()` sections to change:
- Color schemes
- Animations
- Layout
- Fonts

## 📁 File Structure

ai-chatbot-pro/
├── enhanced_chatbot.py # Main application file
├── intents.json # Chatbot training data
├── requirements.txt # Python dependencies
├── .env.example # Environment variables template
├── .env # Your environment variables (create this)
├── chat_log.csv # Chat history (auto-generated)
├── nltk_data/ # NLTK data (auto-downloaded)
└── README.md # This file


## 🤖 How It Works

### Machine Learning Pipeline
- Data Loading: Intents loaded from JSON file
- Preprocessing: Text patterns extracted and labeled
- Vectorization: TF-IDF transformation of text data
- Training: Logistic Regression model training
- Prediction: Real-time intent classification

### Image Handling
- Upload: User selects image file
- Processing: PIL processes and resizes image
- Storage: Base64 encoding for session storage
- Display: Image shown in chat interface
- Acknowledgment: Bot responds to image upload

### Analytics Engine
- Real-time Tracking: Session statistics
- Data Visualization: Plotly charts
- Pattern Analysis: Message distribution and timing
- Export Functionality: CSV download capability

## 🔍 Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

- pip install -r requirements.txt


#### 2. NLTK Download Issues

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


#### 3. Email Not Working
- Check Gmail App Password setup
- Verify 2FA is enabled
- Ensure `.env` file exists with correct credentials

#### 4. Streamlit Issues

Clear cache
streamlit cache clear

Update Streamlit
pip install --upgrade streamlit


### Performance Optimization
- Large Chat History: Clear old conversations regularly
- Image Size: Images are automatically resized
- Memory Usage: Session state is managed efficiently

## 🚀 Deployment Options

### Local Development


streamlit run enhanced_chatbot.py


### Streamlit Cloud
- Push code to GitHub repository
- Connect to Streamlit Cloud
- Add environment variables in dashboard
- Deploy automatically

FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "enhanced_chatbot.py"]



## 🔮 Future Enhancements

### Planned Features
- 🔍 Image Analysis: AI-powered image content description
- 🗣️ Voice Integration: Speech-to-text and text-to-speech
- 🌐 Multi-language: Support for multiple languages
- 🔐 User Authentication: Login system with user profiles
- ☁️ Database Integration: PostgreSQL/MongoDB support
- 🤖 Advanced AI: Integration with GPT/BERT models

### Contribution Ideas
- Add new intent categories
- Improve UI/UX design
- Implement advanced NLP features
- Add API integrations
- Create mobile app version


## 🤝 Contributing
- Fork the repository
- Create a feature branch
- Make your changes
- Test thoroughly
- Submit a pull request

## 📞 Support
If you encounter any issues or have questions:
- Check the troubleshooting section
- Review the configuration guide
- Create an issue on GitHub
- Use the feedback form in the app

## 🎯 About
AI Chatbot Pro is designed to demonstrate modern chatbot development techniques using Python and Streamlit. It combines machine learning, web development, and user experience design to create an engaging conversational AI.

- Version: 2.0 Pro
- Author: SAURABH KUMAR
- Last Updated: September 2025

⭐ If you find this project helpful, please give it a star! ⭐

Made with ❤️ using Python and Streamlit


