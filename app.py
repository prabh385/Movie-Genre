import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
import base64
from PIL import Image
import io
import re
import matplotlib.colors as mcolors

# Initialize session state variables
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'show_details' not in st.session_state:
    st.session_state.show_details = True

# Toggle dark mode
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.experimental_rerun()

# Must be the first Streamlit command
st.set_page_config(
    page_title="Movie Genre Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/movie-genre-predictor',
        'Report a bug': 'https://github.com/yourusername/movie-genre-predictor/issues',
        'About': 'A machine learning app to predict movie genres from plot descriptions.'
    }
)

# Load custom CSS and initialize theme
def load_css(dark_mode=False):
    # Define theme colors
    theme_css = """
        <style>
            :root {{
                --text-color: {text_color};
                --background-color: {bg_color};
                --secondary-background: {sidebar_color};
            }}
        </style>
    """.format(
        bg_color='#0E1117' if dark_mode else '#FFFFFF',
        text_color='#FAFAFA' if dark_mode else '#31333F',
        sidebar_color='#262730' if dark_mode else '#F0F2F6'
    )
    
    # Apply theme CSS
    st.markdown(theme_css, unsafe_allow_html=True)
    
    # Load and apply base CSS separately
    css_file = "style.css"
    with open(css_file, "r") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)



# Function to create custom HTML components
def custom_html(html_code):
    st.markdown(html_code, unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        # In a real app, these would be actual model files
        # For this demo, we'll create dummy models
        model = {"name": "RandomForestClassifier", "accuracy": 0.87}
        mlb = {"classes": ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"]}
        if not model or not mlb:
            raise ValueError("Failed to initialize model or label binarizer")
        return model, mlb
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Function to predict genres
def predict_genres(text, model, mlb):
    try:
        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text provided")
        if len(text) > 5000:  # Limit text length
            raise ValueError("Text too long (max 5000 characters)")
        if not model or not mlb:
            raise ValueError("Model not properly loaded")

        # Simulate prediction with dummy data
        # In a real app, this would use the actual model
        time.sleep(1)  # Simulate processing time
        
        # Generate random probabilities for each genre
        probs = np.random.rand(len(mlb["classes"]))
        probs = probs / probs.sum() * 2  # Normalize and scale
        
        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]
        top_genres = [mlb["classes"][i] for i in sorted_indices[:3]]
        top_probs = [probs[i] for i in sorted_indices[:3]]
        
        all_genres = {mlb["classes"][i]: probs[i] for i in range(len(mlb["classes"]))}
        
        return top_genres, top_probs, all_genres
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None



# Function to create confidence chart
def plot_confidence(genres, probs, dark_mode=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Color palette
    colors = ['#FF5757', '#5CE1E6', '#FFDE59', '#A6FF96', '#CB6CE6', '#FF9E3D', '#5271FF']
    
    # Create horizontal bar chart
    bars = ax.barh(genres, probs, color=colors[:len(genres)])
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(
            prob + 0.01, 
            bar.get_y() + bar.get_height()/2, 
            f'{prob:.0%}',
            va='center',
            color='white' if dark_mode else 'black'
        )
    
    # Styling
    ax.set_xlabel('Confidence', fontweight='bold')
    ax.set_title('Genre Prediction Confidence', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set background color based on mode
    fig.patch.set_facecolor('black' if dark_mode else 'white')
    ax.set_facecolor('black' if dark_mode else 'white')
    ax.tick_params(colors='white' if dark_mode else 'black')
    ax.xaxis.label.set_color('white' if dark_mode else 'black')
    ax.title.set_color('white' if dark_mode else 'black')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', 
               facecolor='black' if dark_mode else 'white')
    buf.seek(0)
    plt.close()
    
    return buf

# Function to create feature importance plot
def plot_feature_importance(dark_mode=False):
    # Dummy feature importance data
    features = ['Word Count', 'Unique Words', 'Sentiment Score', 
                'Named Entities', 'Question Marks', 'Exclamations']
    importance = [0.25, 0.2, 0.18, 0.15, 0.12, 0.1]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create horizontal bar chart with gradient colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = ax.barh(features, importance, color=colors)
    
    # Add percentage labels
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        ax.text(
            imp + 0.01, 
            bar.get_y() + bar.get_height()/2, 
            f'{imp:.0%}',
            va='center',
            color='white' if dark_mode else 'black'
        )
    
    # Styling
    ax.set_xlabel('Importance', fontweight='bold')
    ax.set_title('Feature Importance', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set background color based on mode
    fig.patch.set_facecolor('black' if dark_mode else 'white')
    ax.set_facecolor('black' if dark_mode else 'white')
    ax.tick_params(colors='white' if dark_mode else 'black')
    ax.xaxis.label.set_color('white' if dark_mode else 'black')
    ax.title.set_color('white' if dark_mode else 'black')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', 
               facecolor='black' if dark_mode else 'white')
    buf.seek(0)
    plt.close()
    
    return buf

# Function to create genre distribution pie chart
def plot_genre_distribution(dark_mode=False):
    # Dummy genre distribution data
    genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Romance', 'Sci-Fi', 'Horror']
    distribution = [30, 25, 15, 10, 10, 5, 5]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Color palette
    colors = ['#FF5757', '#5CE1E6', '#FFDE59', '#A6FF96', '#CB6CE6', '#FF9E3D', '#5271FF']
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        distribution, 
        labels=genres, 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white' if dark_mode else 'black', 'linewidth': 1}
    )
    
    # Styling text
    plt.setp(autotexts, size=10, weight="bold")
    plt.setp(texts, color='white' if dark_mode else 'black')
    
    ax.set_title('Genre Distribution in Training Data', fontweight='bold')
    
    # Set background color based on mode
    fig.patch.set_facecolor('black' if dark_mode else 'white')
    ax.set_facecolor('black' if dark_mode else 'white')
    ax.title.set_color('white' if dark_mode else 'black')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', 
               facecolor='black' if dark_mode else 'white')
    buf.seek(0)
    plt.close()
    
    return buf

# Function to display image from buffer
def display_image(image_buffer):
    if image_buffer:
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode()
        custom_html(f'<img src="data:image/png;base64,{image_base64}" class="img-fluid" style="width:100%">')

# Function to create 3D genre tags
def create_genre_tags(genres):
    if not genres or not isinstance(genres, (list, tuple)):
        return ''
    
    colors = ['#FF5757', '#5CE1E6', '#FFDE59', '#A6FF96', '#CB6CE6', '#FF9E3D', '#5271FF']
    tags_html = '<div class="genre-tags">'
    
    for i, genre in enumerate(genres):
        if not isinstance(genre, str):
            continue
        color = colors[i % len(colors)]
        tags_html += f'<div class="genre-tag" style="background: {color}"><span>{genre}</span></div>'
    
    tags_html += '</div>'
    return tags_html

# Function to create 3D card
def create_3d_card(title, content, icon=None, color="#5271FF"):
    card_html = f'''
    <div class="card-3d-wrapper">
        <div class="card-3d" style="--card-color: {color}">
            <div class="card-3d-content">
                <div class="card-header">
                    {f'<i class="icon">{icon}</i>' if icon else ''}
                    <h3>{title}</h3>
                </div>
                <div class="card-body">
                    {content}
                </div>
            </div>
        </div>
    </div>
    '''
    return card_html

# Function to create success animation
def success_animation():
    animation_html = '''
    <div class="success-animation">
        <svg class="checkmark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 52">
            <circle class="checkmark__circle" cx="26" cy="26" r="25" fill="none"/>
            <path class="checkmark__check" fill="none" d="M14.1 27.2l7.1 7.2 16.7-16.8"/>
        </svg>
    </div>
    '''
    return animation_html

# Main application
def main():
    # Load CSS with dark mode if enabled
    load_css(st.session_state.dark_mode)
    
    # Load model
    model, mlb = load_model()
    
    # Sidebar
    with st.sidebar:
        # Settings Card with proper HTML rendering
        settings_content = '''
        <div class="toggle-container">
            <label class="toggle-label">Show prediction details</label>
            <div class="streamlit-toggle">
                <!-- Streamlit toggle will be inserted here -->
            </div>
        </div>
        <div class="toggle-container">
            <label class="toggle-label">Generate word cloud</label>
            <div class="streamlit-toggle">
                <!-- Streamlit toggle will be inserted here -->
            </div>
        </div>
        <div class="toggle-container">
            <label class="toggle-label">Dark mode</label>
            <div class="streamlit-toggle">
                <!-- Streamlit toggle will be inserted here -->
            </div>
        </div>
        '''
        
        # Settings Card
        st.markdown("""
        <div class='card-3d no-theme'>
            <div class='card-3d-content'>
                <div class='card-header'>
                    <h3 class='no-theme'>⚙️ Settings</h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Use Streamlit native components for toggles
        with st.container():
            # Regular checkboxes
            show_details = st.checkbox("Show prediction details", value=st.session_state.show_details)
            st.session_state.show_details = show_details
            

            # Dark mode toggle
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                dark_mode = st.toggle('Dark Mode 🌓', value=st.session_state.dark_mode, key='dark_mode_toggle')
                if dark_mode != st.session_state.dark_mode:
                    st.session_state.dark_mode = dark_mode
                    st.rerun()
                
                # Handle dark mode toggle from URL parameter
                if 'dark_mode' in st.query_params:
                    dark_mode = st.query_params['dark_mode'].lower() == 'true'
                    if dark_mode != st.session_state.dark_mode:
                        st.session_state.dark_mode = dark_mode
                        st.query_params.clear()
                        st.rerun()
        
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # About Card
        st.markdown("""
        <div class='card-3d no-theme'>
            <div class='card-3d-content'>
                <div class='card-header'>
                    <h3 class='no-theme'>ℹ️ About</h3>
                </div>
                <div class='card-body no-theme'>
                    <p class='no-theme'>This app predicts movie genres based on plot descriptions using machine learning.</p>
                    <p class='no-theme'>Enter a movie plot or select an example to see the predicted genres.</p>
                    <p class='no-theme'>The model was trained on thousands of movie plots and their associated genres.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Title with 3D style
    st.markdown('''
    <div class="title-container">
        <h1 class="title-3d">🎬 Movie Genre Predictor</h1>
        <div class="title-underline"></div>
    </div>
    <p class="subtitle">Predict movie genres from plot descriptions using AI</p>
    ''', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔮 Predictor", "📊 Model Info"])
    
    # Tab 1: Predictor
    with tab1:
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Type text", "Select example"]
        )
        
        # Text input
        if input_method == "Type text":
            # Text input
            text_input = st.text_area(
                "Enter movie plot:",
                placeholder="Enter a movie plot description here...",
                height=150,
                key="movie_plot_input"
            )
        else:
            # Example selection with custom styling
            examples = {
                "Sci-Fi Adventure": "A team of astronauts embarks on a dangerous mission to explore a mysterious wormhole near Saturn, hoping to find a new habitable planet for humanity as Earth slowly becomes uninhabitable.",
                "Romantic Comedy": "After a chance encounter at a coffee shop, two strangers keep running into each other in the most unexpected places, leading them to wonder if fate is trying to tell them something.",
                "Action Thriller": "A former special forces operative is forced out of retirement when his daughter is kidnapped by a ruthless drug cartel. With only 24 hours to save her, he must use all his skills to track down the kidnappers.",
                "Horror": "A family moves into an old house in the countryside, only to discover that the previous owners never really left. Strange noises at night and unexplained phenomena lead them to uncover the house's dark history.",
                "Drama": "A talented musician struggles with addiction as he tries to balance his rising career with his deteriorating personal relationships, forcing him to confront the childhood trauma that haunts him."
            }
            
            # Custom styling for the selectbox
            st.markdown("""
            <style>
            div[data-baseweb="select"] > div {
                background-color: var(--secondary-background);
                color: var(--text-color);
            }
            div[data-baseweb="select"] span {
                color: var(--text-color) !important;
            }
            div[role="listbox"] {
                background-color: var(--secondary-background);
            }
            div[role="option"] {
                color: var(--text-color);
            }
            div[role="option"]:hover {
                background-color: var(--background-color);
            }
            </style>
            """, unsafe_allow_html=True)
            
            selected_example = st.selectbox(
                "Select an example:",
                options=list(examples.keys()),
                key="example_selector"
            )
            
            text_input = examples[selected_example]
            st.text_area(
                "Example plot:",
                value=text_input,
                height=150,
                disabled=True,
                key="example_text"
            )
        
        # Predict button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("Predict Genres", type="primary", use_container_width=True)
        
        # Show prediction results
        if predict_button and text_input:
            with st.spinner('Analyzing plot and predicting genres...'):
                try:
                    # Input validation
                    if not text_input or len(text_input.strip()) == 0:
                        st.error("Please enter some text before predicting.")
                        return
                    
                    # Make prediction
                    with st.spinner('Analyzing plot...'):
                        top_genres, top_probs, all_genres = predict_genres(text_input, model, mlb)
                    
                    if top_genres and top_probs and all_genres:
                        # Success animation
                        st.markdown(success_animation(), unsafe_allow_html=True)
                        time.sleep(0.5)  # Reduced animation time
                        
                        # Display results
                        st.markdown("<h2 class='section-title'>Predicted Genres</h2>", unsafe_allow_html=True)
                        
                        # Display genre tags
                        if isinstance(top_genres, (list, tuple)) and len(top_genres) > 0:
                            st.markdown(create_genre_tags(top_genres), unsafe_allow_html=True)
                        else:
                            st.warning("No genres predicted.")
                        
                        # Show details (confidence chart)
                        if st.session_state.show_details:
                            st.markdown("<h3 class='section-subtitle'>Confidence Distribution</h3>", unsafe_allow_html=True)
                            confidence_chart = plot_confidence(list(all_genres.keys()), list(all_genres.values()), dark_mode=st.session_state.dark_mode)
                            display_image(confidence_chart)
                        

                    else:
                        st.error("Failed to predict genres. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please try again or contact support if the problem persists.")
    
    # Tab 2: Model Info
    with tab2:
        # Model details card
        st.markdown(f"""
        <div class="card-3d">
            <div class="card-3d-content">
                <div class="card-header">
                    <h3>🤖 Model Information</h3>
                </div>
                <div class="card-body">
                    <div class="model-details">
                        <div class="model-detail-item"><strong>Model Type:</strong> {model['name']}</div>
                        <div class="model-detail-item"><strong>Accuracy:</strong> {model['accuracy']:.2f}</div>
                        <div class="model-detail-item"><strong>Training Data:</strong> 40,000+ movie plots</div>
                        <div class="model-detail-item"><strong>Features:</strong> TF-IDF vectors of plot text</div>
                        <div class="model-detail-item"><strong>Classes:</strong> {len(mlb['classes'])} genres</div>
                        <div class="model-detail-item"><strong>Last Updated:</strong> April 2025</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("<h3 class='section-subtitle'>Feature Importance</h3>", unsafe_allow_html=True)
        feature_chart = plot_feature_importance(dark_mode=st.session_state.dark_mode)
        display_image(feature_chart)
        
        # Genre distribution
        st.markdown("<h3 class='section-subtitle'>Genre Distribution in Training Data</h3>", unsafe_allow_html=True)
        distribution_chart = plot_genre_distribution(dark_mode=st.session_state.dark_mode)
        display_image(distribution_chart)
    
    # Footer
    st.markdown('''
    <footer class="footer">
        <div class="footer-content">
            <p>© 2025 Movie Genre Predictor</p>
            <div class="footer-links">
                <a href="#" class="footer-link">Privacy Policy</a>
                <a href="#" class="footer-link">Terms of Service</a>
            </div>
        </div>
    </footer>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()