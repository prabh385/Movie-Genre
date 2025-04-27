🎬 Movie Genre Predictor
A machine learning-based web application that predicts movie genres from plot descriptions. It offers an interactive, user-friendly interface with confidence scores, visual analytics, and model insights.


🎯 Objectives

1. Genre Prediction: Accurately classify movie plots into one or more genres.

2. User-Friendly Interface: Smooth text input and intuitive results visualization.

3. Visual Analytics: Confidence scores and key term highlights via charts.

4. Performance: Fast, responsive predictions.

5. Educational Value: Transparency through model information.


🚀 Features

1. Text input for custom movie plots or quick selection of example plots.

2. 3D interactive UI elements for a modern feel.

3. Multi-label genre prediction with associated confidence scores.

4. visualize important keywords.

5. Model performance metrics displayed clearly.

6. Dark mode and light mode toggle for better accessibility.


💻 Installation & Setup
Prerequisites
1. Python 3.8 or higher
2. pip package manager

Steps to Run the Project
1. Clone the repository
    git clone https://github.com/prabh385/Movie-Genre.git

2. Create a virtual environment (recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install dependencies
    pip install -r requirements.txt

4. Download model files
    Place genre_classifier.pkl and mlb.pkl inside the model/ directory. (Ensure these files are included in your project repository.)

5. Run the application
    streamlit run app.py
    Access the app
    It will automatically open at: http://localhost:8501
    If not, manually navigate to the link in your browser.


📚 Tech Stack
1. Frontend: Streamlit
2. Backend: Python
3. Machine Learning: Scikit-learn, pandas, numpy,
4. Visualization: Matplotlib

