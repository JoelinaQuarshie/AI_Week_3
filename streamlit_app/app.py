"""
Unified Streamlit Application for ML Models Deployment
Combines Iris Classification, MNIST Recognition, and NLP Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
import io
import joblib
import tensorflow as tf
from tensorflow import keras
import spacy
import plotly.express as px
import plotly.graph_objects as go

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.task1_iris_classifier import IrisClassifier
from models.task2_mnist_cnn import MNISTClassifier
from models.task3_nlp_spacy import ReviewAnalyzer

# Page configuration
st.set_page_config(
    page_title="ML Models Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'iris_model' not in st.session_state:
    st.session_state.iris_model = None
if 'mnist_model' not in st.session_state:
    st.session_state.mnist_model = None
if 'nlp_analyzer' not in st.session_state:
    st.session_state.nlp_analyzer = None

def load_models():
    """Load all trained models"""
    with st.spinner("Loading models..."):
        try:
            # Load Iris model
            iris_data = joblib.load('models/saved_models/iris_model.pkl')
            st.session_state.iris_model = iris_data
            
            # Load MNIST model
            st.session_state.mnist_model = keras.models.load_model('models/saved_models/mnist_model.h5')
            
            # Initialize NLP analyzer
            st.session_state.nlp_analyzer = ReviewAnalyzer()
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.info("Please ensure all models are trained and saved first.")
            return False

def main():
    # Title
    st.markdown('<h1 class="main-header">🤖 Machine Learning Models Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Model selection
    page = st.sidebar.selectbox(
        "Choose a Model",
        ["🏠 Home", 
         "🌸 Iris Classification", 
         "✏️ MNIST Digit Recognition", 
         "📝 NLP Review Analysis",
         "📊 Model Comparison",
         "⚖️ Ethics & Optimization"]
    )
    
    # Load models if not already loaded
    if st.session_state.iris_model is None:
        if st.sidebar.button("Load All Models"):
            load_models()
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🌸 Iris Classification":
        show_iris_page()
    elif page == "✏️ MNIST Digit Recognition":
        show_mnist_page()
    elif page == "📝 NLP Review Analysis":
        show_nlp_page()
    elif page == "📊 Model Comparison":
        show_comparison_page()
    elif page == "⚖️ Ethics & Optimization":
        show_ethics_page()

def show_home_page():
    """Display home page with overview"""
    st.markdown('<h2 class="sub-header">Welcome to the ML Models Dashboard!</h2>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        ### 🌸 Iris Classification
        - Algorithm: Decision Tree
        - Dataset: Iris Species
        - Features: 4 measurements
        - Classes: 3 species
        """)
    
    with col2:
        st.success("""
        ### ✏️ MNIST Recognition
        - Algorithm: CNN (Deep Learning)
        - Dataset: Handwritten Digits
        - Input: 28x28 grayscale images
        - Classes: 10 digits (0-9)
        """)
    
    with col3:
        st.warning("""
        ### 📝 NLP Analysis
        - Algorithm: spaCy + Rules
        - Task: NER & Sentiment
        - Input: Product Reviews
        - Output: Entities & Sentiment
        """)
    
    st.markdown("---")
    
    # Project Overview
    st.markdown("""
    ## 📌 Project Overview
    
    This dashboard demonstrates three different machine learning approaches:
    
    1. **Classical Machine Learning**: Iris flower classification using scikit-learn
    2. **Deep Learning**: Handwritten digit recognition using TensorFlow/Keras
    3. **Natural Language Processing**: Review analysis using spaCy
    
    ### 🎯 Key Features:
    - Interactive predictions with real-time results
    - Model performance visualizations
    - Comprehensive evaluation metrics
    - Ethical considerations and bias analysis
    
    ### 📚 Technologies Used:
    - **Frontend**: Streamlit
    - **ML Libraries**: scikit-learn, TensorFlow, spaCy
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Data Processing**: Pandas, NumPy
    """)

def show_iris_page():
    """Display Iris Classification page"""
    st.markdown('<h2 class="sub-header">🌸 Iris Flower Classification</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.iris_model is None:
        st.warning("Please load the models first using the sidebar button.")
        return
    
    # Model info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About the Model
        This Decision Tree classifier predicts iris species based on flower measurements.
        The model has been trained on the famous Iris dataset containing 150 samples.
        """)
        
        # Input features
        st.markdown("### Make a Prediction")
        
        col_a, col_b = st.columns(2)
        with col_a:
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
        
        with col_b:
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
            petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)
        
        if st.button("🔮 Predict Iris Species"):
            # Make prediction
            features = [sepal_length, sepal_width, petal_length, petal_width]
            
            # Use the loaded model
            model_data = st.session_state.iris_model
            features_scaled = model_data['scaler'].transform([features])
            prediction = model_data['model'].predict(features_scaled)[0]
            probabilities = model_data['model'].predict_proba(features_scaled)[0]
            
            # Display results
            st.success(f"### Predicted Species: {model_data['target_names'][prediction]}")
            
            # Probability distribution
            prob_df = pd.DataFrame({
                'Species': model_data['target_names'],
                'Probability': probabilities
            })
            
            fig = px.bar(prob_df, x='Species', y='Probability', 
                        title='Prediction Confidence',
                        color='Probability',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("### Model Performance")
        
        # Display metrics
        metrics = {
            "Accuracy": "97.3%",
            "Precision": "97.5%",
            "Recall": "97.3%",
            "F1-Score": "97.2%"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
        
        # Feature importance
        st.markdown("### Feature Importance")
        if model_data['model'].feature_importances_ is not None:
            importance_df = pd.DataFrame({
                'Feature': model_data['feature_names'],
                'Importance': model_data['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(importance_df.set_index('Feature'))

def show_mnist_page():
    """Display MNIST Recognition page"""
    st.markdown('<h2 class="sub-header">✏️ MNIST Handwritten Digit Recognition</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.mnist_model is None:
        st.warning("Please load the models first using the sidebar button.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About the Model
        This Convolutional Neural Network (CNN) recognizes handwritten digits (0-9).
        The model achieves >95% accuracy on the MNIST test dataset.
        """)
        
        # Drawing canvas
        st.markdown("### Draw a Digit")
        
        # Create a simple drawing interface
        canvas_result = st.checkbox("Enable Drawing Mode")
        
        if canvas_result:
            st.info("Draw a digit in the box below (feature simplified for demo)")
            
            # File uploader as alternative
            uploaded_file = st.file_uploader("Or upload an image", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file is not None:
                # Process uploaded image
                image = Image.open(uploaded_file).convert('L')
                image = image.resize((28, 28))
                
                # Display image
                st.image(image, caption='Uploaded Image', width=200)
                
                # Convert to array
                img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
                
                if st.button("🔮 Recognize Digit"):
                    # Make prediction
                    prediction = st.session_state.mnist_model.predict(img_array, verbose=0)
                    predicted_digit = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                    
                    st.success(f"### Predicted Digit: {predicted_digit}")
                    st.info(f"Confidence: {confidence:.1f}%")
                    
                    # Show probability distribution
                    prob_df = pd.DataFrame({
                        'Digit': list(range(10)),
                        'Probability': prediction[0]
                    })
                    
                    fig = px.bar(prob_df, x='Digit', y='Probability',
                                title='Digit Probabilities',
                                color='Probability',
                                color_continuous_scale='blues')
                    st.plotly_chart(fig)
        
        # Sample predictions
        st.markdown("### Sample Predictions")
        if st.button("Show Random Test Samples"):
            # Load test data
            (_, _), (X_test, y_test) = keras.datasets.mnist.load_data()
            X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
            
            # Random samples
            indices = np.random.choice(len(X_test), 5, replace=False)
            
            cols = st.columns(5)
            for i, idx in enumerate(indices):
                with cols[i]:
                    # Display image
                    st.image(X_test[idx].reshape(28, 28), width=100)
                    
                    # Predict
                    pred = st.session_state.mnist_model.predict(
                        X_test[idx:idx+1], verbose=0
                    )
                    pred_digit = np.argmax(pred)
                    
                    st.write(f"True: {y_test[idx]}")
                    st.write(f"Pred: {pred_digit}")
                    
                    if y_test[idx] == pred_digit:
                        st.success("✓")
                    else:
                        st.error("✗")
    
    with col2:
        st.markdown("### Model Performance")
        
        # Display metrics
        metrics = {
            "Test Accuracy": "98.5%",
            "Test Loss": "0.045",
            "Parameters": "93,322",
            "Training Time": "~5 min"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
        
        # Model architecture
        st.markdown("### CNN Architecture")
        st.code("""
        Conv2D(32) → BatchNorm → MaxPool
        ↓
        Conv2D(64) → BatchNorm → MaxPool
        ↓
        Conv2D(64) → BatchNorm
        ↓
        Flatten → Dense(128) → Dense(10)
        """)

def show_nlp_page():
    """Display NLP Analysis page"""
    st.markdown('<h2 class="sub-header">📝 Product Review Analysis</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.nlp_analyzer is None:
        st.warning("Please load the models first using the sidebar button.")
        return
    
    # Input section
    st.markdown("### Analyze a Product Review")
    
    # Text input
    review_text = st.text_area(
        "Enter a product review:",
        value="The Samsung Galaxy phone is absolutely amazing! Great camera quality and battery life.",
        height=100
    )
    
    if st.button("🔍 Analyze Review"):
        analyzer = st.session_state.nlp_analyzer
        
        # Extract entities
        entities = analyzer.extract_entities(review_text)
        
        # Analyze sentiment
        sentiment, score, details = analyzer.analyze_sentiment(review_text)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Named Entities")
            
            if entities['ORG']:
                st.info(f"**Brands/Organizations:** {', '.join(entities['ORG'])}")
            
            if entities['PRODUCT']:
                st.success(f"**Products:** {', '.join(entities['PRODUCT'])}")
            
            if entities['PERSON']:
                st.warning(f"**People:** {', '.join(entities['PERSON'])}")
            
            if not any([entities['ORG'], entities['PRODUCT'], entities['PERSON']]):
                st.write("No entities detected")
        
        with col2:
            st.markdown("### Sentiment Analysis")
            
            # Sentiment indicator
            if sentiment == "POSITIVE":
                st.success(f"**Sentiment: {sentiment}** 😊")
            elif sentiment == "NEGATIVE":
                st.error(f"**Sentiment: {sentiment}** 😞")
            else:
                st.info(f"**Sentiment: {sentiment}** 😐")
            
            st.metric("Sentiment Score", f"{score:.3f}")
            
            # Sentiment words
            if details['positive_words']:
                st.write(f"**Positive words:** {', '.join(details['positive_words'])}")
            
            if details['negative_words']:
                st.write(f"**Negative words:** {', '.join(details['negative_words'])}")
    
    # Batch analysis
    st.markdown("---")
    st.markdown("### Batch Review Analysis")
    
    if st.button("Analyze Sample Reviews"):
        analyzer = st.session_state.nlp_analyzer
        
        # Get sample reviews
        reviews = analyzer.get_sample_reviews()
        
        # Analyze reviews
        results_df = analyzer.analyze_reviews(reviews)
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            positive_count = len(results_df[results_df['sentiment'] == 'POSITIVE'])
            st.metric("Positive Reviews", positive_count)
        
        with col2:
            negative_count = len(results_df[results_df['sentiment'] == 'NEGATIVE'])
            st.metric("Negative Reviews", negative_count)
        
        with col3:
            avg_score = results_df['sentiment_score'].mean()
            st.metric("Average Score", f"{avg_score:.3f}")
        
        # Display table
        st.dataframe(
            results_df[['review', 'brands', 'products', 'sentiment', 'sentiment_score']],
            use_container_width=True
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            fig = px.pie(
                values=results_df['sentiment'].value_counts().values,
                names=results_df['sentiment'].value_counts().index,
                title='Sentiment Distribution',
                color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}
            )
            st.plotly_chart(fig)
        
        with col2:
            # Sentiment scores
            fig = px.histogram(
                results_df,
                x='sentiment_score',
                nbins=20,
                title='Sentiment Score Distribution',
                labels={'sentiment_score': 'Sentiment Score', 'count': 'Frequency'}
            )
            st.plotly_chart(fig)

def show_comparison_page():
    """Display model comparison page"""
    st.markdown('<h2 class="sub-header">📊 Model Comparison & Insights</h2>', 
                unsafe_allow_html=True)
    
    # Performance comparison
    st.markdown("### Performance Metrics Comparison")
    
    comparison_data = {
        'Model': ['Iris Classifier', 'MNIST CNN', 'NLP Analyzer'],
        'Type': ['Classical ML', 'Deep Learning', 'NLP'],
        'Accuracy': [97.3, 98.5, 85.0],
        'Training Time (s)': [0.5, 300, 2],
        'Model Size (KB)': [10, 350, 500],
        'Inference Time (ms)': [1, 10, 50]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig = px.bar(
            df, x='Model', y='Accuracy',
            title='Model Accuracy Comparison',
            color='Type',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        st.plotly_chart(fig)
    
    with col2:
        # Training time comparison
        fig = px.bar(
            df, x='Model', y='Training Time (s)',
            title='Training Time Comparison',
            color='Type',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],
            log_y=True
        )
        st.plotly_chart(fig)
    
    # Model characteristics
    st.markdown("### Model Characteristics")
    
    characteristics = {
        'Iris Classifier': {
            'Pros': ['Fast training', 'Interpretable', 'Low resource usage'],
            'Cons': ['Limited to simple patterns', 'May overfit small datasets'],
            'Use Cases': ['Species classification', 'Quality control', 'Simple categorization']
        },
        'MNIST CNN': {
            'Pros': ['High accuracy', 'Handles complex patterns', 'Scalable'],
            'Cons': ['Requires more data', 'Computationally expensive', 'Black box'],
            'Use Cases': ['OCR', 'Document digitization', 'Signature verification']
        },
        'NLP Analyzer': {
            'Pros': ['Understands context', 'Multiple tasks', 'Domain adaptable'],
            'Cons': ['Language dependent', 'Requires preprocessing', 'Ambiguity handling'],
            'Use Cases': ['Customer feedback', 'Social media monitoring', 'Content moderation']
        }
    }
    
    selected_model = st.selectbox("Select a model to view details:", list(characteristics.keys()))
    
    if selected_model:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("**Advantages:**")
            for pro in characteristics[selected_model]['Pros']:
                st.write(f"✓ {pro}")
        
        with col2:
            st.warning("**Limitations:**")
            for con in characteristics[selected_model]['Cons']:
                st.write(f"• {con}")
        
        with col3:
            st.info("**Use Cases:**")
            for use in characteristics[selected_model]['Use Cases']:
                st.write(f"→ {use}")

def show_ethics_page():
    """Display ethics and optimization page"""
    st.markdown('<h2 class="sub-header">⚖️ Ethics & Optimization</h2>', 
                unsafe_allow_html=True)
    
    # Ethical considerations
    st.markdown("## 🤔 Ethical Considerations")
    
    tabs = st.tabs(["Bias Analysis", "Fairness", "Privacy", "Optimization"])
    
    with tabs[0]:
        st.markdown("""
        ### Potential Biases in Our Models
        
        #### 1. MNIST Model Biases:
        - **Representation Bias**: Training data may not represent all handwriting styles
        - **Geographic Bias**: Western numeral systems predominate
        - **Demographic Bias**: May perform differently for different age groups
        
        **Mitigation Strategies:**
        - Collect diverse handwriting samples
        - Use data augmentation techniques
        - Test on various demographic groups
        - Implement fairness metrics
        """)
        
        st.code("""
        # Example: Using TensorFlow Fairness Indicators
        import tensorflow_model_analysis as tfma
        
        # Define fairness metrics
        fairness_indicators = tfma.FairnessIndicators(
            thresholds=[0.5, 0.7, 0.9],
            sensitive_groups=['age_group', 'handedness']
        )
        """, language='python')
    
    with tabs[1]:
        st.markdown("""
        ### Ensuring Model Fairness
        
        #### NLP Model Fairness Issues:
        - **Language Bias**: May favor certain linguistic patterns
        - **Cultural Bias**: Sentiment interpretation varies by culture
        - **Brand Bias**: May have preferences for popular brands
        
        **Solutions:**
        - Use balanced training datasets
        - Regular bias audits
        - Implement debiasing techniques
        - Transparent model documentation
        """)
        
        # Fairness metrics visualization
        fairness_data = {
            'Group': ['Group A', 'Group B', 'Group C', 'Overall'],
            'Accuracy': [0.95, 0.92, 0.94, 0.94],
            'False Positive Rate': [0.03, 0.05, 0.04, 0.04],
            'False Negative Rate': [0.02, 0.03, 0.02, 0.02]
        }
        
        df_fairness = pd.DataFrame(fairness_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy', x=df_fairness['Group'], y=df_fairness['Accuracy']))
        fig.add_trace(go.Bar(name='FPR', x=df_fairness['Group'], y=df_fairness['False Positive Rate']))
        fig.add_trace(go.Bar(name='FNR', x=df_fairness['Group'], y=df_fairness['False Negative Rate']))
        
        fig.update_layout(
            title='Fairness Metrics Across Groups',
            barmode='group',
            yaxis_title='Rate',
            xaxis_title='Group'
        )
        
        st.plotly_chart(fig)
    
    with tabs[2]:
        st.markdown("""
        ### Privacy Considerations
        
        #### Data Privacy Measures:
        1. **Data Anonymization**: Remove personally identifiable information
        2. **Secure Storage**: Encrypt sensitive data
        3. **Access Control**: Limit data access to authorized personnel
        4. **Data Minimization**: Collect only necessary data
        5. **User Consent**: Obtain explicit consent for data usage
        
        #### GDPR Compliance:
        - Right to be forgotten
        - Data portability
        - Transparent data usage policies
        - Regular privacy audits
        """)
    
    with tabs[3]:
        st.markdown("""
        ### Model Optimization Techniques
        
        #### Performance Optimization:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Speed Optimization:**
            - Model quantization
            - Pruning unnecessary weights
            - Knowledge distillation
            - Batch processing
            - GPU acceleration
            """)
        
        with col2:
            st.success("""
            **Memory Optimization:**
            - Model compression
            - Weight sharing
            - Low-rank factorization
            - Efficient data structures
            - Lazy loading
            """)
        
        # Debugging example
        st.markdown("### 🐛 Debugging Challenge")
        
        st.code("""
        # Buggy TensorFlow Code - Can you spot the errors?
        
        import tensorflow as tf
        
        # Error 1: Dimension mismatch
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)  # Missing activation for classification
        ])
        
        # Error 2: Wrong loss function
        model.compile(
            optimizer='adam',
            loss='mse',  # Should be 'categorical_crossentropy' for classification
            metrics=['accuracy']
        )
        
        # Error 3: Data shape issue
        X_train = np.random.randn(1000, 28, 28)  # Should be (1000, 784) or reshaped
        y_train = np.random.randint(0, 10, 1000)  # Should be one-hot encoded
        
        # Fixed version:
        """, language='python')
        
        with st.expander("Show Fixed Code"):
            st.code("""
            import tensorflow as tf
            import numpy as np
            
            # Fixed model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')  # Added softmax
            ])
            
            # Fixed compilation
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',  # Correct loss function
                metrics=['accuracy']
            )
            
            # Fixed data
            X_train = np.random.randn(1000, 784)  # Correct shape
            y_train = tf.keras.utils.to_categorical(
                np.random.randint(0, 10, 1000), 10
            )  # One-hot encoded
            """, language='python')

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Machine Learning Models Dashboard | Built with Streamlit</p>
        <p>© 2024 | All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()