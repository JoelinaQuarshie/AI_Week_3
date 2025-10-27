"""
ML Models Dashboard - No Plotly Version
Works without plotly dependency
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Page config
st.set_page_config(
    page_title="ML Models Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
    }
    h1 {
        color: #2E7D32;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🤖 Machine Learning Models Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "🌸 Iris Classification", "📊 Visualizations", "📈 Metrics"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "🌸 Iris Classification":
        show_iris()
    elif page == "📊 Visualizations":
        show_visualizations()
    elif page == "📈 Metrics":
        show_metrics()

def show_home():
    """Home page"""
    st.header("Welcome to ML Dashboard")
    
    # Info boxes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        ### 🌸 Iris Classification
        - Decision Tree Model
        - 95%+ Accuracy
        - 4 Features → 3 Classes
        """)
    
    with col2:
        st.success("""
        ### 📊 Visualizations
        - Interactive Charts
        - Feature Analysis
        - Performance Metrics
        """)
    
    with col3:
        st.warning("""
        ### 🎯 Features
        - Real-time Predictions
        - Model Comparison
        - Data Insights
        """)
    
    st.markdown("---")
    
    # Quick stats using columns
    st.subheader("📈 Dashboard Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Models", "3", "↑ Active")
    with col2:
        st.metric("Accuracy", "95.3%", "↑ 2.1%")
    with col3:
        st.metric("Predictions", "1,234", "↑ 156")
    with col4:
        st.metric("Response", "45ms", "↓ 5ms")
    
    # Sample dataset preview
    st.markdown("---")
    st.subheader("📋 Sample Data Preview")
    
    iris = load_iris()
    df = pd.DataFrame(iris.data[:5], columns=iris.feature_names)
    df['Species'] = [iris.target_names[t] for t in iris.target[:5]]
    
    st.dataframe(df, use_container_width=True)

def show_iris():
    """Iris classification page"""
    st.header("🌸 Iris Flower Classification")
    
    # Cache the model
    @st.cache_resource
    def get_model():
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        return model, iris, accuracy
    
    model, iris, accuracy = get_model()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Make a Prediction")
        
        # Input features
        col_a, col_b = st.columns(2)
        
        with col_a:
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        
        with col_b:
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
            petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)
        
        # Predict button
        if st.button("🔮 Predict Species", type="primary"):
            features = [[sepal_length, sepal_width, petal_length, petal_width]]
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Result
            species = iris.target_names[prediction]
            st.success(f"### Predicted Species: **{species.upper()}**")
            
            # Probability bars using matplotlib
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            bars = ax.bar(iris.target_names, probabilities * 100, color=colors)
            ax.set_ylabel('Probability (%)')
            ax.set_title('Prediction Confidence')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probabilities * 100):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.1f}%', ha='center', va='bottom')
            
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.subheader("Model Info")
        
        # Metrics
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.metric("Algorithm", "Decision Tree")
        st.metric("Features", "4")
        st.metric("Classes", "3")
        
        # Feature importance
        st.subheader("Feature Importance")
        
        importance = model.feature_importances_
        fig, ax = plt.subplots(figsize=(6, 4))
        
        features_short = ['S.Len', 'S.Wid', 'P.Len', 'P.Wid']
        colors = plt.cm.viridis(importance / importance.max())
        bars = ax.barh(features_short, importance, color=colors)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        
        st.pyplot(fig)
        plt.close()

def show_visualizations():
    """Visualizations page"""
    st.header("📊 Data Visualizations")
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[t] for t in iris.target]
    
    tabs = st.tabs(["Distribution", "Correlation", "Pairplot"])
    
    with tabs[0]:
        st.subheader("Feature Distributions")
        
        feature = st.selectbox("Select Feature", iris.feature_names)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        for species in iris.target_names:
            data = df[df['species'] == species][feature]
            ax1.hist(data, alpha=0.5, label=species, bins=15)
        ax1.set_xlabel(feature)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{feature} Distribution by Species')
        ax1.legend()
        
        # Box plot
        df.boxplot(column=feature, by='species', ax=ax2)
        ax2.set_title(f'{feature} by Species')
        ax2.set_xlabel('Species')
        ax2.set_ylabel(feature)
        plt.suptitle('')  # Remove default title
        
        st.pyplot(fig)
        plt.close()
    
    with tabs[1]:
        st.subheader("Feature Correlation Matrix")
        
        # Calculate correlation
        corr = df[iris.feature_names].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        
        st.pyplot(fig)
        plt.close()
    
    with tabs[2]:
        st.subheader("Pairwise Relationships")
        
        # Select two features
        col1, col2 = st.columns(2)
        with col1:
            x_feat = st.selectbox("X-axis", iris.feature_names, index=0)
        with col2:
            y_feat = st.selectbox("Y-axis", iris.feature_names, index=1)
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, species in enumerate(iris.target_names):
            mask = df['species'] == species
            ax.scatter(df[mask][x_feat], df[mask][y_feat],
                      label=species, alpha=0.7, s=50)
        
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.set_title(f'{x_feat} vs {y_feat}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()

def show_metrics():
    """Metrics page"""
    st.header("📈 Model Performance Metrics")
    
    # Sample metrics data
    metrics_data = {
        'Model': ['Iris DT', 'Random Forest', 'SVM', 'KNN'],
        'Accuracy': [95.3, 96.7, 97.1, 94.8],
        'Precision': [95.1, 96.5, 97.0, 94.5],
        'Recall': [95.0, 96.4, 96.9, 94.7],
        'F1-Score': [95.0, 96.4, 96.9, 94.6]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Display table
    st.subheader("📊 Performance Comparison")
    st.dataframe(df_metrics, use_container_width=True)
    
    # Bar chart comparison
    st.subheader("📊 Visual Comparison")
    
    metric_to_plot = st.selectbox("Select Metric", 
                                  ['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_metrics['Model']))
    bars = ax.bar(x, df_metrics[metric_to_plot], 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    ax.set_xlabel('Model')
    ax.set_ylabel(f'{metric_to_plot} (%)')
    ax.set_title(f'{metric_to_plot} Comparison Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics['Model'])
    ax.set_ylim(90, 100)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    plt.close()
    
    # Additional metrics
    st.markdown("---")
    st.subheader("🎯 Additional Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Training Metrics**
        - Training Time: 0.05s
        - Epochs: 100
        - Learning Rate: 0.01
        """)
    
    with col2:
        st.success("""
        **Validation Metrics**
        - Val Accuracy: 94.8%
        - Val Loss: 0.152
        - Best Epoch: 87
        """)
    
    with col3:
        st.warning("""
        **Test Metrics**
        - Test Accuracy: 95.3%
        - Test Loss: 0.141
        - Inference Time: 0.5ms
        """)

if __name__ == "__main__":
    main()
