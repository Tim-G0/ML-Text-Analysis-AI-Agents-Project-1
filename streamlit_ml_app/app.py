# STREAMLIT ML CLASSIFICATION APP - DUAL MODEL SUPPORT
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import pypdf
import docx

#######
#     ADDITIONAL FUNCTION
#######
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
STOPWORDS = stopwords.words('english')
nltk.download('wordnet')  # WordNet for lemmatization
# st.bar_chart(probab_bar,color=["#410101"])


def clean_text(input_data):

  #-------lowercase-------------------------
  input_data_lowered = pd.Series(input_data.copy())
  for i in range(0,len(input_data_lowered)):
    input_data_lowered.iat[i] = input_data.iat[i].lower()
  #------punctuation--------------------
  REMOVELIST = string.punctuation
  REMOVELIST = REMOVELIST + "\n" #remove newline characters as well
  input_data_nopunc = pd.Series(input_data_lowered.copy())
  nopunc=''
  for i in range(0,len(input_data_nopunc)):
    for char in REMOVELIST:
      input_data_nopunc.iat[i] = input_data_nopunc.iat[i].replace(char,'')
  # cleaned_data = pd.Series(lemmatizer(input_data.copy())) # working
  # ---------stopwords----------------
  input_data_stop = pd.Series(input_data_nopunc.copy())
  for i in range(len(input_data_stop)):
    essay_arr = input_data_stop.iat[i].split()
    # print(essay_arr)
    for word in essay_arr:
      if word in STOPWORDS:
        essay_arr.remove(word)
    input_data_stop.iat[i] = ' '.join(essay_arr)

  #---------lemmatization--------------
  mylemmatizer = WordNetLemmatizer()
  input_data_lemma = pd.Series(input_data_stop.copy())

  for i in range(len(input_data_lemma)):
    input_data_lemma.iat[i] = ' '.join([mylemmatizer.lemmatize(word) for word in input_data_lemma.iat[i].split()])


  cleaned_data = pd.Series(input_data_lemma.copy())
  for i in range(0,len(cleaned_data)):
    # removing common patterns

    pattern = re.compile(r'‚äôt')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
    pattern = re.compile(r'‚Äôt')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
    pattern = re.compile(r'‚Äî')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
    pattern = re.compile(r'‚äî')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])

    pattern = re.compile(r'äôt')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
    pattern = re.compile(r'Äôt')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
    pattern = re.compile(r'\d')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
    pattern = re.compile(r'’')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
# äî
    pattern = re.compile(r'Äî')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
    pattern = re.compile(r'äî')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
# ¬¥
    pattern = re.compile(r'¬¥')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
# ‚äôs
    pattern = re.compile(r'‚äôs')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
# ‚äô
    pattern = re.compile(r'‚äô')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])

# äé
    pattern = re.compile(r'äé')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
# ,äé
    pattern = re.compile(r',äé')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
# äù
    pattern = re.compile(r'äù')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
# ,äù
    pattern = re.compile(r',äù')
    cleaned_data.iat[i] = re.sub(pattern,'',cleaned_data.iat[i])
  return cleaned_data
def to_series(input_data):
#   print(f"input_data = {input_data}")
#   st.error(f"series_data:{input_data}")

  series_data = pd.Series(input_data)
#   st.error(f"series_data:{series_data}")
  return series_data

# Page Configuration
st.set_page_config(
    page_title="ML Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
#  .st-emotion-cache-zy6yx3.en45cdb4
# 410101
st.markdown(
    """
    <style>
    .stMainBlockContainer.block-container {
    background-color: #202020;
}
    .stSidebar {
    background-color: #410101;
    }
    
   </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS
st.markdown("""
<style>
            
    .main-header {
        font-size: 2.5rem;
        color: #AC0909;
        background: url(.\painting-3135875_1280.jpg)
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #AC0909;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================

@st.cache_resource
def load_models():
    models = {}

    # print("LOADING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    try:
        try:
        # Load the main pipeline (DT)

            models['DT_Pipeline'] = joblib.load('models/DT_PIPELINE.pkl')
            models['DT_pipeline_available'] = True
        except FileNotFoundError as e:
            models['DT_pipeline_available'] = False
        # Load the main pipeline (SVC)
        try:
            models['SVC_Pipeline'] = joblib.load('models/SVC_PIPELINE.pkl')
            models['pipeline_available'] = True
        except FileNotFoundError:
            models['pipeline_available'] = False
        try:
            models['ABC_Pipeline'] = joblib.load('models/ABC_PIPELINE.pkl')
            models['ABC_pipeline_available'] = True
        except FileNotFoundError:
            models['ABC_pipeline_available'] = False

        # Check if at least one complete setup is available
        pipeline_ready = models['pipeline_available']
        dt_pipeline_ready = models['DT_pipeline_available']
        abc_pipeline_ready = models['ABC_pipeline_available']
        # print(models)
    
        # later: or individual_ready
        if not (pipeline_ready or dt_pipeline_ready or abc_pipeline_ready):
            st.error("Error loading models!")
            return None
        
        return models
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(text, model_choice, models):
    
    """Make prediction using the selected model"""
    # print(f"Text: {text} Model: {model_choice} Models: {model_choice}")
    if models is None:
        return None, None
    
    try:
        prediction = None
        probabilities = None
        
        if model_choice == "Decision Tree" and models.get('DT_pipeline_available'):
            prediction = models['DT_Pipeline'].predict(to_series(text))[0]
            probabilities = models['DT_Pipeline'].predict_proba(to_series(text))[0]

        elif model_choice == "Support Vector Classifier":
            if models.get('pipeline_available'):
                compatible_text = to_series(text)
                prediction = models["SVC_Pipeline"].predict(compatible_text)[0]
                probabilities = models["SVC_Pipeline"].predict_proba(compatible_text)[0]
        elif model_choice == "AdaBoost Classifier" and models.get('ABC_pipeline_available'):
                compatible_text = to_series(text)
                prediction = models["ABC_Pipeline"].predict(compatible_text)[0]
                probabilities = models["ABC_Pipeline"].predict_proba(compatible_text)[0]
                
        if prediction is not None and probabilities is not None:
            # Convert to readable format
            class_names = ['Human', 'AI']
            prediction_label = class_names[prediction]
            return prediction_label, probabilities
        else:
            return None, None
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Model choice: {model_choice}")
        st.error(f"Available models: {[k for k, v in models.items() if isinstance(v, bool) and v]}")
        return None, None

def get_available_models(models):
    """Get list of available models for selection"""
    available = []
    
    if models is None:
        return available
    
    if models.get('pipeline_available'):
        available.append(("Support Vector Classifier", "📈 Support Vector Classifier (Pipeline)"))
    if models.get('DT_pipeline_available'):
        available.append(("Decision Tree","🌳 Decision Tree (Pipeline)"))
    if models.get('ABC_pipeline_available'):
        available.append(("AdaBoost Classifier","🔤 AdaBoost Classifier (Pipeline)"))  
    return available

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Select an option!")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Prediction", "📁 Batch Processing", "⚖️ Model Comparison", "📊 Model Statistics","❓ Help"]
    )
# st.sidebar.
# Load models
models = load_models()
# print(f"Here {models}")

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 ML Text Classification App</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to your machine learning web application! This app demonstrates sentiment analysis
    using multiple trained models: **Support Vector Classifier** and **Decision Tree Classifier**.
    """)
    
    # App overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        ### 🔮 Single Prediction
        - Enter text manually
        - Choose between models
        - Get instant predictions
        - See confidence scores
        """)
    
    with col2:
        st.markdown("""
        ### 📁 Batch Processing
        - Upload text files
        - Process multiple texts
        - Compare model performance
        - Download results
        """)
    
    with col3:
        st.markdown("""
        ### ⚖️ Model Comparison
        - Compare all models 
        - Side-by-side
        """)
    with col4:
          st.markdown("""
        ### 📊 Model Statistics:
        - Feature Importance
        - Text Statistics
        - Validation Accuracy
        """)
    # Model status
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if models.get('pipeline_available'):
                st.info("**📈 Support Vector Classifier**\n✅ Pipeline Available")
            else:
                st.warning("**📈 Support Vector Classifier Pipeline**\n❌ Not Available")
        
        with col2:
            if models.get('DT_Pipeline'):
                st.info("**🌳 Binary Tree Classifier NB**\n✅ Available")
            else:
                st.warning("**🌳 Binary Tree Classifier**\n❌ Not Available")
        
        with col3:
            if models.get('ABC_Pipeline'):
                st.info("**🔤 AdaBoost Classifier**  \n ✅ Available")
            else:
                st.warning("**🔤 AdaBoost Classifier**\n❌ Not Available")
        
    else:
        st.error("❌ Models not loaded. Please check model files.")

# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================

elif page == "🔮 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    st.markdown("Enter text below and select a model to get a prediction whether the text is AI generated or Human.")
    model_results = ""
    if models:
        available_models = get_available_models(models)
        # print(available_models)
        if available_models:
            # Model selection
            model_choice = st.selectbox(
                "Choose a model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
            )
            # Text input
            user_input = st.text_area(
                "Enter your text here:",
                placeholder="Type or paste your text here (e.g., product review, feedback, comment)...",
                height=150
            )
            
            # Character count
            if user_input:
                st.caption(f"Character count: {len(user_input)} | Word count: {len(user_input.split())}")
            
            # Use session state for user input
            if 'user_input' in st.session_state:
                user_input = st.session_state.user_input
            
            # Prediction button
            if st.button("🚀 Predict", type="primary"):
                if user_input.strip():
                    with st.spinner('Analyzing text...'):
                        prediction, probabilities = make_prediction(user_input, model_choice, models)
                        
                        if prediction and probabilities is not None:
                            # Display prediction
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                if prediction == "Human Sentiment":
                                    st.success(f"🎯 Prediction: **{prediction} generated 🤖**")
                                else:
                                    st.info(f"🎯 Prediction: **{prediction} generated 😀**")
                            
                            with col2:
                                confidence = max(probabilities)
                                st.metric("Confidence", f"{confidence:.1%}")
                            
                            # Create probability chart
                            st.subheader("📊 Prediction Probabilities")
                            
                            # Detailed probabilities
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("😀 Human", f"{probabilities[0]:.1%}")
                            with col2:
                                st.metric("🤖 AI", f"{probabilities[1]:.1%}")
                            
                            # Bar chart
                            class_names = ['Human', 'AI']
                            prob_df = pd.DataFrame({
                                'Sentiment': class_names,
                                'Probability': probabilities
                            })
                            st.bar_chart(prob_df.set_index('Sentiment'), height=300)
                            
                        else:
                            st.error("Failed to make prediction")
                    model_results = f"""
Report:
Model: {model_choice}
Input Text: {user_input}
Confidence: {confidence:.1%},
Probabilities:
---------------
Human: {probabilities[0]}
AI:    {probabilities[1]}
Prediction: {prediction}"""
                    st.download_button(label="Download Prediction Report",data=model_results,icon="🔥",file_name="model_report.txt")

                else:
                    st.warning("Please enter some text to classify!")

        else:
            st.error("No models available for prediction.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# BATCH PROCESSING PAGE
# ============================================================================

elif page == "📁 Batch Processing":
    st.header("📁 Upload File for Batch Processing")
    st.markdown("Upload a text file to process multiple texts at once.")
    
    if models:
        model_results = ""
        user_input_txt = ""
        available_models = get_available_models(models)
        
        if available_models:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf','docx'],
                help="Upload a .docx or .pdf"
            )
            if uploaded_file:

                # Model selection
                model_choice = st.selectbox(
                    "Choose model for batch processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
                )
                
                # Process file
                if st.button("📊 Process File"):
                    try:
                        if ".pdf" in uploaded_file.name:
                            pdf_data = None
                            pdf_pagelist = []
                            with open("tmp.pdf","wb") as file:
                                file.write(uploaded_file.read())
                                pdf_data = pypdf.PdfReader("tmp.pdf")
                                file.close()
                            pages = pdf_data.pages
                            for page in pages:
                                pdf_pagelist.append(format(page.extract_text()+"\n"))
                            # print(pdf_pagelist)
                            pdf_data = ''.join([page for page in pdf_pagelist])
                            user_input_txt= pdf_data
                            prediction, probabilities = make_prediction(text=pdf_data,model_choice=model_choice,models=models)
                            
                        elif ".docx" in uploaded_file.name:
                            docx_data = None
                            paragph_list = []
                            with open("tmp.docx","wb") as file:
                                file.write(uploaded_file.read())
                                docx_data = docx.Document(docx="tmp.docx")
                                file.close()
                                for paragraph in docx_data.paragraphs:
                                    paragph_list.append(format(paragraph.text+"\n"))
                                docx_data = ''.join([para for para in paragph_list])
                                user_input_txt= docx_data
                                
                                prediction, probabilities = make_prediction(text=docx_data,model_choice=model_choice,models=models)
                        else:
                            st.error("Please select a Docx, PDF!")
                    except Exception as e:
                        print(e)
                    st.subheader("📊 Prediction Probabilities")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("😀 Human", f"{probabilities[0]:.1%}")
                    with col2:
                        st.metric("🤖 AI", f"{probabilities[1]:.1%}")
                    probab_bar = {"Human":probabilities[0],"AI":probabilities[1]}
                    st.bar_chart(probab_bar,color = "#ffaa00")
                    confidence = max(probabilities)
                    st.metric("Confidence", f"{confidence:.1%}")
                    model_results = f"""
Report:
Model: {model_choice}
Input Text: {user_input_txt}
Confidence: {confidence:.1%},
Probabilities:
---------------
Human: {probabilities[0]}
AI:    {probabilities[1]}
Prediction: {prediction}"""
                    st.download_button(label="Download Prediction Report",data=model_results,icon="🔥",file_name="model_report.txt")

            else:
                st.info("Please upload a file to get started.")
                
                # Show example file formats
                with st.expander("📄 List of Supported File Formats"):
                    st.markdown("""
                    **Docx (.docx)**
                    **PDF  (.pdf)**                                
                    """)
        else:
            st.error("No models available for batch processing.")

    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================

elif page == "⚖️ Model Comparison":
    st.header("⚖️ Compare Models")
    st.markdown("""Compare predictions from different models on the same text.""")
    if models:
        available_models = get_available_models(models)
        if len(available_models) >= 2:
            # Text input for comparison
            comparison_text = st.text_area(
                "Enter text to compare models:",
                placeholder="Enter text to see how different models perform...",
                height=100
            )
            
            if st.button("📊 Compare All Models") and comparison_text.strip():
                st.subheader("🔍 Model Comparison Results")
                
                # Get predictions from all available models
                comparison_results = []
                
                for model_key, model_name in available_models:
                    prediction, probabilities = make_prediction(comparison_text, model_key, models)
                    
                    if prediction and probabilities is not None:
                        comparison_results.append({
                            'Model': model_name,
                            'Prediction': prediction,
                            'Confidence': f"{max(probabilities):.1%}",
                            'Human %': f"{probabilities[0]:.1%}",
                            'AI %': f"{probabilities[1]:.1%}",
                            'Raw_Probs': probabilities
                        })
                
                if comparison_results:
                    # Comparison table
                    comparison_df = pd.DataFrame(comparison_results)
                    st.table(comparison_df[['Model', 'Prediction', 'Confidence', 'Human %', 'AI %']])
                    
                    # Agreement analysis
                    predictions = [r['Prediction'] for r in comparison_results]
                    if len(set(predictions)) == 1:
                        st.success(f"✅ All models agree: **{predictions[0]} Sentiment**")
                    else:
                        st.warning("⚠️ Models disagree on prediction")
                        for result in comparison_results:
                            model_name = result['Model'].split(' ')[1] if ' ' in result['Model'] else result['Model']
                            st.write(f"- {model_name}: {result['Prediction']}")
                    
                    # Side-by-side probability charts
                    st.subheader("📊 Detailed Probability Comparison")
                    
                    cols = st.columns(len(comparison_results))
                    
                    for i, result in enumerate(comparison_results):
                        with cols[i]:
                            model_name = result['Model']
                            st.write(f"**{model_name}**")
                            
                            chart_data = pd.DataFrame({
                                'Sentiment': ['Human', 'AI'],
                                'Probability': result['Raw_Probs']
                            })
                            st.bar_chart(chart_data.set_index('Sentiment'))
                    
                else:
                    st.error("Failed to get predictions from models")
        
        elif len(available_models) == 1:
            st.info("Only one model available. Use Single Prediction page for detailed analysis.")
            
        else:
            st.error("No models available for comparison.")
    else:
        st.warning("Models not loaded. Please check the model files.")
            

                
# ============================================================================
# MODEL INFO PAGE
# ============================================================================

elif page == "📊 Model Statistics":
    st.header("📊 Model Statistics")
    
    if models:
        st.success("✅ Models are loaded and ready!")
        
        # Model details
        st.subheader("🔧 Available Models")
        
        col1, col2 = st.columns(2,border=True)
        
        with col1:
            st.markdown("""
            ### 📈 Support Vector Machine
            ---
            - **Type:** Support Vector Classifier
            - **Features:** TF-IDF vectors (unigrams + bigrams)
            - **Model:**    Support Vector Classifier   
            ## SVC Feature Importance Graph:         
            """)
            st.image("./SVC Importance.png")
            st.markdown("""
            ## 📈 TF-IDF for SVC Model: 
            - **Number of Features:** 30502 features
            - **N-Grams** Unigrams and Bigrams
            - **Minimum Document Frequency:**    2
            ## 📈 SVC Model Parameters:  
            - **Kernel:** linear
            - **C**: 10
            - **Gamma**: scale
            ---   
            🎯 Validation Accuracy for SVC: 98.79%   
            
            Validation Performance Statistics
            \n
            Confusion Matrix:**[**[367,6],[3,37]**]**\n
            ### Classification Report (SVC)\n
            ---
            Human:
            - Precision: 0.99
            - Recall: 0.98 
            - F1-Score: 0.99 
            - Support: 373 
                         
            AI: 
            - Precision: 0.98 
            - Recall: 0.99 
            - F1-Score: 0.99 
            - Support: 373          
            """)
            svc_stats_report = """
            ### 📈 Support Vector Machine
            - **Type:** Support Vector Classifier
            - **Features:** TF-IDF vectors (unigrams + bigrams)
            - **Model:**    Support Vector Classifier   
            ## SVC Feature Importance Graph: 

            ## 📈 TF-IDF for SVC Model: 
            - **Number of Features:** 30502 features
            - **N-Grams** Unigrams and Bigrams
            - **Minimum Document Frequency:**    2
            ## 📈 SVC Model Parameters:  
            - **Kernel:** linear
            - **C**: 10
            - **Gamma**: scale
            ---   
          
                                """

            
        with col2:
            st.markdown("""
            ### 🎯 Decision Tree
            ---
            **Type:** Decision Tree Classificatier                        
            **Features:** TF-IDF vectors (Unigrams + Tri-grams)
            ## Decision Tree Feature Importance Graph:
                       

            """)
            st.image("./dt_feature_imptce.png")
            st.markdown("""
            ## 📈 TF-IDF for Decision Tree Model: 
            - **Number of Features:** 30000 features
            - **N-Grams** Unigrams and Tri-grams 
            ## 📈 Decision Tree Model Parameters:  
            - **Max Depth**: 10
            - **Criterion**: gini
            - **Minimum Leaf Samples**: 1
            - **Minimum Sample Splits**:  2
                        
            --- 
            🎯 Accuracy for Decision Tree: 87.27%             
              Validation Performance Statistics
            \n
            Decision Tree Accuracy: 87.27%\n
            Confusion Matrix:**[**[334,39],[51,322]**]**\n
            ### Classification Report (Tree)\n
            ---
            Human:
            - Precision: 0.86
            - Recall: 0.89 
            - F1-Score: 0.87 
            - Support: 373 
                         
            AI: 
            - Precision: 0.89 
            - Recall: 0.86 
            - F1-Score: 0.87 
            - Support: 373
            """)
        container = st.container(border=True)
        with container:
            st.markdown("""
            ### 🎯 AdaBoost Classifier
            ---
            - **Type:** AdaBoost Classifier                        
            - **Estimator:** Support Vector Machine
            - **Learning Rate:** 1
            - **Number of Estimators:** 2
            - *Hyperparemeter Tuning not done
            due to computational and time constraints.*
            Validation Performance Statistics
            \n
            🎯 AdaBoost Accuracy: 89.01\n
            Confusion Matrix:**[**[293,80],[2,371]**]**\n
            ### Classification Report (ABC)\n
            ---
            Human:
            - Precision: 0.99
            - Recall: 0.79 
            - F1-Score: 0.99 
            - Support: 373 
                         
            AI: 
            - Precision: 0.82 
            - Recall: 0.99 
            - F1-Score: 0.90 
            - Support: 373
                  
           """)
#############

#               Classification Report (ABC)
#               precision    recall  f1-score   support

#        Human       0.99      0.79      0.88       373
#           AI       0.82      0.99      0.90       373

#     accuracy                           0.89       746
#    macro avg       0.91      0.89      0.89       746
# weighted avg       0.91      0.89      0.89       746
            
            
#############


            # \dt_feature_imptce.png
        model_stats_report = """
            📈 Support Vector Machine
            - Type: Support Vector Classifier
            - Features: TF-IDF vectors (unigrams + bigrams)
            - Model:    Support Vector Classifier   

            ## 📈 TF-IDF for SVC Model: 
            - Number of Features: 30502 features
            - N-Grams Unigrams and Bigrams
            - Minimum Document Frequency:    2
            📈 SVC Model Parameters:  
            - Kernel: linear
            - C: 10
            - Gamma: scale
            -----------------------------------------------   
            🎯 Validation Accuracy for SVC: 98.79%  
            
            Validation Performance Statistics
            
            Confusion Matrix:[[367,6],[3,37]]
            Classification Report (SVC)
            
            Human:
            - Precision: 0.99
            - Recall: 0.98 
            - F1-Score: 0.99 
            - Support: 373 
                         
            AI: 
            - Precision: 0.98 
            - Recall: 0.99 
            - F1-Score: 0.99 
            - Support: 373
            -----------------------------------------------
            🎯 Decision Tree
            Type: Decision Tree Classificatier                        
            Features: TF-IDF vectors (Unigrams + Tri-grams)

            📈 TF-IDF for Decision Tree Model: 
            - Number of Features: 30000 features
            - N-Grams Unigrams and Tri-grams 
            📈 Tree Model Parameters:  
            - Max Depth: 10
            - Criterion: gini
            - Minimum Leaf Samples: 1
            - Minimum Sample Splits:  2
                        
            -----------------------------------------------
            🎯 Accuracy for Decision Tree: 87.27%  
                    Validation Performance Statistics
            
            Decision Tree Accuracy: 87.27%
            Confusion Matrix:[[332,41],[54,319]]
            Classification Report (Tree)
            -----------------------------------------------
            Human:
            - Precision: 0.86
            - Recall: 0.89 
            - F1-Score: 0.87 
            - Support: 373 
                         
            AI: 
            - Precision: 0.89 
            - Recall: 0.86 
            - F1-Score: 0.87 
            - Support: 373
            -----------------------------------------------
            🎯 AdaBoost Classifier
            -----------------------------------------------
            - Type: AdaBoost Classifier                        
            - Estimator: Support Vector Machine
            - Learning Rate: 1
            - Number of Estimators: 2
            - *Hyperparemeter Tuning not done
            due to computational and time constraints.*

            🎯 AdaBoost Accuracy: 89.01
            Validation Performance Statistics
            Confusion Matrix:[[293,80],[2,371]]
            Classification Report (ABC)
            -----------------------------------------------
            Human:
            - Precision: 0.99
            - Recall: 0.79 
            - F1-Score: 0.99 
            - Support: 373 
                         
            AI: 
            - Precision: 0.82 
            - Recall: 0.99 
            - F1-Score: 0.90 
            - Support: 373
                  
                                """
        st.download_button(label="Download Model Statistics Report",data=model_stats_report,icon="🔥",file_name="model_report.txt")

        # File status
        st.subheader("📁 Model Files Status")
        file_status = []
        
        files_to_check = [
            ("SVC_PIPELINE.pkl", "Complete Support Vector Classifier Pipeline", models.get('pipeline_available', False)),
            ("DT_PIPELINE.pkl", "Complete Decision Tree Pipeline", models.get('DT_pipeline_available', False)),
            ("ABC_PIPELINE.pkl", "Complete AdaBoost Classifier Pipeline", models.get('ABC_pipeline_available', False))

        ]
        
        for filename, description, status in files_to_check:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })
        
        st.table(pd.DataFrame(file_status))
        
        # Training information
        st.subheader("📚 Training Information")
        st.markdown("""
        **Dataset:** Collection of AI and Human created essays.
        - **Classes:** Human or AI generated
        - **Preprocessing:** Punctuation, stopword, and numeric pattern removal, Lemmatization, TFIDF-vectorization.
        - **Training:** Models use different TFIDF Vectorizers. GridsearchCV indicated different numbers of max features.
        """)
        
    else:
        st.warning("Models not loaded. Please check model files in the 'models/' directory.")

# ============================================================================
# HELP PAGE
# ============================================================================

elif page == "❓ Help":
    st.header("❓ How to Use This App")
    
    with st.expander("🔮 Single Prediction"):
        st.write("""
        1. **Select a model** from the dropdown (Support Vector, Decision Tree, or AdaBoost)
        2. **Enter text** in the text area (product reviews, comments, feedback)
        3. **Click 'Predict'** to get sentiment analysis results
        4. **View results:** prediction, confidence score, and probability breakdown
        6. ** Download the results of the prediction.**
        """)
    
    with st.expander("📁 Batch Processing"):
        st.write("""
        1. **Prepare your file:**
           - **.docx file:** Word Document
           - **.pdf file:** PDF Document
        2. **Upload the file** using the file uploader
        3. **Select a model** for processing.
        4. **Click 'Process File'** to analyze all texts.
        5. ** View whether the model thinks it is AI or Human.**
        6. ** Download the results of the prediction.**
        """)
    
    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Choose whether you want to enter text or upload a document.**
        2. **Click 'Compare All Models'** to get predictions from all three models
        3. **View comparison table** showing predictions and confidence scores
        4. **Analyze agreement:** See if models agree or disagree
        5. **Compare probabilities:** Side-by-side probability charts
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and Solutions:**
        
        **Models not loading:**
        - Ensure model files (.pkl) are in the 'models/' directory
        - Check that required files exist:
          - ABC_PIPELINE.pkl (AdaBoost)
          - DT_PIPELINE.pkl (Decision Tree)
          - SVC_PIPELINE.pkl (SVC)
        
        **Prediction errors:**
        - Make sure input text is not empty
        - Try shorter texts if getting memory errors
        - Check that text contains readable characters
        
        **File upload issues:**
        - Ensure file format is .docx or .pdf
        """)
    
    # System information
    st.subheader("💻 Your Project Structure")
    st.code("""
    streamlit_ml_app/
    ├── app.py                              # Main application
    ├── requirements.txt                    # Dependencies
    ├── models/                            # Model files
    │   ├── SVC_PIPELINE.pkl # Support Vector Classifier
    │   ├── DT_PIPELINE.pkl  # Decision Tree Classifier
    │   ├── ABC_PIPELINE.pkl # AdaBoost Classifier 
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.warning("""
**Machine Learning AI Essay Classifier App**
Built with Streamlit

**Models:** 
- 📈 Support Vector Classifier
- 🌳 Decision Tree Classifier
- 🔤 AdaBoost Classifier

**Framework:** scikit-learn
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #FFFF;'>
    Built with ❤️ using Streamlit | Machine Learning AI Essay Classifier | By Timothy<br>
    <small>A project for the course **Intro to AI Agents**</small><br>
    <small>Basic project template provided by **Professor Maaz**</small><br>
    <small>This app demonstrates text analysis using trained ML models</small>
</div>
""", unsafe_allow_html=True)