import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# App Title & Student Meta Info
st.set_page_config(page_title="Machine Learning Assignment 2", layout="wide")
st.title("Machine Learning Assignment - 2")
st.markdown("### **Student Name:** Christeen Jose | **Roll No:** 2026AK05508")
st.markdown("---")

# Dictionary mapping model options to their path files
MODEL_DIR = "model"
MODEL_MAPPING = {
    "Logistic Regression": "Logistic Regression.joblib",
    "Decision Tree": "Decision Tree.joblib",
    "KNN": "KNN.joblib",
    "Naive Bayes": "Naive Bayes.joblib",
    "Support Vector Machine": "Support Vector Machine.joblib"
}

def load_saved_model(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Error: Model file '{path}' not found.")
        return None

# Sidebar components
st.sidebar.header("Inputs & Configuration")

# a. Test Dataset upload
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV Format)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    test_df = pd.read_csv(uploaded_file)
    st.subheader("Test Dataset View")
    st.dataframe(test_df.head())
    
    # Configure Target column
    columns_list = test_df.columns.tolist()
    target_var = st.sidebar.selectbox("Select Target Column (y)", columns_list, index=len(columns_list)-1)
    
    # Slice features and target without retraining 
    X_test = test_df.drop(columns=[target_var])
    y_test = test_df[target_var]
    
    # b. Model selection box
    st.sidebar.markdown("---")
    st.sidebar.header("Model Selection")
    chosen_models = st.sidebar.multiselect(
        "Select Models to Run", 
        options=list(MODEL_MAPPING.keys()),
        default=list(MODEL_MAPPING.keys())
    )
    
    if not chosen_models:
        st.warning("Please choose one or more models from the sidebar to evaluate.")
    else:
        summary_records = []
        evaluation_details = {}
        
        # Evaluate each selected model statically
        for model_title in chosen_models:
            loaded_clf = load_saved_model(MODEL_MAPPING[model_title])
            
            if loaded_clf is None:
                continue
                
            # Perform predictions directly
            y_pred = loaded_clf.predict(X_test)
            
            # Safe calculation for AUC scores using proba or decision function
            if hasattr(loaded_clf, "predict_proba"):
                y_prob = loaded_clf.predict_proba(X_test)[:, 1]
            elif hasattr(loaded_clf, "decision_function"):
                y_prob = loaded_clf.decision_function(X_test)
            else:
                y_prob = y_pred
                
            # Compute required assignment metrics
            accuracy = accuracy_score(y_test, y_pred)
            try:
                auc_score = roc_auc_score(y_test, y_prob)
            except:
                auc_score = np.nan
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Store summary row results
            summary_records.append({
                "Model Name": model_title,
                "Accuracy": round(accuracy, 4),
                "AUC": round(auc_score, 4) if not np.isnan(auc_score) else "N/A",
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4),
                "MCC": round(mcc, 4)
            })
            
            # Save classification breakdown details
            evaluation_details[model_title] = {
                "conf_matrix": confusion_matrix(y_test, y_pred),
                "class_report": classification_report(y_test, y_pred, zero_division=0)
            }
            
        if summary_records:
            # c. Display evaluation metrics matrix table
            st.subheader("Model Performance Evaluation Matrix")
            results_table = pd.DataFrame(summary_records).set_index("Model Name")
            st.dataframe(results_table)
            
            # d. Display Confusion Matrix and Classification Reports
            st.subheader("Model Diagnostics (Classification Reports & Confusion Matrices)")
            tabs = st.tabs(list(evaluation_details.keys()))
            
            for i, model_title in enumerate(evaluation_details.keys()):
                with tabs[i]:
                    st.markdown(f"#### Reports for {model_title}")
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        st.markdown("**Classification Report Output:**")
                        st.code(evaluation_details[model_title]["class_report"])
                        
                    with col_right:
                        st.markdown("**Confusion Matrix DataFrame:**")
                        matrix_data = evaluation_details[model_title]["conf_matrix"]
                        matrix_df = pd.DataFrame(
                            matrix_data,
                            index=[f"Actual Class {cls}" for cls in range(len(matrix_data))],
                            columns=[f"Predicted Class {cls}" for cls in range(len(matrix_data))]
                        )
                        st.dataframe(matrix_df)
else:
    st.info("Awaiting input data. Please upload a test dataset CSV file using the sidebar panel to see results.")
