# Ilangin mean test, bisa sampe display churn vs not

# Ver 1/7 with XGBoost Fixed Hyperparameters

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import xgboost as xgb

st.set_page_config(page_title="Model Prediksi Churn Bank X", page_icon="Logo-Bank.png")

st.image("Logo-Bank.png", width=100)
st.title("Model Prediksi Churn Bank X")

st.write("### Langkah Penggunaan:")
st.write("##### A. Jika belum memiliki file model:")
st.write("####### 1. Siapkan file .xlsx data historis")
st.write("####### 2. Klik tab Model Training dan upload file .xlsx data historis")
st.write("####### 3. Tunggu hingga proses selesai dan download file model yang dihasilkan")
st.write("####### 4. Tunggu hingga proses selesai dan download file model yang dihasilkan")
st.write("####### ")
st.write("##### B. Jika sudah memiliki file model:")
st.write("####### 1. Upload file model .pkl")
st.write("####### 2. Upload file .xlsx data yang ingin diprediksi")
st.write("####### 3. Tunggu hingga proses selesai")
st.write("####### 4. Hasil prediksi churn dihasilkan")
