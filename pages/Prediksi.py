# Ilangin mean test, bisa sampe display churn vs not

# Ver 1/7 with XGBoost Fixed Hyperparameters

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.utils import resample
import xgboost as xgb
from Model_Churn_v1 import load_data, process_data  # pastikan untuk mengimpor fungsi dari file lain

st.set_page_config(page_title="Model Prediksi Churn CMS/Qlola", page_icon="Logo-White.png")

st.image("Logo-Bank-BRI.png", width=100)
st.title("Model Prediksi Churn CMS/Qlola BRI")

# Section for Customer Churn Prediction
st.write("## Prediksi Customer Churn")
uploaded_model = st.file_uploader("Upload Model File", type=["pkl"], key="model_upload")

if uploaded_model is not None:
    model = joblib.load(uploaded_model)
    st.success("Model loaded successfully!")

uploaded_data_pred = st.file_uploader("Upload Data untuk Prediksi", type=["xlsx"], key="predict_upload")

if uploaded_data_pred is not None and 'model' in locals():
    df_pred = load_data(uploaded_data_pred)
    processed_data_pred = process_data(df_pred, is_training_data=False)
    cifno = processed_data_pred['cifno'].copy()
    predictions = model.predict(processed_data_pred.drop(['cifno'], axis=1))

    hasil_prediksi = pd.DataFrame({
        'cifno': cifno,
        'prediksi': predictions
    })

    st.write("Hasil Prediksi Churn:", hasil_prediksi)

    analisis_data = pd.merge(hasil_prediksi, processed_data_pred, on='cifno', how='inner')

    st.write("## Analisis Karakteristik Nasabah Berdasarkan Status Churn")

    st.write("### Data untuk Analisis")
    st.write(analisis_data.head())

    fitur_agregat = analisis_data.groupby('prediksi').sum().T
    st.write("### Data Agregat")
    st.write(fitur_agregat)

    fitur_normalisasi = fitur_agregat.div(fitur_agregat.sum(axis=1), axis=0) * 100
    st.write("### Data Normalisasi untuk Proporsi Churn")
    st.write(fitur_normalisasi)

    fitur_normalisasi = fitur_normalisasi.sort_values(by=1, ascending=True)
    fitur_normalisasi.drop(['Other', 'ratas_trx_january', 'ratas_trx_february', 'ratas_trx_march', 'ratas_trx_april', 'ratas_trx_may', 'ratas_trx_june', 'ratas_trx_july', 'vol_trx_january', 'vol_trx_february', 'vol_trx_march', 'vol_trx_april', 'vol_trx_may', 'vol_trx_june', 'vol_trx_july', 'frek_trx_january', 'frek_trx_februari', 'frek_trx_march', 'frek_trx_april', 'frek_trx_may', 'frek_trx_june', 'frek_trx_july'], axis=0, inplace=True)

    fig, ax = plt.subplots(figsize=(15, 10))
    fitur_normalisasi.plot(kind='barh', stacked=True, colormap='viridis', ax=ax)
    ax.set_title('Komposisi Status Churn 100% untuk Setiap Fitur')
    ax.set_xlabel('Proporsi (%)')
    ax.set_ylabel('Fitur')
    ax.legend(title='Status Churn', labels=['0', '1'])
    st.pyplot(fig)

    fitur_persentase = fitur_normalisasi.applymap(lambda x: f"{x:.2f}%")
    st.dataframe(fitur_persentase)

    volume_columns = [col for col in analisis_data.columns if 'vol_trx_' in col]
    frequency_columns = [col for col in analisis_data.columns if 'frek_trx_' in col]

    analisis_data['average_volume'] = analisis_data[volume_columns].mean(axis=1)
    analisis_data['total_frequency'] = analisis_data[frequency_columns].sum(axis=1)

    analisis_data['profitability_score'] = analisis_data['average_volume'] * analisis_data['total_frequency']

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(analisis_data['profitability_score'], analisis_data['prediksi'], c=analisis_data['prediksi'], cmap='viridis', alpha=0.6)
    ax.set_title('Profitability Score vs Churn Prediction')
    ax.set_xlabel('Profitability Score')
    ax.set_ylabel('Churn Prediction (0=Low, 1=High)')
    ax.grid(True)
    fig.colorbar(scatter, label='Churn Prediction')
    st.pyplot(fig)

    filtered_data = analisis_data.query('prediksi == 1')
    sorted_filtered_data = filtered_data.sort_values(by='profitability_score', ascending=False)

    kolom_tabel = ['average_volume', 'total_frequency', 'profitability_score', 'prediksi']
    tabel_urut_profitabilitas = sorted_filtered_data[kolom_tabel].copy()

    st.write("### Tabel Nasabah dengan Profitability Score")
    st.dataframe(tabel_urut_profitabilitas)

    analisis_data_dropped = analisis_data.drop(['Other', 'ratas_trx_january', 'ratas_trx_february', 'ratas_trx_march', 'ratas_trx_april', 'ratas_trx_may', 'ratas_trx_june', 'ratas_trx_july', 'vol_trx_january', 'vol_trx_februari', 'vol_trx_march', 'vol_trx_april', 'vol_trx_may', 'vol_trx_june', 'vol_trx_july', 'frek_trx_january', 'frek_trx_februari', 'frek_trx_march', 'frek_trx_april', 'frek_trx_may', 'frek_trx_june', 'frek_trx_july'], axis=1)

    correlation_matrix = analisis_data_dropped.corr()

    st.write("### Correlation Heatmap antar Fitur")
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5}, ax=ax)
    ax.set_title('Heatmap Korelasi Fitur')
    st.pyplot(fig)
