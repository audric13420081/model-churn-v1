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

st.set_page_config(page_title="Model Prediksi Churn Bank X", page_icon="Logo-Bank.png")

st.image("Logo-Predict.png", width=100)

# Function to load data
def load_data(file):
    return pd.read_excel(file)

# Function to process data
def process_data(df):
    st.write("Starting data processing...")

    # Seleksi variabel data awal
    selected_columns = [
        'cifno', 'nama_nasabah', 'segmentasi_bpr', 'rgdesc', 'Platform_Channel',
        'Giro Type', 'Loan Type', 'Balance Giro', 'Ratas Giro', 'total_amount_transaksi',
        'bulan_transaksi', 'segmen', 'freq_transaksi'
    ]
    df = df[selected_columns]

    # Pembersihan Data: Menghapus duplikasi
    df = df.drop_duplicates()

    # Identifikasi missing data
    missing_data_summary = df.isnull().sum()
    st.write("Missing Data Summary:")
    st.write(missing_data_summary)

    # Menangani missing data
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna('Unknown', inplace=True)
        else:
            df[col].fillna(0, inplace=True)

    # Transformasi Variabel: Grup untuk Giro Type dan Loan Type
    def giro_type_group(giro_type):
        if pd.isna(giro_type):
            return 'Other'
        elif giro_type.startswith('G'):
            return 'Giro Type G'
        elif giro_type.startswith('H'):
            return 'Giro Type H'
        else:
            return 'Other'

    def loan_type_group(loan_type):
        if pd.isna(loan_type) or loan_type == '(blank)':
            return 'Blank/Other'
        elif any(term in loan_type for term in ['Komersial', 'Kecil', 'KUPEDES']):
            return 'Loan Type: Ritel & Kecil'
        elif any term in loan_type for term in ['Menengah', 'MNGH', 'DIV BUMN']):
            return 'Loan Type: Menengah & Besar'
        elif any term in loan_type for term in ['VALAS', 'CASH', 'FPJP', 'VLS']):
            return 'Loan Type: Valas & Fasilitas Khusus'
        elif any term in loan_type for term in ['DKM', 'KREDIT', 'Kredit', 'Program']):
            return 'Loan Type: Kredit Spesial & Program'
        else:
            return 'Loan Type: Lainnya'

    df['Giro Type Group'] = df['Giro Type'].apply(giro_type_group)
    df['Loan Type Group'] = df['Loan Type'].apply(loan_type_group)

    # One-hot Encoding: Pembuatan variabel dummy
    giro_type_group_dummies = pd.get_dummies(df['Giro Type Group'])
    loan_type_group_dummies = pd.get_dummies(df['Loan Type Group'])
    segmen_dummies = pd.get_dummies(df['segmentasi_bpr'])
    rgdesc_dummies = pd.get_dummies(df['rgdesc'])
    platform_channel_dummies = pd.get_dummies(df['Platform_Channel'])

    # Normalisasi Data: Pengelompokan data berdasarkan cifno
    segmen_agg = segmen_dummies.groupby(df['cifno']).max()
    rgdesc_agg = rgdesc_dummies.groupby(df['cifno']).max()
    platform_channel_agg = platform_channel_dummies.groupby(df['cifno']).max()
    giro_type_group_agg = giro_type_group_dummies.groupby(df['cifno']).max()
    loan_type_group_agg = loan_type_group_dummies.groupby(df['cifno']).max()

    df_final = pd.concat([segmen_agg, rgdesc_agg, platform_channel_agg, giro_type_group_agg, loan_type_group_agg], axis=1)
    df_final.reset_index(inplace=True)
    df_final = pd.merge(df_final, df.drop_duplicates('cifno'), on='cifno', how='inner')
    df_final = df_final.drop(['segmentasi_bpr', 'rgdesc', 'Platform_Channel', 'Giro Type', 'Loan Type', 'Giro Type Group', 'Loan Type Group'], axis=1, errors='ignore')

    # Transformasi Variabel (lanjutan): Konversi tanggal dan nilai numerik
    df['bulan_transaksi'] = pd.to_datetime(df['bulan_transaksi'], format='%B')
    df.sort_values('bulan_transaksi', inplace=True)
    df['Ratas Giro'] = pd.to_numeric(df['Ratas Giro'], errors='coerce')
    df['total_amount_transaksi'] = pd.to_numeric(df['total_amount_transaksi'], errors='coerce')
    df['freq_transaksi'] = pd.to_numeric(df['freq_transaksi'], errors='coerce')

    # Transformasi Data Lainnya: Pembuatan pivot table untuk transaksi
    pivot_freq = df.pivot_table(index='cifno',
                                columns=df['bulan_transaksi'].dt.month_name(),
                                values='freq_transaksi',
                                aggfunc='sum',
                                fill_value=0)

    pivot_vol = df.pivot_table(index='cifno',
                               columns=df['bulan_transaksi'].dt.month_name(),
                               values='total_amount_transaksi',
                               aggfunc='sum',
                               fill_value=0)

    pivot_ratas = df.pivot_table(index='cifno',
                                 columns=df['bulan_transaksi'].dt.month_name(),
                                 values='Ratas Giro',
                                 aggfunc='mean',
                                 fill_value=0)

    pivot_freq.columns = [f'frek_trx_{col.lower()}' for col in pivot_freq.columns]
    pivot_vol.columns = [f'vol_trx_{col.lower()}' for col in pivot_vol.columns]
    pivot_ratas.columns = [f'ratas_trx_{col.lower()}' for col in pivot_ratas.columns]

    df_trx_final = pd.concat([pivot_freq, pivot_vol, pivot_ratas], axis=1).reset_index()

    combined_df = pd.merge(df_final, df_trx_final, on='cifno', how='inner')

    # Handle any remaining NaNs
    combined_df.fillna(0, inplace=True)

    return combined_df

# Section for Customer Churn Prediction
st.write("## Prediksi Customer Churn")
uploaded_model = st.file_uploader("Upload Model File", type=["pkl"], key="model_upload")

if uploaded_model is not None:
    model = joblib.load(uploaded_model)
    st.success("Model loaded successfully!")

    uploaded_data_pred = st.file_uploader("Upload Data untuk Prediksi", type=["xlsx"], key="predict_upload")

    if uploaded_data_pred is not None:
        df_pred = load_data(uploaded_data_pred)
        processed_data_pred = process_data(df_pred)
        
        # Ensure all features are numeric
        processed_data_pred = processed_data_pred.apply(pd.to_numeric, errors='coerce')
        processed_data_pred.fillna(0, inplace=True)

        cifno = processed_data_pred['cifno'].copy()
        predictions = model.predict(processed_data_pred.drop(['cifno', 'nama_nasabah'], axis=1))

        hasil_prediksi = pd.DataFrame({
            'cifno': cifno,
            'prediksi': predictions
        })

        st.write("Hasil Prediksi Churn:")
        st.write(hasil_prediksi)
