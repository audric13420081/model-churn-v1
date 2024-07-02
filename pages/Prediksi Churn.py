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

# Define load_data function
def load_data(file):
    return pd.read_excel(file)

# Define process_data function
def process_data(df):
    st.write("Starting data processing...")

    # Display the columns present in the dataframe
    st.write("Dataset Columns:")
    st.write(df.columns)

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
        elif any term in loan type for term in ['VALAS', 'CASH', 'FPJP', 'VLS']):
            return 'Loan Type: Valas & Fasilitas Khusus'
        elif any term in loan type for term in ['DKM', 'KREDIT', 'Kredit', 'Program']):
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
    df_final = df_final.drop(['Giro Type', 'Loan Type', 'Giro Type Group', 'Loan Type Group'], axis=1, errors='ignore')

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

    df_trx_last_transaction = df.groupby('cifno')['bulan_transaksi'].max()
    df_trx_last_transaction = df_trx_last_transaction.dt.month_name()

    last_month = df['bulan_transaksi'].max().month
    df['Month'] = df['bulan_transaksi'].dt.month
    df_trx_consecutive = df.groupby('cifno')['Month'].apply(lambda x: all(month in x.values for month in range(last_month - 2, last_month + 1)))

    df_trx_bulan_terakhir = pd.DataFrame({
        'cifno': df_trx_last_transaction.index,
        'Last Transaction': df_trx_last_transaction.values,
        'Consecutive Transactions (Last 3 Months)': df_trx_consecutive
    }).reset_index(drop=True)

    combined_df = pd.merge(df_final, df_trx_final, on='cifno', how='inner')
    combined_df = pd.merge(combined_df, df_trx_bulan_terakhir, on='cifno', how='inner')

    # Label Encoding: Mengubah bulan menjadi angka dan tipe boolean menjadi integer
    bulan_ke_angka = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
        'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
        'November': 11, 'December': 12
    }
    combined_df['Last Transaction'] = combined_df['Last Transaction'].map(bulan_ke_angka)

    for kolom in combined_df.columns:
        if combined_df[kolom].dtype == 'bool':
            combined_df[kolom] = combined_df[kolom].astype(int)

    # Handle any remaining NaNs
    combined_df.fillna(0, inplace=True)

    return combined_df

# Section for Customer Churn Prediction
st.write("## Prediksi Customer Churn")
uploaded_model = st.file_uploader("Upload Model File", type=["pkl"], key="model_upload")

if uploaded_model is not None:
    # Model sudah didefinisikan di kode, tidak diunggah oleh pengguna
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=10,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=1,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    uploaded_data_pred = st.file_uploader("Upload Data untuk Prediksi", type=["xlsx"], key="predict_upload")

    if uploaded_data_pred is not None:
        df_pred = load_data(uploaded_data_pred)
        processed_data_pred = process_data(df_pred)
        
        # Ensure all features are numeric
        processed_data_pred = processed_data_pred.apply(pd.to_numeric, errors='coerce')
        processed_data_pred.fillna(0, inplace=True)

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