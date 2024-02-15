#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Fungsi Pengunggahan dan Pemrosesan Data
def load_data(file):
    return pd.read_excel(file)

def process_data(df_data_akun, df_trx, df_tutup_rek):
    # Fungsi untuk mengelompokkan Giro Type
    def giro_type_group(giro_type):
        if pd.isna(giro_type):
            return 'Other'
        elif giro_type.startswith('G'):
            return 'Giro Type G'
        elif giro_type.startswith('H'):
            return 'Giro Type H'
        else:
            return 'Other'
        
    # Fungsi untuk mengelompokkan Loan Type
    def loan_type_group(loan_type):
        if pd.isna(loan_type) or loan_type == '(blank)':
            return 'Blank/Other'
        elif 'Ritel Komersial' in loan_type:
            return 'Ritel Komersial'
        elif 'KUR Kecil' in loan_type:
            return 'Kredit Usaha Kecil (KUR)'
        elif loan_type == 'Kredit Pangan':
            return 'Kredit Pangan'
        elif 'Menengah Kanwil' in loan_type:
            return 'Menengah Kanwil'
        else:
            return 'Program Lainnya'
    
    # Terapkan fungsi ke kolom Giro Type dan Loan Type
    df_data_akun['Giro Type Group'] = df_data_akun['Giro Type'].apply(giro_type_group)
    df_data_akun['Loan Type Group'] = df_data_akun['Loan Type'].apply(loan_type_group)
    
    # Buat one-hot encoding untuk Giro Type Group dan Loan Type Group
    giro_type_group_dummies = pd.get_dummies(df_data_akun['Giro Type Group'])
    loan_type_group_dummies = pd.get_dummies(df_data_akun['Loan Type Group'])
    
    # Mendapatkan one-hot encoding untuk sisa kolom kategorikal
    segmen_dummies = pd.get_dummies(df_data_akun['segmentasi_bpr'])
    rgdesc_dummies = pd.get_dummies(df_data_akun['rgdesc'])
    platform_channel_dummies = pd.get_dummies(df_data_akun['Platform_Channel'])

    # Agregasi dengan groupby dan max untuk setiap nasabah
    segmen_agg = segmen_dummies.groupby(df_data_akun['nama_nasabah']).max()
    rgdesc_agg = rgdesc_dummies.groupby(df_data_akun['nama_nasabah']).max()
    platform_channel_agg = platform_channel_dummies.groupby(df_data_akun['nama_nasabah']).max()
    giro_type_group_agg = giro_type_group_dummies.groupby(df_data_akun['nama_nasabah']).max()
    loan_type_group_agg = loan_type_group_dummies.groupby(df_data_akun['nama_nasabah']).max()

    # Gabungkan semua DataFrame yang telah di-agregasi
    df_data_akun_final = pd.concat([segmen_agg, rgdesc_agg, platform_channel_agg, giro_type_group_agg, loan_type_group_agg], axis=1)
    df_data_akun_final.reset_index(inplace=True)  # Opsional: Reset index jika Anda ingin 'nama_nasabah' sebagai kolom
    df_data_akun_final
    
    # Konversi bulan_transaksi menjadi format tanggal
    df_trx['bulan_transaksi'] = pd.to_datetime(df_trx['bulan_transaksi'], format='%B')
    
    # Urutkan dataframe berdasarkan bulan_transaksi
    df_trx.sort_values('bulan_transaksi', inplace=True)
    
    # Buat pivot table untuk frekuensi transaksi
    pivot_freq = df_trx.pivot_table(index='nama_nasabah', 
                                    columns=df_trx['bulan_transaksi'].dt.month_name(), 
                                    values='Sum of freq_transaksi',
                                    aggfunc='sum',
                                    fill_value=0)

    # Buat pivot table untuk volume transaksi
    pivot_vol = df_trx.pivot_table(index='nama_nasabah', 
                                   columns=df_trx['bulan_transaksi'].dt.month_name(), 
                                   values='Sum of total_amount_transaksi',
                                   aggfunc='sum',
                                   fill_value=0)

    # Buat pivot table untuk ratas giro
    pivot_ratas = df_trx.pivot_table(index='nama_nasabah', 
                                     columns=df_trx['bulan_transaksi'].dt.month_name(), 
                                     values='Average of Ratas Giro',
                                     aggfunc='sum',
                                     fill_value=0)

    # Ubah nama kolom untuk membedakan antara frekuensi dan volume
    pivot_freq.columns = [f'frek_trx_{col.lower()}' for col in pivot_freq.columns]
    pivot_vol.columns = [f'vol_trx_{col.lower()}' for col in pivot_vol.columns]
    pivot_ratas.columns = [f'ratas_trx_{col.lower()}' for col in pivot_ratas.columns]

    # Gabungkan kedua pivot table
    df_trx_final = pd.concat([pivot_freq, pivot_vol, pivot_ratas], axis=1).reset_index()
    df_trx_final
    
    # Konversi `bulan_transaksi` ke datetime dan asumsikan semua transaksi terjadi pada tahun yang sama
    df_trx['bulan_transaksi'] = pd.to_datetime(df_trx['bulan_transaksi'], format='%B')
    
    # Temukan transaksi terakhir untuk setiap nasabah
    df_trx_last_transaction = df_trx.groupby('nama_nasabah')['bulan_transaksi'].max()
    
    # Ubah timestamp menjadi nama bulan
    df_trx_last_transaction = df_trx_last_transaction.dt.month_name()
    
    # Periksa apakah setiap nasabah melakukan transaksi berturut-turut dalam 3 bulan terakhir
    # Pertama, temukan bulan terakhir dalam data
    last_month = df_trx['bulan_transaksi'].max().month
    
    # Lalu, periksa apakah ada transaksi untuk setiap bulan dari 3 bulan terakhir
    df_trx['Month'] = df_trx['bulan_transaksi'].dt.month
    df_trx_consecutive = df_trx.groupby('nama_nasabah')['Month'].apply(lambda x: all(month in x.values for month in range(last_month-2, last_month+1)))
    
    # Gabungkan hasilnya ke dataframe
    df_trx_bulan_terakhir = pd.DataFrame({
        'nama_nasabah': df_trx_last_transaction.index,
        'Last Transaction': df_trx_last_transaction.values,
        'Consecutive Transactions (Last 3 Months)': df_trx_consecutive
    }).reset_index(drop=True)
    df_trx_bulan_terakhir
    
    df_tutup_rek.drop('Sum of freq_transaksi',axis=1,inplace=True)
    df_tutup_rek.sort_values(by='nama_nasabah')
    # Mengubah kolom TUTUP_REKENING menjadi boolean
    df_tutup_rek['TUTUP_REKENING'] = df_tutup_rek['TUTUP_REKENING'].apply(lambda x: False if x == '(blank)' else True)

    # COMBINED
    
    # Gabungkan df_data_akun_final dengan df_trx_final
    combined_df = pd.merge(df_data_akun_final, df_trx_final, on='nama_nasabah', how='inner')

    # Gabungkan hasilnya dengan df_trx_bulan_terakhir
    combined_df = pd.merge(combined_df, df_trx_bulan_terakhir, on='nama_nasabah', how='inner')

    # Gabungkan hasilnya dengan df_tutup_rek
    combined_df = pd.merge(combined_df, df_tutup_rek, on='nama_nasabah', how='inner')

    def tentukan_status_churn(row):
        if row['TUTUP_REKENING'] == True:
            return 'CHURN'
        elif row['Consecutive Transactions (Last 3 Months)'] == True:
            return 'TIDAK'
        else:
            return 'BERPOTENSI'

    # Terapkan fungsi ke setiap baris DataFrame
    combined_df['STATUS_CHURN'] = combined_df.apply(tentukan_status_churn, axis=1)

    combined_df_test = combined_df
    
    # 1. Ubah nama_nasabah menjadi ID
    combined_df_test = combined_df_test.reset_index(drop=True)  # Reset index jika belum
    combined_df_test.index += 1  # Tambahkan 1 agar ID dimulai dari 1
    combined_df_test['id'] = combined_df_test.index  # Tambahkan kolom ID

    # 2. Ubah Last Transaction menjadi angka sesuai nama bulan
    bulan_ke_angka = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 
        'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 
        'November': 11, 'December': 12
    }
    combined_df_test['Last Transaction'] = combined_df_test['Last Transaction'].map(bulan_ke_angka)

    # 3. Ubah STATUS_CHURN
    def convert_status_churn(value):
        if value == 'TIDAK':
            return 0
        elif value == 'BERPOTENSI':
            return 1
        elif value == 'CHURN':
            return 2
    
    combined_df_test['STATUS_CHURN'] = combined_df_test['STATUS_CHURN'].apply(convert_status_churn)

    # 4. Ubah semua kolom boolean
    # Iterasi melalui setiap kolom di DataFrame
    for kolom in combined_df_test.columns:
        # Cek jika tipe data kolom adalah boolean
        if combined_df_test[kolom].dtype == 'bool':
            # Konversi kolom boolean ke integer
            combined_df_test[kolom] = combined_df_test[kolom].astype(int)
    
    # Hapus kolom nama_nasabah
    combined_df_test = combined_df_test.drop('nama_nasabah', axis=1)
    # Menempatkan kolom id di paling kiri
    kolom = ['id'] + [col for col in combined_df_test.columns if col != 'id']  # Buat list kolom baru dengan 'id' di awal
    combined_df_test = combined_df_test[kolom]  # Reindex DataFrame dengan urutan kolom baru

    return combined_df_test

# 2. Fungsi Training Model
def train_model(X_train, Y_train):
    # Inisiasi model Random Forest dengan parameter terbaik
    RF = RandomForestClassifier(
        n_estimators=10,
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    # Latih model dengan data pelatihan
    RF.fit(X_train, Y_train)
    
    # Simpan model
    joblib.dump(RF, 'rf_model.pkl')
    
    return RF

def predict_churn(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# UI Streamlit #

st.title('Aplikasi Prediksi Churnn')
st.write("## Pelatihan Model")

# Upload satu file Excel untuk pelatihan dan prediksi
uploaded_file = st.file_uploader("Upload File Data", type=["xlsx"])

if uploaded_file is not None:
    # Membaca setiap sheet ke dalam DataFrame terpisah
    df_data_akun = pd.read_excel(uploaded_file, sheet_name='Data Akun', header=9)
    df_trx = pd.read_excel(uploaded_file, sheet_name='Data Transaksi', header=9)
    df_tutup_rek = pd.read_excel(uploaded_file, sheet_name='Data Tutup Rekening', header=9)

    # Proses data
    combined_data = process_data(df_data_akun, df_trx, df_tutup_rek)
    st.write("Data berhasil diproses.")

    # Menampilkan cuplikan data dan statistik
    st.write(combined_data.head())
    st.write(combined_data.describe())

    # Pisahkan data untuk pelatihan dan uji
    X = combined_data.drop(['STATUS_CHURN','id'], axis=1)
    Y = combined_data['STATUS_CHURN']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Pelatihan model
    RF_model = train_model(X_train, Y_train)
    st.write("Model telah dilatih.")

    # Tampilkan hasil pelatihan
    Y_pred_rf = predict_churn(RF_model, X_test)
    report = classification_report(Y_test, Y_pred_rf, digits=4)
    confusion = confusion_matrix(Y_test, Y_pred_rf)
    st.write("Classification Report:", report)
    st.write("Confusion Matrix:", confusion)

# Prediksi churn dengan data baru
st.write("## Prediksi Churn dengan Data Baru")
uploaded_data_pred = st.file_uploader("Upload Data untuk Prediksi", type=["xlsx"], key="predict")

if uploaded_data_pred is not None:
    try:
        RF_model = joblib.load('rf_model.pkl')  # Muat model yang telah disimpan

        # Baca data
        df_data_akun_pred = pd.read_excel(uploaded_data_pred, sheet_name='Data Akun', header=9)
        df_trx_pred = pd.read_excel(uploaded_data_pred, sheet_name='Data Transaksi', header=9)
        df_tutup_rek_pred = pd.read_excel(uploaded_data_pred, sheet_name='Data Tutup Rekening', header=9)

        # Proses data untuk prediksi
        processed_data_pred = process_data(df_data_akun_pred, df_trx_pred, df_tutup_rek_pred)
            
        # Simpan id sebelum menghapus kolom untuk prediksi
        id_nasabah = processed_data_pred['id'].copy()

        # Lakukan prediksi (pastikan untuk menghapus kolom nama_nasabah dari data yang akan diprediksi)
        predictions = predict_churn(RF_model, processed_data_pred.drop(['id', 'STATUS_CHURN'], axis=1))

        # Gabungkan nama nasabah dengan prediksi dalam DataFrame baru
        hasil_prediksi = pd.DataFrame({
            'id': id_nasabah,
            'prediksi': predictions
        })
            
        # Menampilkan hasil prediksi
        st.write("Hasil Prediksi Churn:", hasil_prediksi)

        # Gabungkan hasil prediksi dengan data asli berdasarkan id
        analisis_data = pd.merge(hasil_prediksi, processed_data_pred, on='id', how='inner')
        # Drop kolom 'id' dan 'STATUS_CHURN' karena tidak diperlukan untuk analisis ini
        analisis_data.drop(['id', 'STATUS_CHURN'], axis=1, inplace=True)
        # Visualisasi distribusi fitur untuk setiap status churn
        st.write("## Analisis Karakteristik Nasabah Berdasarkan Status Churn")

        # Gabungkan hasil prediksi dengan data asli berdasarkan id
        analisis_data = pd.merge(hasil_prediksi, processed_data_pred, on='id', how='inner')

        # Drop kolom 'id' dan 'STATUS_CHURN' karena tidak diperlukan untuk analisis ini
        analisis_data.drop(['id', 'STATUS_CHURN'], axis=1, inplace=True)

        # Visualisasi semua fitur
        st.write("### Visualisasi Fitur")
        for fitur in analisis_data.columns:
            if analisis_data[fitur].dtype == 'int64' or analisis_data[fitur].dtype == 'float64':
                # Untuk fitur numerik, gunakan boxplot
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='prediksi', y=fitur, data=analisis_data)
                plt.title(f'Distribusi {fitur} untuk Setiap Status Churn')
                plt.xlabel('Status Churn')
                plt.ylabel(fitur)
                st.pyplot(plt)
                plt.clf()
            else:
                # Untuk fitur biner, gunakan count plot
                plt.figure(figsize=(10, 6))
                sns.countplot(x=fitur, hue='prediksi', data=analisis_data)
                plt.title(f'Distribusi {fitur} untuk Setiap Status Churn')
                plt.xlabel(fitur)
                plt.ylabel('Jumlah')
                st.pyplot(plt)
                plt.clf()      
    except Exception as e:
        st.error(f"An error occurred: {e}")


