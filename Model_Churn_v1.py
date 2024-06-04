import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

st.set_page_config(page_title="Model Prediksi Churn CMS/Qlola", page_icon="Logo-White.png")

st.image("Logo-Bank-BRI.png", width=100)
st.title("Model Prediksi Churn CMS/Qlola BRI")

def load_data(file):
    return pd.read_excel(file)

def process_data(df, is_training_data=True):
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
        elif any(term in loan_type for term in ['Menengah', 'MNGH', 'DIV BUMN']):
            return 'Loan Type: Menengah & Besar'
        elif any(term in loan_type for term in ['VALAS', 'CASH', 'FPJP', 'VLS']):
            return 'Loan Type: Valas & Fasilitas Khusus'
        elif any(term in loan_type for term in ['DKM', 'KREDIT', 'Kredit', 'Program']):
            return 'Loan Type: Kredit Spesial & Program'
        else:
            return 'Loan Type: Lainnya'

    df['Giro Type Group'] = df['Giro Type'].apply(giro_type_group)
    df['Loan Type Group'] = df['Loan Type'].apply(loan_type_group)

    giro_type_group_dummies = pd.get_dummies(df['Giro Type Group'])
    loan_type_group_dummies = pd.get_dummies(df['Loan Type Group'])

    segmen_dummies = pd.get_dummies(df['segmentasi_bpr'])
    rgdesc_dummies = pd.get_dummies(df['rgdesc'])
    platform_channel_dummies = pd.get_dummies(df['Platform_Channel'])

    segmen_agg = segmen_dummies.groupby(df['cifno']).max()
    rgdesc_agg = rgdesc_dummies.groupby(df['cifno']).max()
    platform_channel_agg = platform_channel_dummies.groupby(df['cifno']).max()
    giro_type_group_agg = giro_type_group_dummies.groupby(df['cifno']).max()
    loan_type_group_agg = loan_type_group_dummies.groupby(df['cifno']).max()

    df_final = pd.concat([segmen_agg, rgdesc_agg, platform_channel_agg, giro_type_group_agg, loan_type_group_agg], axis=1)
    df_final.reset_index(inplace=True)
    df_final = pd.merge(df_final, df.drop_duplicates('cifno'), on='cifno', how='inner')
    df_final = df_final.drop(['segmentasi_bpr','rgdesc','Platform_Channel','Giro Type','Loan Type','Giro Type Group','Loan Type Group'], axis=1, errors='ignore')

    df['bulan_transaksi'] = pd.to_datetime(df['bulan_transaksi'], format='%B')
    df.sort_values('bulan_transaksi', inplace=True)

    pivot_freq = df.pivot_table(index='cifno',
                                columns=df['bulan_transaksi'].dt.month_name(),
                                values='total_amount_transaksi',
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
                                 aggfunc='sum',
                                 fill_value=0)

    pivot_freq.columns = [f'frek_trx_{col.lower()}' for col in pivot_freq.columns]
    pivot_vol.columns = [f'vol_trx_{col.lower()}' for col in pivot_vol.columns]
    pivot_ratas.columns = [f'ratas_trx_{col.lower()}' for col in pivot_ratas.columns]

    df_trx_final = pd.concat([pivot_freq, pivot_vol, pivot_ratas], axis=1).reset_index()

    df_trx_last_transaction = df.groupby('cifno')['bulan_transaksi'].max()
    df_trx_last_transaction = df_trx_last_transaction.dt.month_name()

    last_month = df['bulan_transaksi'].max().month
    df['Month'] = df['bulan_transaksi'].dt.month
    df_trx_consecutive = df.groupby('cifno')['Month'].apply(lambda x: all(month in x.values for month in range(last_month-2, last_month+1)))

    df_trx_bulan_terakhir = pd.DataFrame({
        'cifno': df_trx_last_transaction.index,
        'Last Transaction': df_trx_last_transaction.values,
        'Consecutive Transactions (Last 3 Months)': df_trx_consecutive
    }).reset_index(drop=True)

    combined_df = pd.merge(df_final, df_trx_final, on='cifno', how='inner')
    combined_df = pd.merge(combined_df, df_trx_bulan_terakhir, on='cifno', how='inner')

    if is_training_data:
        combined_df['TUTUP_REKENING'] = combined_df['TUTUP_REKENING'].apply(lambda x: False if x == '(blank)' else True)

        def tentukan_status_churn(row):
            return 'CHURN' if row['TUTUP_REKENING'] else 'TIDAK'

        combined_df['STATUS_CHURN'] = combined_df.apply(tentukan_status_churn, axis=1)
        combined_df['STATUS_CHURN'] = combined_df['STATUS_CHURN'].apply(lambda x: 0 if x == 'TIDAK' else 1)

        st.write("Distribusi STATUS_CHURN sebelum upsampling.")
        st.write(combined_df.STATUS_CHURN.value_counts())

        df_majority = combined_df[combined_df.STATUS_CHURN==0]
        df_minority = combined_df[combined_df.STATUS_CHURN==1]

        df_minority_upsampled = resample(df_minority,
                                         replace=True,
                                         n_samples=len(df_majority),
                                         random_state=123)

        combined_df = pd.concat([df_majority, df_minority_upsampled])

        st.write("Distribusi STATUS_CHURN setelah upsampling.")
        st.write(combined_df.STATUS_CHURN.value_counts())

    combined_df = combined_df.drop(['TUTUP_REKENING'], axis=1, errors='ignore')

    bulan_ke_angka = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
        'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
        'November': 11, 'December': 12
    }
    combined_df['Last Transaction'] = combined_df['Last Transaction'].map(bulan_ke_angka)

    for kolom in combined_df.columns:
        if combined_df[kolom].dtype == 'bool':
            combined_df[kolom] = combined_df[kolom].astype(int)

    return combined_df

def train_model(X_train, Y_train):
    RF = RandomForestClassifier(
        n_estimators=10,
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    RF.fit(X_train, Y_train)
    joblib.dump(RF, 'rf_model.pkl')
    return RF

def predict_churn(model, X_test):
    return model.predict(X_test)

st.write("## Pelatihan Model")

uploaded_file = st.file_uploader("Upload File Data", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    combined_data = process_data(df)
    st.write("Data berhasil diproses.")

    st.write(combined_data.head())
    st.write(combined_data.describe())

    X = combined_data.drop(['STATUS_CHURN'], axis=1)
    Y = combined_data['STATUS_CHURN']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    RF_model = train_model(X_train, Y_train)

    st.write("Model telah dilatih.")

    Y_pred_rf = predict_churn(RF_model, X_test)
    report = classification_report(Y_test, Y_pred_rf, digits=4)
    confusion = confusion_matrix(Y_test, Y_pred_rf)
    st.write("Classification Report:", report)
    st.write("Confusion Matrix:", confusion)

    feature_importances = RF_model.feature_importances_
    features = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })
    features = features.sort_values(by='Importance', ascending=False)
    st.write(features)

    fig, ax = plt.subplots()
    features.plot(kind='bar', x='Feature', y='Importance', ax=ax)
    ax.set_title("Feature Importances")
    ax.set_ylabel("Importance")
    st.pyplot(fig)

st.write("## Prediksi Churn dengan Data Baru")
uploaded_data_pred = st.file_uploader("Upload Data untuk Prediksi", type=["xlsx"], key="predict")

if uploaded_data_pred is not None:
    try:
        RF_model = joblib.load('rf_model.pkl')

        df_pred = load_data(uploaded_data_pred)

        processed_data_pred = process_data(df_pred, is_training_data=False)
        cifno = processed_data_pred['cifno'].copy()
        predictions = predict_churn(RF_model, processed_data_pred.drop(['cifno'], axis=1))

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
        fitur_normalisasi.drop(['Other','ratas_trx_january','ratas_trx_february','ratas_trx_march','ratas_trx_april','ratas_trx_may','ratas_trx_june','ratas_trx_july','vol_trx_january','vol_trx_february','vol_trx_march','vol_trx_april','vol_trx_may','vol_trx_june','vol_trx_july','frek_trx_january','frek_trx_february','frek_trx_march','frek_trx_april','frek_trx_may','frek_trx_june','frek_trx_july'], axis=0, inplace=True)

        fitur_normalisasi.plot(kind='barh', stacked=True, figsize=(15, 10), colormap='viridis')
        plt.title('Komposisi Status Churn 100% untuk Setiap Fitur')
        plt.xlabel('Fitur')
        plt.ylabel('Proporsi (%)')
        plt.legend(title='Status Churn', labels=['0', '1'])
        plt.tight_layout()

        st.pyplot(plt)

        fitur_persentase = fitur_normalisasi.applymap(lambda x: f"{x:.2f}%")
        st.dataframe(fitur_persentase)

        volume_columns = [col for col in analisis_data.columns if 'vol_trx_' in col]
        frequency_columns = [col for col in analisis_data.columns if 'frek_trx_' in col]

        analisis_data['average_volume'] = analisis_data[volume_columns].mean(axis=1)
        analisis_data['total_frequency'] = analisis_data[frequency_columns].sum(axis=1)

        analisis_data['profitability_score'] = analisis_data['average_volume'] * analisis_data['total_frequency']

        plt.figure(figsize=(10, 6))
        plt.scatter(analisis_data['profitability_score'], analisis_data['prediksi'], c=analisis_data['prediksi'], cmap='viridis', alpha=0.6)
        plt.title('Profitability Score vs Churn Prediction')
        plt.xlabel('Profitability Score')
        plt.ylabel('Churn Prediction (0=Low, 1=High)')
        plt.grid(True)
        plt.colorbar(label='Churn Prediction')
        plt.show()
        st.pyplot(plt)

        filtered_data = analisis_data.query('prediksi == 1')
        sorted_filtered_data = filtered_data.sort_values(by='profitability_score', ascending=False)

        kolom_tabel = ['average_volume', 'total_frequency', 'profitability_score', 'prediksi']
        tabel_urut_profitabilitas = sorted_filtered_data[kolom_tabel].copy()

        st.write("### Tabel Nasabah dengan Profitability Score")
        st.dataframe(tabel_urut_profitabilitas)

        analisis_data_dropped = analisis_data.drop(['Other','ratas_trx_january','ratas_trx_february','ratas_trx_march','ratas_trx_april','ratas_trx_may','ratas_trx_june','ratas_trx_july','vol_trx_january','vol_trx_february','vol_trx_march','vol_trx_april','vol_trx_may','vol_trx_june','vol_trx_july','frek_trx_january','frek_trx_february','frek_trx_march','frek_trx_april','frek_trx_may','frek_trx_june','frek_trx_july'], axis=1)

        correlation_matrix = analisis_data_dropped.corr()

        st.write("### Correlation Heatmap antar Fitur")
        plt.figure(figsize=(20, 15))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
        plt.title('Heatmap Korelasi Fitur')
        plt.tight_layout()

        plt.show()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {e}")
