import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import numpy as np

st.set_page_config(page_title="Model Prediksi Churn CMS/Qlola", page_icon="Logo-White.png")

st.image("Logo-Bank-BRI.png", width=100)
st.title("Model Prediksi Churn CMS/Qlola BRI")

# Function to load data
def load_data(file):
    return pd.read_excel(file)

# Function to process data
def process_data(df, is_training_data=True):
    st.write("Starting data processing...")

    # Display the columns present in the dataframe
    st.write("Dataset Columns:")
    st.write(df.columns)

    # Ensure 'TUTUP_REKENING' column exists if processing training data
    if 'TUTUP_REKENING' not in df.columns and is_training_data:
        st.error("Column 'TUTUP_REKENING' not found in the dataset.")
        st.stop()

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
    df_final = df_final.drop(['segmentasi_bpr', 'rgdesc', 'Platform_Channel', 'Giro Type', 'Loan Type', 'Giro Type Group', 'Loan Type Group'], axis=1, errors='ignore')

    df['bulan_transaksi'] = pd.to_datetime(df['bulan_transaksi'], format='%B')
    df.sort_values('bulan_transaksi', inplace=True)

    df['Ratas Giro'] = pd.to_numeric(df['Ratas Giro'], errors='coerce')
    df['total_amount_transaksi'] = pd.to_numeric(df['total_amount_transaksi'], errors='coerce')
    df['freq_transaksi'] = pd.to_numeric(df['freq_transaksi'], errors='coerce')

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

    if is_training_data:
        st.write("Training data detected. Processing 'TUTUP_REKENING' and 'STATUS_CHURN' columns.")

        combined_df['TUTUP_REKENING'] = combined_df['TUTUP_REKENING'].apply(lambda x: False if x == '(blank)' else True)

        def tentukan_status_churn(row):
            return 'CHURN' if row['TUTUP_REKENING'] else 'TIDAK'

        combined_df['STATUS_CHURN'] = combined_df.apply(tentukan_status_churn, axis=1)
        combined_df['STATUS_CHURN'] = combined_df['STATUS_CHURN'].apply(lambda x: 0 if x == 'TIDAK' else 1)

        st.write("Distribusi STATUS_CHURN sebelum upsampling.")
        st.write(combined_df.STATUS_CHURN.value_counts())

        df_majority = combined_df[combined_df.STATUS_CHURN == 0]
        df_minority = combined_df[combined_df.STATUS_CHURN == 1]

        df_minority_upsampled = resample(df_minority,
                                         replace=True,
                                         n_samples=len(df_majority),
                                         random_state=123)

        combined_df = pd.concat([df_majority, df_minority_upsampled])

        st.write("Distribusi STATUS_CHURN setelah upsampling.")
        st.write(combined_df.STATUS_CHURN.value_counts())

        combined_df = combined_df.drop(['TUTUP_REKENING'], axis=1, errors='ignore')

        st.write("Columns in combined_df after processing for training:")
        st.write(combined_df.columns)

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

def train_and_evaluate_models(X_train, Y_train, X_test, Y_test):
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,  # num_trees
            max_depth=10,
            min_samples_split=6,  # minsplit
            min_samples_leaf=3,  # minbucket
            max_features='sqrt',  # mtry
            random_state=42,
            n_jobs=-1,
            bootstrap=True,  # replace
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,  # maxdepth
            min_samples_split=6,  # minsplit
            min_samples_leaf=3,  # minbucket
            ccp_alpha=0.01,  # cp
            random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50,  # n_estimators
            learning_rate=1.0,  # learning_rate
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,  # nrounds
            max_depth=10,  # maxdepth
            learning_rate=0.1,  # eta
            subsample=0.8,  # subsample
            colsample_bytree=0.8,  # colsample_bytree
            gamma=0.1,  # gamma
            min_child_weight=1,  # min_child_weight
            reg_lambda=1.0,  # lambda
            reg_alpha=0.1,  # alpha
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }

    results = {}

    for model_name, model in models.items():
        st.write(f"Training {model_name} model...")
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        report = classification_report(Y_test, predictions, output_dict=True)
        results[model_name] = {
            'model': model,
            'report': report,
            'confusion_matrix': confusion_matrix(Y_test, predictions)
        }

    return results

# Section for Data Processing
st.write("## Data Processing")
uploaded_file = st.file_uploader("Upload File Data", type=["xlsx"], key="data_upload")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Dataset Columns:")
    st.write(df.columns)

    processed_data = process_data(df)
    st.write("Processed Data:")
    st.write(processed_data.head())
    st.write(processed_data.describe())

# Section for Model Training
st.write("## Training Model")

uploaded_file_train = st.file_uploader("Upload File Data for Training", type=["xlsx"], key="train_upload")

if uploaded_file_train is not None:
    df_train = load_data(uploaded_file_train)
    st.write("Dataset Columns:")
    st.write(df_train.columns)

    processed_data_train = process_data(df_train)
    st.write("Processed Training Data:")
    st.write(processed_data_train.head())
    st.write(processed_data_train.describe())

    if st.button("Train Model"):
        X = processed_data_train.drop(['STATUS_CHURN'], axis=1)
        Y = processed_data_train['STATUS_CHURN']
        
        # Ensure all features are numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        X.fillna(0, inplace=True)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        results = train_and_evaluate_models(X_train, Y_train, X_test, Y_test)

        for model_name, result in results.items():
            st.write(f"### {model_name}")
            st.write("Classification Report:")
            st.json(result['report'])
            st.write("Confusion Matrix:")
            st.write(result['confusion_matrix'])

            feature_importances = result['model'].feature_importances_
            features = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': feature_importances
            })
            features = features.sort_values(by='Importance', ascending=False)
            st.write(features)

            fig, ax = plt.subplots()
            features.plot(kind='bar', x='Feature', y='Importance', ax=ax)
            ax.set_title(f"Feature Importances - {model_name}")
            ax.set_ylabel("Importance")
            st.pyplot(fig)

        if st.button("Save Model"):
            joblib.dump(results['Random Forest']['model'], 'rf_model.pkl')
            st.success("Model saved successfully!")

# Section for Customer Churn Prediction
st.write("## Prediksi Customer Churn")
uploaded_model = st.file_uploader("Upload Model File", type=["pkl"], key="model_upload")

if uploaded_model is not None:
    RF_model = joblib.load(uploaded_model)
    st.success("Model loaded successfully!")

uploaded_data_pred = st.file_uploader("Upload Data untuk Prediksi", type=["xlsx"], key="predict_upload")

if uploaded_data_pred is not None and 'RF_model' in locals():
    df_pred = load_data(uploaded_data_pred)
    processed_data_pred = process_data(df_pred, is_training_data=False)
    cifno = processed_data_pred['cifno'].copy()
    predictions = RF_model.predict(processed_data_pred.drop(['cifno'], axis=1))

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

    analisis_data_dropped = analisis_data.drop(['Other','ratas_trx_january','ratas_trx_february','ratas_trx_march','ratas_trx_april','ratas_trx_may','ratas_trx_june','ratas_trx_july','vol_trx_january','vol_trx_february','vol_trx_march','vol_trx_april','vol_trx_may','vol_trx_june','vol_trx_july','frek_trx_january','frek_trx_february','frek_trx_march','frek_trx_april','frek_trx_may','frek_trx_june','frek_trx_july'], axis=1)

    correlation_matrix = analisis_data_dropped.corr()

    st.write("### Correlation Heatmap antar Fitur")
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5}, ax=ax)
    ax.set_title('Heatmap Korelasi Fitur')
    st.pyplot(fig)
