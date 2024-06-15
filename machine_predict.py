# -*- coding: utf-8 -*-
"""
@author: kerim
"""

import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures



# Önceden eğitilmiş modellerin yüklenmesi
model_failure_comp1_bagging = joblib.load('bagging_model_failure_comp1.joblib')
model_failure_comp1_gb = joblib.load('gb_model_failure_comp1.joblib')
model_failure_comp2_bagging = joblib.load('bagging_model_failure_comp2.joblib')
model_failure_comp2_gb = joblib.load('gb_model_failure_comp2.joblib')
model_failure_comp4 = joblib.load('gb_model_failure_comp4.joblib')
model_failure_none = joblib.load('bagging_model_failure_none.joblib')




# Veri işleme fonksiyonları
def create_lag_features(data):
    # Lag özelliklerinin oluşturulması
    columns_to_lag = {
        'rotate_mean_3h': [4],
        'vibration_mean_3h': [2, 4],
        'volt_mean_24h': [8],
        'rotate_min_24h': [1, 8],
        'rotate_max_24h': [1],
        'rotate_mean_24h': [1, 8]
    }
    
    for col, lags in columns_to_lag.items():
        for lag in lags:
            lag_col_name = f'{col}_lag{lag}'
            data[lag_col_name] = data.groupby('machineID')[col].shift(lag)
            data[lag_col_name].fillna(data[col], inplace=True)
    
    return data

def prepare_data_for_model(raw_data):
    # Gerekli sütunları seçme ve gereksiz sütunları düşürme
    data = raw_data.drop(columns=['datetime'])

    
    # Kategorik değişkenleri kodlama
    data = data.drop(columns=['model'])
    data = data.drop(columns=['failure'])

    
    # Lag özelliklerini oluşturma
    data = create_lag_features(data)
    
    data['comp1_comp2_product'] = data['comp1'] * data['comp2']
    
    
    
    # Etkileşim özellikleri veya polinom özellikleri oluşturma
    poly_features = ['comp1', 'comp2']
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features_transformed = poly.fit_transform(data[poly_features])
    poly_feature_names = poly.get_feature_names_out(poly_features)
    poly_df = pd.DataFrame(poly_features_transformed, columns=poly_feature_names, index=data.index)
    data = pd.concat([data, poly_df], axis=1)
    
    
    # İşlem yapılacak özelliklerin belirlenmesi
    poly_feature_cols = ['vibration_mean_3h', 'volt_mean_3h']
    
    # PolynomialFeatures kullanarak etkileşim özelliklerini oluşturma
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features_transformed = poly.fit_transform(data[poly_feature_cols])
    poly_feature_names = poly.get_feature_names_out(poly_feature_cols)
    
    # Yeni özellikleri DataFrame'e ekleme
    poly_df = pd.DataFrame(poly_features_transformed, columns=poly_feature_names, index=data.index)
    data = pd.concat([data, poly_df], axis=1)
        
    
    # Model için seçilen sütunları seçme
    columns_to_select = [
        'rotate_min_3h', 'vibration_min_3h', 'vibration_max_3h', 'volt_mean_3h',
        'vibration_mean_3h', 'pressure_sd_3h', 'volt_min_24h', 'vibration_min_24h',
        'rotate_max_24h', 'volt_mean_24h', 'rotate_mean_24h', 'vibration_mean_24h',
        'error1count', 'error5count', 'comp1', 'comp2', 'comp4', 'rotate_mean_3h',
        'rotate_mean_3h_lag4', 'vibration_mean_3h_lag2', 'vibration_mean_3h_lag4',
        'volt_mean_24h_lag8', 'rotate_min_24h_lag1', 'rotate_min_24h_lag8',
        'rotate_max_24h_lag1', 'rotate_mean_24h_lag1', 'rotate_mean_24h_lag8',
        'vibration_mean_3h volt_mean_3h', 'comp1_comp2_product',
        'comp1 comp2'
    ]
    data_selected = data[columns_to_select]
    
    # Ölçeklendirme
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_selected), columns=data_selected.columns)
    
    return data_scaled

# Veri yükleme
raw_data = pd.read_csv('Makine verileri.csv')
#raw_data = pd.read_csv('Makine verileri_3.csv') .test için kullanılmıştır.



# Model parametreleri
params = {
    'failure_comp1': {'max_samples': 0.7, 'n_estimators': 50},
    'failure_comp2': {'max_samples': 0.5, 'n_estimators': 50},
    'failure_comp4': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50},
    'failure_none': {'max_samples': 1.0, 'n_estimators': 50},
}

# Veriyi modelleme için hazırlama
data_scaled = prepare_data_for_model(raw_data)

# Özellikler ve hedef değişkenleri ayırma
X = data_scaled



# Tahmin yapma işlemi ve ortalamaları alma
pred_failure_comp1_avg = (model_failure_comp1_bagging.predict(X) + model_failure_comp1_gb.predict(X)) / 2
pred_failure_comp2_avg = (model_failure_comp2_bagging.predict(X) + model_failure_comp2_gb.predict(X)) / 2

# Tahmin sonuçlarını bir DataFrame'e dönüştürme
predictions_df = pd.DataFrame({
    'failure_comp1': pred_failure_comp1_avg,
    'failure_comp2': pred_failure_comp2_avg,
    'failure_comp4': model_failure_comp4.predict(X),
    'failure_none': model_failure_none.predict(X)
})

# Tahminleri CSV dosyasına kaydetme
predictions_df.to_csv('predictions.csv', index=False)
