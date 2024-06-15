# predict_machine_failures_ML

Bu proje, çeşitli sensör verileri ve hata sayımlarına dayalı olarak makine arızalarını tahmin etmek için bir makine öğrenme modeli geliştirmeyi amaçlamaktadır. Geçmiş sensör verileri ve hata sayımlarını kullanarak makine arızalarını doğru bir şekilde tahmin eden bir model geliştirilmiştir.

## Proje İçeriği

### Veri Ön İşleme (Data Preprocessing)
- Veriler üzerinde temizlik, dönüşüm ve eksik değerlerin işlenmesi gibi işlemler yapılmıştır.
- EDA (Exploratory Data Analysis) ve zaman serisi analizleri gerçekleştirilmiştir.

### Özellik Mühendisliği (Feature Engineering)
- Sensör verileri ve hata sayımları kullanılarak yeni özellikler türetilmiştir.

### Model Seçimi ve Hiperparametre Analizi (Model Selection and Hyperparameter Tuning)
- Çeşitli makine öğrenme algoritmaları denenmiş ve en iyi sonuç veren modeller seçilmiştir.

### Hibrit Model (Hybrid Model)
- İki algoritmayı hibrit olarak kullanan bir model geliştirilmiştir.
- Gradient Boosting ve Bagging Classifier algoritmaları birlikte kullanılarak daha doğru tahminler yapılması sağlanmıştır.

## Çalışma Dosyaları

- **data_preprocess_EDA.ipynb**:
  - Veri ön işleme ve çeşitli görsel analizler (EDA) yapılmıştır.

- **data_preprocess_timeseries_featureengineering.ipynb**:
  - Özellik mühendisliği ve zaman serisi analizleri gerçekleştirilmiştir.

- **model_creator.ipynb**:
  - İşlenmiş verilerle çeşitli algoritmalar kullanılarak çalışmalar yapılmış ve en iyi sonuçlara ulaşılan algoritmalar, parametreler ve veriler belirlenmiştir.

- **final_model_create.ipynb**:
  - Belirlenen özellikler, algoritmalar ve parametreler kullanılarak nihai model oluşturulmuş ve kaydedilmiştir.

- **machine_predict.py**:
  - Başlangıçta verilen `Makine_verileri.csv` formatındaki ham verileri input olarak alarak tahminler output'u oluşturmaktadır.

## Model

Eğitilen model, `failure_none`, `failure_comp1`, `failure_comp2` ve `failure_comp4` isimli encode edilmiş hedefleri Gradient Boosting ve Bagging Classifier algoritmalarını hibrit bir şekilde kullanarak tahmin etmektedir.
