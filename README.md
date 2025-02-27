# Water Quality
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
### 📌 Proje Açıklaması
Bu projede, model performansını artırmak amacıyla **RandomizedSearchCV** kullanılarak **Hyperparameter Tuning** yapılmıştır. Kullanılan makine öğrenmesi modelleri arasında **CatBoost** en yüksek başarıyı elde etmiştir. Test veri seti üzerinde yapılan değerlendirmelere göre, **CatBoost** modelinin accuracy_score **%80** olarak ölçülmüştür. Ayrıca, **CatBoost** modelinin **değişken (feature) önem düzeyleri** incelenmiş ve hangi değişkenlerin model üzerinde daha fazla etkisi olduğu belirlenmiştir.

**Confusion Matrix**: for Test Data

![Confusion Matrix](https://github.com/user-attachments/assets/92428188-a969-4920-b69b-aa2725cc07f4)

# Heart Attack
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/sonialikhan/heart-attack-analysis-and-prediction-dataset)
### 📌 Proje Açıklaması
Bu projede, veri setindeki **aykırı değerlerin (outliers) tespiti** için **Z-skoru** ve **Winsorizing** yöntemleri kullanılmıştır. Ayrıca, **kategorik özellikler (categorical features)** için uygun **Encoding** işlemleri gerçekleştirilmiştir. Model optimizasyonu aşamasında ise **GridSearchCV** kullanılarak **Hyperparameter Tuning** yapılmıştır.
Bu projede kullanılan makine öğrenmesi modelleri arasında **LogisticRegression** en yüksek başarıyı elde etmiştir. Test veri seti üzerinde yapılan değerlendirmelere göre, **LogisticRegression** modelinin accuracy_score **%88** olarak ölçülmüştür.

| Confusion Matrix | ROC Curve |
|------------------|-----------|
| ![Confusion Matrix](https://github.com/user-attachments/assets/ef96fbd7-da96-4a9f-9f19-6681d97cede0) | ![ROC](https://github.com/user-attachments/assets/5f8a5c4d-b083-4fdb-8ba0-235093186701) |

# MNIST
### 📌 Proje Açıklaması
Bu projede **Principal Component Analysis (PCA)** kullanılarak **boyut indirgeme** işlemi gerçekleştirilmiştir.  
Model optimizasyonu için **Hyperparameter Tuning** yöntemi uygulanmış ve **GridSearchCV** kullanılarak en iyi parametreler belirlenmiştir.  

Bu projede kullanılan makine öğrenmesi modelleri arasında **MLP** ve **SVM**, en yüksek başarıyı elde etmiştir. Test veri seti üzerinde yapılan değerlendirmelere göre, **doğruluk oranı %97** olarak ölçülmüştür. Model değerlendirme sürecinde **Voting Classifier (Soft Voting)** kullanıldı ve **test verisi üzerindeki doğruluk oranı %96** olarak hesaplanmıştır.

<details>
  <summary><b>MLP Confusion Matrix</b></summary>
  <img src="https://github.com/user-attachments/assets/216e09e5-1ecc-4b31-b411-0cd33719b6b2">
</details>

<details>
  <summary><b>SVM Confusion Matrix</b></summary>
  <img src="https://github.com/user-attachments/assets/0e95062d-467d-496e-9d79-7b30c03b8a77">
</details>

<details>
  <summary><b>Voting Classifier (Soft) Confusion Matrix</b></summary>
  <img src="https://github.com/user-attachments/assets/6faa04c2-d4db-46ff-8939-e080db12cd10">
</details>
