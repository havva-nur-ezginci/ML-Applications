Proje kapsamında farklı veri setleri üzerinde makine öğrenmesi algoritmaları ile çalışılmıştır.

# Table of Contents
- [Water Quality](#water-quality)
- [Heart Attack](#heart-attack)
- [MNIST](#mnist)
- [California Housing Prices](#california-housing-prices)
- [Customer Segmentation - Clustering](#customer-segmentation---clustering)
- [User Login Logs (Random) - Clustering (GMM) - Create DataSet](#random-user-login-logs---clustering)
- [Taxi-v3 Reinforcement Learning (Q-Learning)](#taxi-v3-reinforcement-learning)
- [FashionMNIST | CNN + RMSprop + ImageDataGenerator](#fashionmnist--cnn--rmsprop--imagedatagenerator)
- [Liver Cirrhosis Outcome Classification](#-liver-cirrhosis-outcome-classification)

----

## Water Quality
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
#### 📌 Proje Açıklaması
Bu projede, model performansını artırmak amacıyla **RandomizedSearchCV** kullanılarak **Hyperparameter Tuning** yapılmıştır. Kullanılan makine öğrenmesi modelleri arasında **CatBoost** en yüksek başarıyı elde etmiştir. Test veri seti üzerinde yapılan değerlendirmelere göre, **CatBoost** modelinin accuracy_score **%80** olarak ölçülmüştür. Ayrıca, **CatBoost** modelinin **değişken (feature) önem düzeyleri** incelenmiş ve hangi değişkenlerin model üzerinde daha fazla etkisi olduğu belirlenmiştir.

**Confusion Matrix**: for Test Data

![Confusion Matrix](https://github.com/user-attachments/assets/92428188-a969-4920-b69b-aa2725cc07f4)

----

## Heart Attack
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/sonialikhan/heart-attack-analysis-and-prediction-dataset)
#### 📌 Proje Açıklaması
Bu projede, veri setindeki **aykırı değerlerin (outliers) tespiti** için **Z-skoru** ve **Winsorizing** yöntemleri kullanılmıştır. Ayrıca, **kategorik özellikler (categorical features)** için uygun **Encoding** işlemleri gerçekleştirilmiştir. Model optimizasyonu aşamasında ise **GridSearchCV** kullanılarak **Hyperparameter Tuning** yapılmıştır.
Bu projede kullanılan makine öğrenmesi modelleri arasında **LogisticRegression** en yüksek başarıyı elde etmiştir. Test veri seti üzerinde yapılan değerlendirmelere göre, **LogisticRegression** modelinin accuracy_score **%88** olarak ölçülmüştür.

| Confusion Matrix | ROC Curve |
|------------------|-----------|
| ![Confusion Matrix](https://github.com/user-attachments/assets/ef96fbd7-da96-4a9f-9f19-6681d97cede0) | ![ROC](https://github.com/user-attachments/assets/5f8a5c4d-b083-4fdb-8ba0-235093186701) |

----

## MNIST
#### 📌 Proje Açıklaması
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

----


## California Housing Prices
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data)
#### 📌 Proje Açıklaması
To download the dataset, you need to set up the Kaggle API using the kaggle.json API key.
<details>
    <summary><h4>Steps to Set Up Kaggle API</h4></summary>

1. **Sign in to Kaggle**:
   - Go to [Kaggle](https://www.kaggle.com) and log in to your account.

2. **Create a New Kaggle API Token**:
   - Visit the [Kaggle API page](https://www.kaggle.com/docs/api).
   - Click on the "Create New API Token" button.
   - This will download the `kaggle.json` file.

3. **Place the `kaggle.json` File in the Appropriate Directory**:
   - **Windows**: Move the `kaggle.json` file to the following path:
     ```
     C:\Users\YourUser\.kaggle\kaggle.json
     ```
   - **Mac/Linux**: Move the `kaggle.json` file to the following path:
     ```
     ~/.kaggle/kaggle.json
     ```

4. **Install the Kaggle Package**:
   Run the following command to install the Kaggle API Python package:
   ```bash
   pip install kaggle

</details>

Bu projede, konut fiyatlarını tahmin etmek amacıyla çeşitli regresyon modelleri kullanılmıştır. Veri seti üzerinde bazı veri işleme teknikleri uygulanmış, model hiperparametre ayarlamaları yapılmış ve en iyi sonuç veren model seçilmiştir.

##### Veri İşleme Adımları

- **Kategorik Verilerin Kodlanması**: Kategorik sütunlar için One-Hot Encoding kullanılmıştır.
- **Eksik Verilerin Tamamlanması**: Eksik veriler, K-Nearest Neighbors (KNN) algoritması ile tamamlanmıştır.
- **Korelasyon Analizi**: Değişkenler arasındaki ilişkiler incelenmiş ve hedef değişkenle en güçlü ilişkiyi gösteren özellikler belirlenmiştir.

##### Model Seçimi ve Hiperparametre Ayarlaması

- **Hiperparametre Ayarlaması**: GridSearchCV kullanılarak modellerin hiperparametreleri optimize edilmiştir.
- **En İyi Model**: En iyi performans **DecisionTreeRegressor (DTR)** modeli ile elde edilmiştir. Bu modelin performans metrikleri:
  - **MSE (Ortalama Kare Hatası)**: 3,5
  - **R² Skoru**: 0.7308
- **Özellik Önem Düzeyleri**: Final modeldeki özelliklerin önem düzeyleri incelenmiştir.

----


## Customer Segmentation - Clustering
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/data)
#### 📌 Proje Açıklaması
- Veri Analizi ve Ön İşleme: Kategorik değişkenlerin **encode** edilmesi, **verinin standartlaştırılması**
- Boyut Azaltma: **PCA ile optimum bileşen sayısının belirlenmesi**
- Optimum Küme Sayısının Belirlenmesi: **Yellowbrick** 
- Kümeleme Modelleri: **K-Means ve Hiyerarşik Kümeleme (Dendrogram - Complete yöntemi)**

----

## Random User Login Logs - Clustering
#### 📌 Proje Açıklaması
Bu projede, random kullanıcı giriş kayıtları verisi oluşturulmuş ve çeşitli analizler ile kümeleme yöntemleri uygulanmıştır.

- **Veri Seti Oluşumu**: Normal dağılıma sahip **rastgele kullanıcı giriş logları** oluşturuldu.
- **Veri Analizi ve Görselleştirme**: Özelliklerin dağılımları ve korelasyonlar incelenmiştir. Verinin daha iyi anlaşılması için **t-SNE tekniği ile görselleştirme** yapılmıştır.
- **Kümeleme ve Modelleme**: **Elbow yöntemi ile optimum küme sayısı** belirlenmiştir. Daha esnek bir model olan **Gaussian Mixture Model (GMM)** kullanılarak kümeleme gerçekleştirilmiştir. **Dendrogram** yöntemiyle kümeleme, ward yöntemi kullanılarak gerçekleştirilmiştir.

----

## Taxi-v3 Reinforcement Learning
![Taxi](https://www.gymlibrary.dev/_images/taxi.gif)
#### 📌 Proje Açıklaması
Bu proje, OpenAI Gym ortamındaki **Taxi-v3** problemi üzerinde **Q-learning** algoritması kullanılarak gerçekleştirilmiştir.🚖 Ortam hakkında detaylı bilgi için 👉
https://gymnasium.farama.org/environments/toy_text/taxi/

----


## 👗FashionMNIST | CNN + RMSprop + ImageDataGenerator
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/zalando-research/fashionmnist/data)
#### 📌 Proje Açıklaması

Bu proje, Fashion-MNIST veri setindeki görselleri 10 farklı giyim kategorisine sınıflandırmak için bir Convolutional Neural Network (CNN) uygulanmıştır.  
Genelleştirmeyi artırmak için `ImageDataGenerator` ile veri artırımı yapılmıştır. Model, `RMSprop` optimizer ile eğitilmiş, EarlyStopping ve ModelCheckpoint ile izlenmiş ve test setinde %89 doğruluk elde etmiştir.

##### 1- Data Preparation

- **Veri Seti:** Fashion-MNIST (Kaggle)  
- **Dosya Formatı:** UBYTE dosyaları, `idx2numpy` ile okundu  
- **Ortam & İndirme:** Google Colab + Kaggle API (`kaggle.json`)  
- **Ön İşleme & Görselleştirme:** Görseller normalize edildi ve örnekler gösterildi  
- **Data Augmentation:** Sadece training set üzerinde, %20 validation ayırarak `ImageDataGenerator` kullanıldı  
- **Generator:** `flow` ile **train_generator** ve **val_generator** oluşturuldu  


##### 2- CNN Architecture

- **Model Yapısı:**  
  - Feature Extraction : **Conv + ReLU + Pool + Dropout**   
  - Classification : **Flatten + Dense + ReLU + BatchNormalization + Dropout**   
  - Çıkış katmanında **Softmax** aktivasyonu kullanıldı  

- **Optimizer & Loss:**  
  - Optimizer: `RMSprop` (learning_rate=0.0001, decay=1e-6)  
  - Loss: `sparse_categorical_crossentropy`  
  - Metrics: `accuracy`  

- **Callbacks:** `EarlyStopping` ve `ModelCheckpoint`  
- **Epochs:** 60  


##### 3- 📊 Results

- Eğitim ve validasyon **accuracy** ile **loss** grafikleri :  

<img width="%50" height="%50" alt="image" src="https://github.com/user-attachments/assets/ce2a566d-e450-47e1-9101-a8ef04fe4bba" />

- 🧪 Evaluation
  - Best saved model was loaded  
  - Evaluated on the test set:  
    - **Accuracy:** 0.8937  
    - **Loss:** 0.3100
    - **Classification Report** and **Confusion Matrix** were calculated and visualized:
  
| Confusion Matrix |Classification Report |
|------------------|-----------|
| ![Confusion Matrix](https://github.com/user-attachments/assets/fe74f5f8-ef18-4b91-8c92-75f8d4f4e208) | ![Classification Report](https://github.com/user-attachments/assets/39536dfa-cac2-4a04-bd38-7ea553e359b3) |


----

## 🩺 Liver Cirrhosis Outcome Classification
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)](https://www.kaggle.com/competitions/ai-lab-turkiye-datathon-2025)

#### 📌 Proje Açıklaması

Bu proje, klinik, demografik ve laboratuvar özelliklerini kullanarak karaciğer sirozu hastalarının sonuçlarını (C / CL / D) tahmin etmeye yönelik bir **multi-class sınıflandırma** çalışmasıdır. Amaç, her hasta için üç sınıftan hangisine ait olduğuna dair **olasılık tahminleri** üretmektir; modelin çıktısı `Status_C`, `Status_CL` ve `Status_D` sütunlarını içerecektir. Veri seti, orijinal Cirrhosis Patient Survival Prediction verisinden türetilmiş ve AI ile üretilmiş örneklemeleri içermektedir.

#### 📌 Neden Önemli?

Siroz hastalarının yaşam süresi tahmininin doğru yapılması, sağlık profesyonelleri açısından kritik bir rol oynar:

- Acil müdahale gerektiren hastaların önceliklendirilmesi
- Hastaya uygun tedavi planının stratejik olarak belirlenmesi
- Sağlık kaynaklarının daha verimli yönetilmesi
- Hastaların genel bakım kalitesinin artırılması

#### Veri Seti
- Kaynak: AI tarafından oluşturulmuş Liver Cirrhosis dataset 
- Boyut: **20 features** and **35,000 rows**
- Dosyalar:
  - `train.csv` — özellikler + hedef (`Status`)  
  - `test.csv` — özellikler (submit için)  
  - `sample_submission.csv` — örnek gönderim formatı
    
### 🔍 Ek Veri Seti: [Original Cirrhosis Data](https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction)

Yarışma organizatörleri, verilen train ve test dosyalarının orijinal “Cirrhosis Patient Survival Prediction” veri setiyle eğitilmiş bir derin öğrenme modeli tarafından üretildiğini belirtiyor. 

Bu projede orijinal Cirrhosis veri setini ayrıca yükleyip sadece keşifsel analiz (EDA) ve ek testler için kullandım. 

#### Target Variable:
- `Status` — Üç sınıf:
  - **C**: Censored (hasta N_Days'de hayatta)  
  - **CL**: Hayatta, karaciğer nakli nedeniyle  
  - **D**: N_Days'de vefat etmiş

#### Evaluation
- Metric: **Multi-Class Log Loss** (kaggle-style multiclass log loss).  
- Gönderim formatı: `id,Status_C,Status_CL,Status_D` (olasılıklar).  

####  🛠️ Data Preprocessing

EDA aşamasında veri dağılımları incelendi ve eksik değerler tespit edildi. Ardından aşağıdaki veri ön işleme adımları uygulandı:

**1. Eksik Değer İşlemleri**

- **Sayısal değişkenler**: Train seti üzerinden **KNNImputer** ile tahmin edilerek dolduruldu.

- **Kategorik değişkenler**: Train setindeki **mode** tamamlandı.

**2. Korelasyon Analizi**

**3. Encoding İşlemleri**

- **One-Hot Encoding**: Hedef değişken hariç tüm kategorik kolonlar için uygulandı.

- **Label Encoding**: Hedef değişken *Status* → **C, CL, D** kategorileri sırasıyla **0, 1, 2** olarak kodlandı.

**4. Train–Test**

- **X_train**: Status ve id kolonları çıkarıldı.
- **y_train**: Status kolonu hedef olarak alındı.
- **X_test**: Sadece id kolonu çıkarıldı.

**5. Scaling**

Logistic Regression, KNN, SVC ve MLP gibi ölçeklemeye duyarlı modeller için **StandardScaler** kullanıldı.
Hem ölçekli hem ölçeklenmemiş versiyonlar oluşturularak model gereksinimlerine göre kullanıldı.

**6. Sınıf Dengesizliği – SMOTE**

Hedef değişkenin dengesiz yapısı nedeniyle **SMOTE** uygulanarak azınlık sınıflar için sentetik örnekler üretildi ve sınıf dağılımı dengelendi. Smote işlemi sonucunda veri boyutları:
  - X_train: (30390, 19)
  - X_test: (30390, 19)
  - y_train: (30390,)


<h4 align="center">SMOTE Uygulaması: Öncesi ve Sonrası</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/620b99ce-f9b8-4f44-ab56-f7e1762abf2a" alt="smote-oncesi" width="45%" />
  <img src="https://github.com/user-attachments/assets/aa07e83e-3a33-402b-b517-e82b8e1bca71" alt="smote-sonrasi" width="45%" />
</p>


#### Modeling

**📐 Model Değerlendirme (K-Fold)**

Tüm modeller, hedef etiketleri test setinde bulunmadığı için **5-fold cross-validation** ile train seti üzerinden değerlendirildi.
**Accuracy, precision, recall ve log loss metrikleri** hesaplandı.

En iyi temel sonuçları veren modeller:

- **XGBoost**
- **LightGBM**
- **RandomForest**

Yarışmada kullanılan metrik olan **multi-class log loss** dikkate alınarak en başarılı iki model seçildi:

**📊 K-Fold Sonuçları (XGBoost & LightGBM)**

| Model     | Accuracy | Precision | Recall  | LogLoss  |
|-----------|----------|-----------|---------|----------|
| XGBoost   | 0.918230 | 0.918020  | 0.918230 | 0.226986 |
| LightGBM  | 0.907799 | 0.907690  | 0.907799 | 0.263790 |


**🎯 Hiperparametre Optimizasyonu**

Her iki model için de **RandomizedSearchCV** ile
**5-fold CV + neg_log_loss** kullanılarak hiperparametre araması yapıldı.

**🏆 En Başarılı Model**

**RandomizedSearchCV** ile yapılan hiperparametre araması sonucunda **en düşük log loss** değerine sahip model:

| Model    | CV LogLoss |
|----------|------------|
| LightGBM | 0.218144   |

> **LightGBM, sınıf dengesizliği ve özellik çeşitliliği göz önünde bulundurulduğunda, hiperparametre optimizasyonu sonrası en iyi performansı göstermiştir.**

#### 🏁 Final Model ve Submission

Tüm hiperparametre optimizasyonu sonrası **LGBMClassifier**, en iyi log loss değerini elde etti.  

Optimizasyonlu parametrelerle **final model** (`best_model_final`) eğitildi, değerlendirildi ve yarışma gönderimi için kaydedildi.  
Bu model ile test seti üzerinde **submission dosyası** oluşturuldu.

##### 📤 Yarışma Gönderimleri

- **Notebook V1:** Daha yüksek leaderboard skoru (LogLoss = 0.368) elde etti ve resmi gönderim olarak kullanıldı.  
- **Notebook V4 (bu sürüm):** Sınıf dengesizliği ve daha sağlam modelleme stratejilerini içerir. **Leaderboard Private skoru: LogLoss = 0.397.**


#### 📊 Model Değerlendirmesi

Final model, farklı veri setleri üzerinde değerlendirildi. Gerçek test etiketleri bulunmasa da **LogLoss, Accuracy, Classification Report, Confusion Matrix** metrikleri incelendi:

- **Train verisi**
- **SMOTE ile dengelenmiş veri**:
- **Orijinal Cirrhosis veri seti**: Preprocessing aşamaları uygulandıktan sonra değerlendirme gerçekleştirilmiştir.

> Bu değerlendirmeler, modelin farklı veri senaryolarında ne kadar sağlam ve genellenebilir olduğunu incelemek için yapıldı.

#### Requirements

Bu proje Kaggle ortamında gerçekleştirilmiştir.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn joblib
 ```
veya

```bash
pip install -r requirements.txt
 ```

----


