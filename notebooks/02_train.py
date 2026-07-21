import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve, 
    auc, 
    roc_auc_score
)
import xgboost as xgb

def run_pipeline():
    print("🚀 Veriler yükleniyor...")
    # 1. Veri Yükleme (Kendi dosya yollarına göre güncelleyebilirsin)
    features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv("data/elliptic_txs_classes.csv")

    # Kolon İsimlendirmeleri
    features.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, 166)]
    classes['label'] = classes['class'].map({'1': 1, '2': 0, 'unknown': -1})

    # 2. Sadece Etiketli Verileri Filtrele (0: Legitimate, 1: Fraud)
    merged = pd.merge(features, classes[['txId', 'label']], on='txId')
    labeled_df = merged[merged['label'] != -1].copy()

    X = labeled_df.drop(columns=['txId', 'label'])
    y = labeled_df['label'].astype(int)

    print(f"📊 Toplam Etiketli Veri: {len(labeled_df)} | Fraud (1): {sum(y==1)} | Normal (0): {sum(y==0)}")

    # 3. Train / Test Ayrımı (Stratify ile %10'luk Fraud oranını koruyoruz)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Class Imbalance Oranı (scale_pos_weight)
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos
    print(f"⚖️ Hesaplanan scale_pos_weight Oranı: {scale_pos_weight:.2f}")

    # 5. XGBoost Model Tanımı ve Eğitimi
    print("\n🌲 XGBoost Modeli Eğitiliyor...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, # Dengesizlik çözümü
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 6. Tahminler
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Fraud olma olasılığı (0 ile 1 arası)

    # 7. Performans Metrikleri & PR-AUC Hesaplama
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "="*50)
    print("🎯 MODEL DEĞERLENDİRME SONUÇLARI")
    print("="*50)
    print(f"📈 PR-AUC Skoru  : {pr_auc:.4f}  (En Önemli Metrik)")
    print(f"📈 ROC-AUC Skoru : {roc_auc:.4f}")
    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraud (1)']))

    # 8. Karmaşıklık Matrisi (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    print("--- Confusion Matrix ---")
    print(f"True Negative (Başarılı Normal)  : {cm[0][0]}")
    print(f"False Positive (Yanlış Alarm)    : {cm[0][1]}")
    print(f"False Negative (Kaçan Fraud!)    : {cm[1][0]}")
    print(f"True Positive (Yakalana Fraud)   : {cm[1][1]}")

    # 9. Precision-Recall Eğrisi Çizimi
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f'XGBoost (PR-AUC = {pr_auc:.3f})', color='darkorange', lw=2)
    plt.xlabel('Recall (Duyarlılık)')
    plt.ylabel('Precision (Kesinlik)')
    plt.title('Precision-Recall Curve (Bitcoin Fraud Detection)')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('pr_curve.png')
    print("\n🖼️ Precision-Recall grafiği 'pr_curve.png' olarak kaydedildi.")

if __name__ == "__main__":
    run_pipeline()