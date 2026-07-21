import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve, 
    auc, 
    roc_auc_score
)
import xgboost as xgb

def run_temporal_pipeline():
    print("🚀 Veriler yükleniyor...")
    features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv("data/elliptic_txs_classes.csv")

    features.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, 166)]
    classes['label'] = classes['class'].map({'1': 1, '2': 0, 'unknown': -1})

    merged = pd.merge(features, classes[['txId', 'label']], on='txId')
    labeled_df = merged[merged['label'] != -1].copy()

    # Time Step Analizi (Elliptic veri setinde 1'den 49'a kadar adım vardır)
    min_step = labeled_df['time_step'].min()
    max_step = labeled_df['time_step'].max()
    print(f"⏱️ Toplam Zaman Adımları: {min_step} -> {max_step}")

    # %70 Train / %30 Test Eşiği (Yaklaşık 34. adım sınır çizgisidir)
    split_step = int(max_step * 0.70)
    print(f"✂️ Zaman Bazlı Bölme Noktası: Step 1-{split_step} (Train) | Step {split_step+1}-{max_step} (Test)")

    # Zaman Bazlı Ayrım
    train_df = labeled_df[labeled_df['time_step'] <= split_step]
    test_df = labeled_df[labeled_df['time_step'] > split_step]

    X_train = train_df.drop(columns=['txId', 'label'])
    y_train = train_df['label'].astype(int)

    X_test = test_df.drop(columns=['txId', 'label'])
    y_test = test_df['label'].astype(int)

    print(f"📈 Train Seti Boyutu: {len(train_df)} (Fraud: {sum(y_train==1)})")
    print(f"📉 Test Seti Boyutu : {len(test_df)} (Fraud: {sum(y_test==1)})")

    # Train seti üzerinden scale_pos_weight hesaplama
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ Train kümesine göre scale_pos_weight: {scale_pos_weight:.2f}")

    # Modeli Eğit
    print("\n🌲 XGBoost Modeli Zaman Bazlı Eğitiliyor...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Gelecekteki Adımları Tahmin Et
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrikler
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "="*50)
    print("⏳ ZAMAN BAZLI (TEMPORAL) MODEL DEĞERLENDİRME SONUÇLARI")
    print("="*50)
    print(f"📈 PR-AUC Skoru  : {pr_auc:.4f}")
    print(f"📈 ROC-AUC Skoru : {roc_auc:.4f}")
    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraud (1)']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("--- Confusion Matrix ---")
    print(f"True Negative (Başarılı Normal)  : {cm[0][0]}")
    print(f"False Positive (Yanlış Alarm)    : {cm[0][1]}")
    print(f"False Negative (Kaçan Fraud!)    : {cm[1][0]}")
    print(f"True Positive (Yakalana Fraud)   : {cm[1][1]}")

if __name__ == "__main__":
    run_temporal_pipeline()