import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

def analyze_shap():
    print("🚀 Veri ve model hazırlanıyor...")
    features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv("data/elliptic_txs_classes.csv")

    features.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, 166)]
    classes['label'] = classes['class'].map({'1': 1, '2': 0, 'unknown': -1})

    merged = pd.merge(features, classes[['txId', 'label']], on='txId')
    labeled_df = merged[merged['label'] != -1].copy()

    X = labeled_df.drop(columns=['txId', 'label'])
    y = labeled_df['label'].astype(int)

    # Modeli eğit
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X, y)

    print("📊 SHAP Değerleri hesaplanıyor (Bu işlem birkaç saniye sürebilir)...")
    # TreeExplainer kullanarak SHAP analizini başlat
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # 1. SHAP Feature Importance Özeti
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title("Bitcoin Fraud Detection - En Önemli 15 Özellik (SHAP)", fontsize=12)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300)
    plt.close()

    print("🖼️ SHAP özeti 'shap_summary.png' olarak kaydedildi!")

if __name__ == "__main__":
    analyze_shap()