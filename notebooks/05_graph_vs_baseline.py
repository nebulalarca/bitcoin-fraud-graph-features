import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, classification_report
import xgboost as xgb

print(" Veriler yükleniyor...")
features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
classes = pd.read_csv("data/elliptic_txs_classes.csv")
edges = pd.read_csv("data/elliptic_txs_edgelist.csv")

features.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, 166)]
classes['label'] = classes['class'].map({'1': 1, '2': 0, 'unknown': -1})

merged = pd.merge(features, classes[['txId', 'label']], on='txId')
labeled_df = merged[merged['label'] != -1].copy()

print(" Graf kuruluyor ve özellikler çıkarılıyor...")
G = nx.from_pandas_edgelist(edges, source='txId1', target='txId2', create_using=nx.DiGraph())
degree_dict = dict(G.degree())
pagerank_dict = nx.pagerank(G)

label_dict = dict(zip(labeled_df['txId'], labeled_df['label']))

def neighborhood_illicit_ratio(node):
    neighbors = list(G.predecessors(node)) + list(G.successors(node))
    known = [n for n in neighbors if n in label_dict]
    if len(known) == 0:
        return 0.0
    return sum(label_dict[n] for n in known) / len(known)

graph_features = pd.DataFrame({
    'txId': list(G.nodes()),
    'degree': [degree_dict[n] for n in G.nodes()],
    'pagerank': [pagerank_dict[n] for n in G.nodes()],
})
graph_features['neighborhood_illicit_ratio'] = graph_features['txId'].apply(neighborhood_illicit_ratio)

labeled_df = labeled_df.merge(graph_features, on='txId', how='left')

# Aynı temporal split, 04_temporal_split.py ile birebir aynı mantık
min_step = labeled_df['time_step'].min()
max_step = labeled_df['time_step'].max()
split_step = int(max_step * 0.70)

train_df = labeled_df[labeled_df['time_step'] <= split_step]
test_df = labeled_df[labeled_df['time_step'] > split_step]

baseline_cols = [c for c in labeled_df.columns if c.startswith('feat_')]
graph_cols = baseline_cols + ['degree', 'pagerank', 'neighborhood_illicit_ratio']

def train_and_eval(feature_cols, label):
    X_train, y_train = train_df[feature_cols], train_df['label'].astype(int)
    X_test, y_test = test_df[feature_cols], test_df['label'].astype(int)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, random_state=42,
        eval_metric='logloss', n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n=== {label} ===")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    return pr_auc, roc_auc, y_test, y_proba

print("\n Baseline model (sadece 165 ham özellik) eğitiliyor...")
baseline_pr, baseline_roc, baseline_ytest, baseline_yproba = train_and_eval(baseline_cols, "BASELINE (ham özellikler)")

print("\n Graph-enhanced model (ham özellikler + graf özellikleri) eğitiliyor...")
graph_pr, graph_roc, graph_ytest, graph_yproba = train_and_eval(graph_cols, "GRAPH-ENHANCED (ham + graf özellikleri)")

print("\n" + "="*50)
print(" SONUÇ KARŞILAŞTIRMASI")
print("="*50)
print(f"Baseline PR-AUC        : {baseline_pr:.4f}")
print(f"Graph-enhanced PR-AUC  : {graph_pr:.4f}")
print(f"Fark (Δ)               : {graph_pr - baseline_pr:+.4f}")

print("\n İstatistiksel anlamlılık testi (bootstrap)...")

def bootstrap_pr_auc(y_test, y_proba, n_iterations=1000):
    scores = []
    y_test_arr = y_test.values
    rng = np.random.RandomState(42)
    for _ in range(n_iterations):
        idx = rng.randint(0, len(y_test_arr), len(y_test_arr))
        if len(np.unique(y_test_arr[idx])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_test_arr[idx], y_proba[idx])
        scores.append(auc(recall, precision))
    return np.array(scores)

# Not: iki modelin y_proba'sını da saklayıp burada kullanmamız gerekiyor,
# bunun için train_and_eval fonksiyonunun sonuna y_proba'yı da return edelim

print("\n Bootstrap ile PR-AUC güven aralığı hesaplanıyor (1000 iterasyon)...")

def bootstrap_pr_auc(y_test, y_proba, n_iterations=1000, seed=42):
    y_test_arr = np.asarray(y_test)
    scores = []
    rng = np.random.RandomState(seed)
    n = len(y_test_arr)
    for _ in range(n_iterations):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_test_arr[idx])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_test_arr[idx], y_proba[idx])
        scores.append(auc(recall, precision))
    return np.array(scores)

baseline_boot = bootstrap_pr_auc(baseline_ytest, baseline_yproba)
graph_boot = bootstrap_pr_auc(graph_ytest, graph_yproba)

print(f"Baseline PR-AUC       : {baseline_boot.mean():.4f}  (%95 GA: {np.percentile(baseline_boot, 2.5):.4f} - {np.percentile(baseline_boot, 97.5):.4f})")
print(f"Graph-enhanced PR-AUC : {graph_boot.mean():.4f}  (%95 GA: {np.percentile(graph_boot, 2.5):.4f} - {np.percentile(graph_boot, 97.5):.4f})")

diff = graph_boot - baseline_boot
print(f"\nFark (graph - baseline) ortalaması: {diff.mean():+.4f}")
print(f"Farkın %95 güven aralığı           : ({np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f})")
if np.percentile(diff, 2.5) > 0:
    print(" Güven aralığı 0'ı içermiyor → fark istatistiksel olarak anlamlı")
else:
    print(" Güven aralığı 0'ı içeriyor → fark anlamlı olmayabilir")