import pandas as pd

num_features = 165
feature_cols = ['txId', 'time_step'] + [f'feat_{i}' for i in range(num_features)]

features = pd.read_csv('data/elliptic_txs_features.csv', header=None, names=feature_cols)
classes = pd.read_csv('data/elliptic_txs_classes.csv')
edges = pd.read_csv('data/elliptic_txs_edgelist.csv')

print("features shape:", features.shape)
print("classes shape:", classes.shape)
print("edges shape:", edges.shape)

print("\n--- Sınıf dağılımı ---")
print(classes['class'].value_counts())
print("\nYüzde olarak:")
print(classes['class'].value_counts(normalize=True) * 100)

print("\n--- Veriyi birleştir ve filtrele ---")

# class sütununu okunabilir hale getir: 1=illicit, 0=licit, etiketsizleri at
classes_clean = classes[classes['class'] != 'unknown'].copy()
classes_clean['label'] = classes_clean['class'].map({'1': 1, '2': 0})

# features ile birleştir (merge)
merged = features.merge(classes_clean[['txId', 'label']], on='txId', how='inner')

print("Etiketli veri boyutu:", merged.shape)
print("\nHedef değişken dağılımı:")
print(merged['label'].value_counts())
print(merged['label'].value_counts(normalize=True) * 100)