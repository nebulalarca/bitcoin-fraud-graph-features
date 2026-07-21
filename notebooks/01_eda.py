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

import networkx as nx

print("\n--- Graf kur ve özellik çıkar ---")

G = nx.from_pandas_edgelist(edges, source='txId1', target='txId2', create_using=nx.DiGraph())
print("Graf düğüm sayısı:", G.number_of_nodes())
print("Graf kenar sayısı:", G.number_of_edges())

# Derece (in + out)
degree_dict = dict(G.degree())

# PageRank
pagerank_dict = nx.pagerank(G)

print("\nÖrnek 5 düğümün derecesi:")
print(list(degree_dict.items())[:5])
print("\n--- Graf özelliklerini tabloya dök ---")

# txId -> label sözlüğü (komşuluk illicit oranını hesaplamak için)
label_dict = dict(zip(merged['txId'], merged['label']))

def neighborhood_illicit_ratio(node):
    neighbors = list(G.predecessors(node)) + list(G.successors(node))
    known_neighbors = [n for n in neighbors if n in label_dict]
    if len(known_neighbors) == 0:
        return 0.0
    illicit_count = sum(label_dict[n] for n in known_neighbors)
    return illicit_count / len(known_neighbors)

graph_features = pd.DataFrame({
    'txId': list(G.nodes()),
    'degree': [degree_dict[n] for n in G.nodes()],
    'pagerank': [pagerank_dict[n] for n in G.nodes()],
})
graph_features['neighborhood_illicit_ratio'] = graph_features['txId'].apply(neighborhood_illicit_ratio)

print("Graf özellikleri tablosu boyutu:", graph_features.shape)
print(graph_features.head())

# Etiketli veriyle birleştir
merged_with_graph = merged.merge(graph_features, on='txId', how='left')
print("\nGraf özellikli birleşik tablo boyutu:", merged_with_graph.shape)
print(merged_with_graph[['txId', 'degree', 'pagerank', 'neighborhood_illicit_ratio', 'label']].head())