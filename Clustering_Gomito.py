import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Carichiamo i dati (fillna evita che i valori nulli blocchino tutto)
df = pd.read_csv('netflix_finale.csv').fillna('None')

# 2. Creiamo un testo combinato (BK + Descrizione)
# Questo assicura che il vocabolario NON sia mai vuoto
df['text_for_clustering'] = df['background_knowledge'].astype(str) + " " + df['description'].astype(str)

# Usiamo parametri più permissivi per il vectorizer
vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=1)
X_bk = vectorizer.fit_transform(df['text_for_clustering']).toarray()

# 3. Metodo del Gomito
distortions = []
K = range(1, 11)
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeanModel.fit(X_bk)
    distortions.append(kmeanModel.inertia_)

# 4. Creazione del Grafico
plt.figure(figsize=(10, 6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Numero di Cluster (k)')
plt.ylabel('Inertia (Somma dei quadrati delle distanze)')
plt.title('Metodo del Gomito (Dataset Arricchito)')
plt.grid(True)
plt.show()