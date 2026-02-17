import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Caricamento dati
df = pd.read_csv('netflix_finale.csv').fillna('None')
y = df['type']

# 2. Ingegneria delle Feature
X_base = pd.get_dummies(df[['release_year', 'country']])
X_base.columns = X_base.columns.astype(str)

df['text_combined'] = df['background_knowledge'].astype(str) + " " + df['description'].astype(str)
vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
X_text = vectorizer.fit_transform(df['text_combined']).toarray()

X_enriched = pd.concat([X_base.reset_index(drop=True), pd.DataFrame(X_text)], axis=1)
X_enriched.columns = X_enriched.columns.astype(str)

# 3. Definizione Modelli e Metriche
modelli = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "SVM (Linear)": SVC(kernel='linear', random_state=42),
    "Naive Bayes": GaussianNB()
}

metriche = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

print(f"{'Modello':<15} | {'Metrica':<12} | {'Base (Avg)':<12} | {'Enriched (Avg)':<15} | {'Delta (%)'}")
print("-" * 85)

for nome, clf in modelli.items():
    res_base = cross_validate(clf, X_base, y, cv=5, scoring=metriche)
    res_enriched = cross_validate(clf, X_enriched, y, cv=5, scoring=metriche)

    for m in metriche:
        m_name = m.replace('_macro', '').capitalize()

        base_m = res_base[f'test_{m}'].mean()
        enr_m = res_enriched[f'test_{m}'].mean()

        # Calcolo dell'aumento o decremento
        delta = enr_m - base_m

        # Formattazione del delta con segno
        delta_str = f"{delta:+.2f}"

        print(f"{nome:<15} | {m_name:<12} | {base_m:<12.2f} | {enr_m:<15.2f} | {delta_str}")
    print("-" * 85)