import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Caricamento e preparazione (2500 righe o quelle che hai nel file finale)
df = pd.read_csv('netflix_finale.csv').fillna('None')
y = df['type']

X_base = pd.get_dummies(df[['release_year', 'country']])
X_base.columns = X_base.columns.astype(str)

df['text_combined'] = df['background_knowledge'].astype(str) + " " + df['description'].astype(str)
vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
X_text = vectorizer.fit_transform(df['text_combined']).toarray()
X_enriched = pd.concat([X_base.reset_index(drop=True), pd.DataFrame(X_text)], axis=1)
X_enriched.columns = X_enriched.columns.astype(str)

# 2. Griglie di Iperparametri (Hyperparameter Tuning)
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'criterion': ['gini', 'entropy']
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

modelli_nested = [
    ("Random Forest", RandomForestClassifier(random_state=42), param_grid_rf),
    ("SVM", SVC(random_state=42), param_grid_svm)
]

# Definiamo la lista delle metriche che vogliamo calcolare
# 'macro' calcola la metrica per ogni classe e ne fa la media (ideale per classi sbilanciate)
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

print("Avvio Nested Cross-Validation con metriche multiple...")
print("-" * 65)
print(f"{'Modello':<15} | {'Metrica':<12} | {'Media ± Dev.Std'}")
print("-" * 65)

for nome, modello, params in modelli_nested:
    # Inner Loop: GridSearchCV (trova i parametri migliori)
    clf_search = GridSearchCV(estimator=modello, param_grid=params, cv=3)

    # Outer Loop: cross_validate (valuta la stabilità con metriche multiple)
    cv_results = cross_validate(clf_search, X_enriched, y, cv=5, scoring=scoring)

    for metrica in scoring:
        mean_score = cv_results[f'test_{metrica}'].mean()
        std_score = cv_results[f'test_{metrica}'].std()

        # Puliamo il nome della metrica per la stampa
        metrica_label = metrica.replace('_macro', '').capitalize()

        print(f"{nome:<15} | {metrica_label:<12} | {mean_score:.4f} ± {std_score:.4f}")
    print("-" * 65)