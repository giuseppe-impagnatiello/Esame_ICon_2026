# Progetto ICon: Netflix Content Classification with Semantic Enrichment

Questo progetto per l'esame di **Ingegneria della Conoscenza** analizza come l'integrazione di **Background Knowledge (BK)** dal Web Semantico possa potenziare un sistema di Machine Learning.

## Obiettivo
Effettuare una classificazione binaria per distinguere tra **Movie** e **TV Show** utilizzando dati originali (Kaggle) arricchiti con descrizioni enciclopediche estratte da **DBpedia** via **SPARQL**.

## Risultati Sperimentali (Media su 5-fold CV)
| Modello | Accuracy (Base) | Accuracy (Enriched) | Delta |
| :--- | :--- | :--- | :--- |
| **SVM (Linear)** | 0.70 | **0.72** | **+0.02** |
| **Random Forest** | 0.71 | 0.71 | +0.00 |
| **Naive Bayes** | 0.67 | 0.63 | -0.04 |

*L'arricchimento semantico ha mostrato un miglioramento tangibile nel modello SVM, mentre ha aumentato la robustezza statistica (riduzione della varianza) nel Random Forest.*

## Tecnologie utilizzate
- **Python**: Pandas, Scikit-learn
- **Semantic Web**: SPARQLWrapper, DBpedia Ontology
- **NLP**: TF-IDF Vectorization
