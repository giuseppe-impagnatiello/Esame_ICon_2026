import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import time


def query_dbpedia(title):
    # Ci colleghiamo all'endpoint ufficiale di DBpedia
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    # Query SPARQL: cerchiamo la descrizione (abstract) del film
    # Usiamo rdfs:label per far combaciare il titolo del CSV con quello di DBpedia
    query = f"""
    SELECT ?abstract WHERE {{
      ?film rdfs:label "{title}"@en ;
            a dbo:Film ;
            dbo:abstract ?abstract .
      FILTER (LANG(?abstract) = "en")
    }} LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        if results["results"]["bindings"]:
            return results["results"]["bindings"][0]["abstract"]["value"]
    except Exception as e:
        return None
    return None


# Carichiamo il dataset mini
df = pd.read_csv('netflix_mini.csv')

print("Inizio arricchimento dati da DBpedia... attendere qualche istante.")
# Creiamo la nuova colonna 'background_knowledge'
df['background_knowledge'] = df['title'].apply(query_dbpedia)

# Salviamo il risultato finale
df.to_csv('netflix_finale.csv', index=False)
print("Processo completato! Il file 'netflix_finale.csv' contiene ora la conoscenza extra.")