import pandas as pd

# Carichiamo il file originale
df = pd.read_csv('netflix_titles_nov_2019.csv')

# Prendiamo le prime 100 righe
df_mini = df.head(100)

# Salviamo il file piccolo
df_mini.to_csv('netflix_mini.csv', index=False)
print("File 'netflix_mini.csv' creato correttamente!")