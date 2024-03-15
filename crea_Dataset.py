import pandas as pd


def merge_movies_data(final_df):
    # Carica il dataset dei film
    df_movies = pd.read_csv("data/movies.csv")

    # Converti la colonna 'id' nel DataFrame dei film in tipo di dato stringa
    df_movies['id'] = df_movies['id'].astype(str)

    # Seleziona solo le colonne necessarie dal DataFrame dei film
    df_movies = df_movies[['id', 'date', 'minute', 'rating']]

    # Merge dei dati dei film con il DataFrame finale basato sull'ID
    final_df = final_df.merge(df_movies, on='id', how='left')

    return final_df


# Carica i due dataset CSV
df_genre = pd.read_csv("data/genres.csv")
df_themes = pd.read_csv("data/themes.csv")

# Converti la colonna 'id' in entrambi i dataframe in tipo di dato stringa
df_genre['id'] = df_genre['id'].astype(str)
df_themes['id'] = df_themes['id'].astype(str)

# Unisci i due dataframe su 'id'
merged_df = pd.merge(df_genre, df_themes, on='id', how='inner')

# Utilizza pd.pivot_table() per creare una tabella pivot sui temi
pivot_table = pd.pivot_table(merged_df, index='id', columns='theme', aggfunc=lambda x: 1, fill_value=0)

# Rimuovi il nome della colonna dall'indice della tabella pivot
pivot_table.columns = pivot_table.columns.droplevel()

# Converti i valori della tabella pivot in interi
pivot_table = pivot_table.astype(int)

# Converti i valori in stringhe
pivot_table = pivot_table.astype(str)

# Unisci la tabella pivot con il dataframe dei generi
final_df = df_genre.merge(pivot_table, on='id', how='left')

# Utilizzo della funzione per aggiungere i dati dei film al DataFrame finale
final_df = merge_movies_data(final_df)

# Salva il DataFrame finale in un nuovo file CSV
final_df.to_csv("data/nuovo_dataset.csv", index=False)
