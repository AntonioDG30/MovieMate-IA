import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from deep_translator import GoogleTranslator
from difflib import get_close_matches

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Carica il dataset dei film
df = pd.read_csv("data/nuovo_dataset.csv")

# Carica il dataset con i nomi e le descrizioni dei film
movies_df = pd.read_csv("data/movies.csv")

# Rimuovi eventuali righe con valori mancanti
df.dropna(inplace=True)

# Dizionario dei 10 temi più comuni per ogni genere
top_themes_per_genre = {}
for genre in df['genre'].unique():
    genre_df = df[df['genre'] == genre].drop(
        columns=['id', 'genre', 'minute', 'rating'])  # Rimuovi le colonne aggiunte
    top_themes = genre_df.sum().nlargest(10).index.tolist()
    top_themes_per_genre[genre] = top_themes


# Funzione per ottenere raccomandazioni di film
def recommend_movies(X):
    print("Benvenuto al nostro servizio di consigli sui film!")
    while True:
        response = input("Sei interessato a un film? (si/no): ").lower()
        if response != 'si':
            print("Grazie per aver utilizzato il nostro servizio!")
            break
        else:
            user_input_genre = input("Per favore, inserisci il tuo genere preferito di film: ")
            translated_genre = get_correct_genre(user_input_genre)
            if not translated_genre:
                continue

            genre_top_themes = top_themes_per_genre[translated_genre]
            translated_genre_top_themes = translate_themes(genre_top_themes)

            user_theme_responses = {}
            for theme in translated_genre_top_themes:
                while True:
                    user_response = input(
                        f"Ti piace il tema '{translated_genre_top_themes[theme]}'? (sì/si/no): ").lower()
                    if user_response in ['sì', 'si', 'yes', 'no']:
                        user_theme_responses[theme] = 1 if user_response in ['sì', 'si', 'yes'] else 0
                        break
                    else:
                        print("Risposta non valida. Si prega di rispondere con 'sì' o 'no'.")

            # Nuove domande aggiunte
            # Aggiunta delle nuove colonne per data, minute e rating
            user_input_max_duration = int(input("Inserisci la durata massima del film in minuti: "))
            user_input_min_rating = float(input("Inserisci il rating minimo desiderato: "))

            user_input_features = pd.DataFrame(columns=X.columns)
            for col in user_input_features.columns:
                if col in user_theme_responses:
                    user_input_features[col] = [user_theme_responses[col]]
                else:
                    user_input_features[col] = [0]

            prediction = model.predict(user_input_features)
            recommended_movies_ids = df[(df['genre'] == label_encoder.inverse_transform(prediction).ravel()[0]) &
                                        (df[genre_top_themes].sum(axis=1) >= len(genre_top_themes) // 2) &
                                        (df['minute'] <= user_input_max_duration) &
                                        (df['rating'] >= user_input_min_rating)]['id'].tolist()

            recommended_movies_info = movies_df[movies_df['id'].isin(recommended_movies_ids)][
                ['name', 'description', 'minute']].head(3)
            if not recommended_movies_info.empty:
                print("Ti consigliamo i seguenti film:")
                print(recommended_movies_info)
            else:
                print(
                    "Ci dispiace, non abbiamo raccomandazioni per questo genere o i tuoi criteri di selezione.")
            print()


# Funzione per tradurre i temi in italiano
def translate_themes(themes):
    translated_themes = {}
    for theme in themes:
        translated_theme = GoogleTranslator(source='auto', target='it').translate(theme)
        translated_themes[theme] = translated_theme
    return translated_themes


# Funzione per tradurre il genere in inglese solo per confronto
def translate_genre(genre):
    translated_genre = GoogleTranslator(source='auto', target='en').translate(genre)
    return translated_genre

def translate_genre2(genres):
    translated_genres = []
    for genre in genres:
        translated_genre = GoogleTranslator(source='auto', target='it').translate(genre)
        translated_genres.append(translated_genre)
    return translated_genres


# Funzione per suggerire generi simili
def suggest_similar_genre(genre):
    # Generi disponibili nel dataset
    genres = list(top_themes_per_genre.keys())

    # Cerca generi simili
    similar_genres = get_close_matches(genre, genres)
    translated_similar_genres = translate_genre2(similar_genres)
    return translated_similar_genres



# Funzione per ottenere il genere corretto
def get_correct_genre(genre):
    while True:
        translated_genre = translate_genre(genre)
        if translated_genre in top_themes_per_genre:
            return translated_genre
        else:
            # Suggerisci generi simili
            similar_genres = suggest_similar_genre(genre)
            if similar_genres:
                print(f"Il genere '{genre}' non è valido. Forse intendevi: {', '.join(similar_genres)}")
            else:
                print("Il genere inserito non è valido e non sono stati trovati generi simili.")
            genre = input("Per favore, inserisci un genere valido: ")



# Dividi il dataset in variabili indipendenti (X) e variabile dipendente (y)
X = df.drop(columns=['id', 'genre'])
y = df['genre']

# Codifica le etichette di genere
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividi il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utilizza l'imputatore semplice per gestire eventuali valori mancanti nelle caratteristiche
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Crea il modello di classificazione con un decision tree
model = make_pipeline(DecisionTreeClassifier())

# Addestra il modello
model.fit(X_train_imputed, y_train)

# Esegui il chatbot
recommend_movies(X)
