import warnings
import pandas as pd
import random
import matplotlib.pyplot as plt  # Importa matplotlib per la visualizzazione grafica
import numpy as np  # Importa NumPy per la gestione di array e operazioni matematiche
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
    print("MovieMate-IA: Benvenuto in MovieMate-IA, il tuo suggeritore personale di film!")
    while True:
        print("MovieMate-IA: Vuoi un consiglio per un nuovo film? (si/no): ")
        response = input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: ").lower()
        if response != 'si':
            print("MovieMate-IA: Grazie per aver utilizzato il nostro servizio!")
            break
        else:
            print("MovieMate-IA: Per favore, inserisci il tuo genere di film che vorresti vedere: ")
            user_input_genre = input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: ")
            translated_genre = get_correct_genre(user_input_genre)
            if not translated_genre:
                continue

            genre_top_themes = top_themes_per_genre[translated_genre]
            translated_genre_top_themes = translate_themes(genre_top_themes)

            user_theme_responses = {}
            for theme in translated_genre_top_themes:
                while True:
                    print(f"MovieMate-IA: Ti piace il tema '{translated_genre_top_themes[theme]}'? (si/no): ")
                    user_response = input(
                        "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: ").lower()
                    if user_response in ['sì', 'si', 'yes', 'no']:
                        user_theme_responses[theme] = 1 if user_response in ['sì', 'si', 'yes'] else 0
                        break
                    else:
                        print("MovieMate-IA: Risposta non valida. Si prega di rispondere con 'sì' o 'no'.")

            # Nuove domande aggiunte
            # Aggiunta delle nuove colonne per data, minute e rating
            print("MovieMate-IA: Inserisci la durata massima del film in minuti: ")
            user_input_max_duration = int(input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: "))
            print("MovieMate-IA: Inserisci il rating minimo desiderato: ")
            user_input_min_rating = float(input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: "))

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
                ['name', 'description', 'minute']]
            if not recommended_movies_info.empty:
                print("MovieMate-IA: Ti consigliamo i seguenti film:\n")
                # Imposta uno stato casuale diverso ad ogni esecuzione
                random_state = random.randint(1, 1000)
                # Seleziona casualmente 3 film dall'elenco raccomandato
                random_movies = recommended_movies_info.sample(n=min(3, len(recommended_movies_info)),
                                                               random_state=random_state)
                for idx, movie in random_movies.iterrows():
                    translated_description = GoogleTranslator(source='auto', target='it').translate(movie['description'])
                    formatted_info = format_movie_info(movie, translated_description)
                    print(formatted_info)
                    print("-" * 50)  # Linea divisoria tra i film consigliati
            else:
                print(
                    "MovieMate-IA: Ci dispiace, non abbiamo raccomandazioni per questo genere o i tuoi criteri di selezione.")
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
                print(
                    f"MovieMate-IA: Il genere '{genre}' non è valido. Forse intendevi: {', '.join(similar_genres)}")
            else:
                print(
                    "MovieMate-IA: Il genere inserito non è valido e non sono stati trovati generi simili.")
                print("MovieMate-IA: Per favore, inserisci un genere valido: ")
            genre = input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: ")

# Funzione per formattare le informazioni di un film
def format_movie_info(movie, description):
    words = description.split()
    segmented_description = [words[i:i+12] for i in range(0, len(words), 12)]
    formatted_description = '\n'.join([' '.join(segment) for segment in segmented_description])
    return f"Nome: {movie['name']}\nDescrizione:\n{formatted_description}\nDurata: {movie['minute']} minuti\n"

# Funzione per la scelta del modello di classificazione
def choose_model():
    while True:
        print("MovieMate-IA: Quale modello di classificazione desideri utilizzare? (Decision Tree/Random Forest/Naive Bayes/GBM): ")
        model_choice = input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: ").lower()
        if model_choice in ['decision tree', 'random forest', 'naive bayes', 'gbm']:
            return model_choice
        else:
            print("MovieMate-IA: Scelta non valida. Per favore, scegli tra Decision Tree, Random Forest, Naive Bayes e GBM.")

# Dividi il dataset in variabili indipendenti (X) e variabile dipendente (y)
X = df.drop(columns=['id', 'genre', 'minute', 'rating'])
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

# Scegli il modello di classificazione
chosen_model = choose_model()

if chosen_model == 'decision tree':
    # Crea il modello di classificazione con un decision tree
    model = make_pipeline(DecisionTreeClassifier())

elif chosen_model == 'random forest':
    # Crea il modello di classificazione con un random forest
    model = make_pipeline(RandomForestClassifier())

elif chosen_model == 'naive bayes':
    # Crea il modello di classificazione con Gaussian Naive Bayes
    model = make_pipeline(GaussianNB())

elif chosen_model == 'gbm':
    # Crea il modello di classificazione con Gradient Boosting Machine (GBM)
    model = make_pipeline(GradientBoostingClassifier())

# Addestra il modello
model.fit(X_train_imputed, y_train)

# Esegui il chatbot
recommend_movies(X)

# Effettua le predizioni sul set di test
y_pred = model.predict(X_test_imputed)

# Calcola le misure di prestazione
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Stampare le misure di prestazione
print(f'Performance del {chosen_model.capitalize()}:')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

# Calcola e visualizza la matrice di confusione
confusion_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice di Confusione')
plt.colorbar()
tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
plt.yticks(tick_marks, label_encoder.classes_)
plt.tight_layout()
plt.ylabel('Valore vero')
plt.xlabel('Valore predetto')
plt.show()
