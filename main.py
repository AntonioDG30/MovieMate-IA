import warnings  # Importa il modulo per gestire gli avvisi
import pandas as pd  # Importa la libreria pandas per la manipolazione dei dati
import random  # Importa il modulo random per la generazione di numeri casuali
import matplotlib.pyplot as plt  # Importa matplotlib per la visualizzazione dei dati
import numpy as np  # Importa numpy per la manipolazione efficiente di array
from sklearn.model_selection import train_test_split  # Importa train_test_split per dividere i dati in set di addestramento e test
from sklearn.tree import DecisionTreeClassifier  # Importa DecisionTreeClassifier come modello di classificazione
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Importa modelli di classificazione ensemble
from sklearn.naive_bayes import GaussianNB  # Importa il classificatore Naive Bayes
from sklearn.pipeline import make_pipeline  # Importa make_pipeline per creare una pipeline di trasformazione
from sklearn.preprocessing import LabelEncoder  # Importa LabelEncoder per codificare le etichette di classe
from sklearn.impute import SimpleImputer  # Importa SimpleImputer per la gestione dei valori mancanti
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix  # Importa metriche di valutazione del modello
from deep_translator import GoogleTranslator  # Importa GoogleTranslator per tradurre testo
from difflib import get_close_matches  # Importa get_close_matches per trovare corrispondenze simili

# Impostazione della modalità admin
modalita_Admin = False  # Imposta la modalità amministratore su False

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignora gli avvisi di deprecazione
warnings.filterwarnings("ignore", category=UserWarning)  # Ignora gli avvisi utente

# Carica il dataset dei film
df = pd.read_csv("data/nuovo_dataset.csv")  # Legge il dataset principale dei film

# Carica il dataset con i nomi e le descrizioni dei film
movies_df = pd.read_csv("data/movies.csv")  # Legge il dataset dei film con nomi e descrizioni

# Rimuovi eventuali righe con valori mancanti dal dataset principale
df.dropna(inplace=True)  # Rimuove le righe con valori mancanti dal dataset principale

# Dizionario dei 10 temi più comuni per ogni genere
top_themes_per_genre = {}  # Dizionario per i 10 temi più comuni per ogni genere
for genre in df['genre'].unique():  # Loop attraverso i generi unici nel dataset
    genre_df = df[df['genre'] == genre].drop(  # DataFrame per il genere specifico
        columns=['id', 'genre', 'minute', 'rating'])  # Rimuove colonne non necessarie
    top_themes = genre_df.sum().nlargest(10).index.tolist()  # Ottiene i 10 temi più comuni
    top_themes_per_genre[genre] = top_themes  # Aggiunge i temi al dizionario


def recommend_movies(X):
    """
    Consiglia film basati sulle preferenze dell'utente.

    Args:
    X : pandas.DataFrame
        Il DataFrame contenente i dati sui film.
    """
    print("MovieMate-IA: Benvenuto in MovieMate-IA, il tuo suggeritore personale di film!")  # Messaggio di benvenuto
    while True:  # Loop fino a quando l'utente esce
        print("MovieMate-IA: Vuoi un consiglio per un nuovo film? (si/no): ")  # Domanda all'utente
        response = input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: ").lower()  # Input dell'utente
        if response != 'si':  # Se l'utente non vuole un consiglio
            print("MovieMate-IA: Grazie per aver utilizzato il nostro servizio!")  # Messaggio di ringraziamento
            break  # Esce dal loop
        else:  # Se l'utente vuole un consiglio
            print(
                "MovieMate-IA: Per favore, inserisci il genere di film che vorresti vedere: ")  # Richiesta del genere di film
            user_input_genre = input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: ")  # Input dell'utente
            translated_genre = get_correct_genre(user_input_genre)  # Ottiene il genere corretto tradotto
            if not translated_genre:  # Se il genere non è valido
                continue  # Ritorna al punto di inizio del loop

            genre_top_themes = top_themes_per_genre[translated_genre]  # Ottiene i temi principali per il genere
            translated_genre_top_themes = translate_themes(genre_top_themes)  # Traduce i temi principali
            user_theme_responses = {}  # Dizionario per le risposte dell'utente sui temi
            for theme in translated_genre_top_themes:  # Loop attraverso i temi
                while True:  # Loop fino a quando l'utente fornisce una risposta valida
                    print(
                        f"MovieMate-IA: Ti piace il tema '{translated_genre_top_themes[theme]}'? (si/no): ")  # Domanda all'utente
                    user_response = input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: ").lower()  # Input dell'utente
                    if user_response in ['sì', 'si', 'yes', 'no']:  # Se la risposta è valida
                        user_theme_responses[theme] = 1 if user_response in ['sì', 'si',
                                                                             'yes'] else 0  # Registra la risposta dell'utente
                        break  # Esce dal loop
                    else:  # Se la risposta non è valida
                        print(
                            "MovieMate-IA: Risposta non valida. Si prega di rispondere con 'sì' o 'no'.")  # Avviso di risposta non valida

            print("MovieMate-IA: Inserisci la durata massima del film in minuti: ")  # Richiesta della durata massima
            user_input_max_duration = int(input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: "))  # Input dell'utente
            print("MovieMate-IA: Inserisci il rating minimo desiderato: ")  # Richiesta del rating minimo
            user_input_min_rating = float(input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: "))  # Input dell'utente

            user_input_features = pd.DataFrame(columns=X.columns)  # DataFrame per le caratteristiche dell'utente
            for col in user_input_features.columns:  # Loop attraverso le colonne del DataFrame
                if col in user_theme_responses:  # Se il tema è presente nelle risposte dell'utente
                    user_input_features[col] = [user_theme_responses[col]]  # Assegna la risposta dell'utente
                else:  # Se il tema non è presente nelle risposte dell'utente
                    user_input_features[col] = [0]  # Assegna 0

            prediction = model.predict(user_input_features)  # Esegue la predizione con il modello
            recommended_movies_ids = df[(df['genre'] == label_encoder.inverse_transform(prediction).ravel()[
                0]) &  # Ottiene gli ID dei film raccomandati
                                        (df[genre_top_themes].sum(axis=1) >= len(
                                            genre_top_themes) // 2) &  # Filtra per i temi principali
                                        (df['minute'] <= user_input_max_duration) &  # Filtra per la durata massima
                                        (df['rating'] >= user_input_min_rating)][
                'id'].tolist()  # Filtra per il rating minimo

            recommended_movies_info = movies_df[movies_df['id'].isin(recommended_movies_ids)][
                # Ottiene le informazioni dei film raccomandati
                ['name', 'description', 'minute']]  # Seleziona solo nome, descrizione e durata
            if not recommended_movies_info.empty:  # Se ci sono film raccomandati
                print("MovieMate-IA: Ti consigliamo i seguenti film:\n")  # Messaggio di raccomandazione
                random_state = random.randint(1, 1000)  # Stato casuale per la selezione casuale
                random_movies = recommended_movies_info.sample(n=min(3, len(recommended_movies_info)),
                                                               # Seleziona al massimo 3 film raccomandati
                                                               random_state=random_state)  # Utilizza il random state per la coerenza
                for idx, movie in random_movies.iterrows():  # Loop attraverso i film raccomandati
                    translated_description = GoogleTranslator(source='auto', target='it').translate(
                        movie['description'])  # Traduce la descrizione del film
                    formatted_info = format_movie_info(movie,
                                                       translated_description)  # Formatta le informazioni del film
                    print(formatted_info)  # Stampa le informazioni formattate
                    print("-" * 50)  # Linea di separazione
            else:  # Se non ci sono film raccomandati
                print(
                    "MovieMate-IA: Ci dispiace, non abbiamo raccomandazioni per questo genere o i tuoi criteri di selezione.")  # Messaggio di avviso
            print()  # Linea vuota


def translate_themes(themes):
    """
    Traduce i temi dall'inglese all'italiano.

    Args:
    themes : list
        Elenco dei temi da tradurre.

    Returns:
    dict: Dizionario contenente i temi tradotti.
    """
    translated_themes = {}  # Dizionario per i temi tradotti
    for theme in themes:  # Loop attraverso i temi
        translated_theme = GoogleTranslator(source='auto', target='it').translate(theme)  # Traduce il tema
        translated_themes[theme] = translated_theme  # Aggiunge il tema tradotto al dizionario
    return translated_themes  # Restituisce il dizionario dei temi tradotti


def translate_genre(genre):
    """
    Traduce il genere dall'italiano all'inglese.

    Args:
    genre : str
        Il genere da tradurre.

    Returns:
    str: Il genere tradotto.
    """
    translated_genre = GoogleTranslator(source='auto', target='en').translate(genre)  # Traduce il genere
    return translated_genre  # Restituisce il genere tradotto


def translate_genre2(genres):
    """
    Traduce una lista di generi dall'inglese all'italiano.

    Args:
    genres : list
        Elenco dei generi da tradurre.

    Returns:
    list: Lista dei generi tradotti.
    """
    translated_genres = []  # Lista per i generi tradotti
    for genre in genres:  # Loop attraverso i generi
        translated_genre = GoogleTranslator(source='auto', target='it').translate(genre)  # Traduce il genere
        translated_genres.append(translated_genre)  # Aggiunge il genere tradotto alla lista
    return translated_genres  # Restituisce la lista dei generi tradotti


def suggest_similar_genre(genre):
    """
    Suggerisce generi simili basati su una corrispondenza approssimativa.

    Args:
    genre : str
        Il genere di riferimento.

    Returns:
    list: Lista dei generi simili suggeriti.
    """
    genres = list(top_themes_per_genre.keys())  # Ottiene tutti i generi disponibili

    similar_genres = get_close_matches(genre, genres)  # Trova corrispondenze simili al genere fornito
    translated_similar_genres = translate_genre2(similar_genres)  # Traduce i generi simili
    return translated_similar_genres  # Restituisce i generi simili tradotti


def get_correct_genre(genre):
    """
    Ottiene il genere corretto dall'utente.

    Args:
    genre : str
        Il genere inserito dall'utente.

    Returns:
    str: Il genere corretto tradotto.
    """
    while True:  # Loop fino a quando non viene fornito un genere valido
        translated_genre = translate_genre(genre)  # Traduce il genere inserito
        if translated_genre in top_themes_per_genre:  # Se il genere è valido
            return translated_genre  # Restituisce il genere corretto
        else:  # Se il genere non è valido
            similar_genres = suggest_similar_genre(genre)  # Suggerisce generi simili
            if similar_genres:  # Se ci sono generi simili suggeriti
                print(
                    f"MovieMate-IA: Il genere '{genre}' non è valido. Forse intendevi: {', '.join(similar_genres)}")  # Avviso di genere non valido con suggerimenti
            else:  # Se non ci sono generi simili suggeriti
                print(
                    "MovieMate-IA: Il genere inserito non è valido e non sono stati trovati generi simili.")  # Avviso di genere non valido
                print(
                    "MovieMate-IA: Per favore, inserisci un genere valido: ")  # Richiesta di inserimento di un genere valido
            genre = input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: ")  # Input dell'utente


def format_movie_info(movie, description):
    """
    Formatta le informazioni sul film.

    Args:
    movie : pandas.Series
        Serie contenente le informazioni del film.
    description : str
        Descrizione del film.

    Returns:
    str: Informazioni formattate sul film.
    """
    words = description.split()  # Splitta la descrizione in parole
    segmented_description = [words[i:i + 12] for i in
                             range(0, len(words), 12)]  # Segmenta la descrizione in gruppi di 12 parole
    formatted_description = '\n'.join(
        [' '.join(segment) for segment in segmented_description])  # Unisce i segmenti di descrizione
    return f"Nome: {movie['name']}\nDescrizione:\n{formatted_description}\nDurata: {movie['minute']} minuti\n"  # Restituisce le informazioni formattate


def choose_model():
    """
    Permette all'utente di scegliere il modello di classificazione.

    Returns:
    str: Il modello di classificazione scelto dall'utente.
    """
    while True:  # Loop fino a quando l'utente sceglie un modello valido
        print(
            "MovieMate-IA: Quale modello di classificazione desideri utilizzare? (Decision Tree/Random Forest/Naive Bayes/GBM): ")  # Domanda all'utente
        model_choice = input("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tYou: ").lower()  # Input dell'utente
        if model_choice in ['decision tree', 'random forest', 'naive bayes', 'gbm']:  # Se il modello scelto è valido
            return model_choice  # Restituisce il modello scelto
        else:  # Se il modello scelto non è valido
            print(
                "MovieMate-IA: Scelta non valida. Per favore, scegli tra Decision Tree, Random Forest, Naive Bayes e GBM.")  # Avviso di scelta non valida


X = df.drop(columns=['id', 'genre', 'minute', 'rating'])  # Caratteristiche del film
y = df['genre']  # Target (genere del film)

label_encoder = LabelEncoder()  # Inizializza l'encoder di etichette
y = label_encoder.fit_transform(y)  # Codifica le etichette del genere

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)  # Divisione dei dati in set di addestramento e test

imputer = SimpleImputer(strategy='mean')  # Inizializza l'imputatore per i valori mancanti
X_train_imputed = imputer.fit_transform(X_train)  # Trasforma i dati di addestramento
X_test_imputed = imputer.transform(X_test)  # Trasforma i dati di test

if modalita_Admin:  # Se la modalità amministratore è attiva
    chosen_model = choose_model()  # L'utente sceglie il modello

    if chosen_model == 'decision tree':  # Se il modello scelto è Decision Tree
        model = make_pipeline(DecisionTreeClassifier())  # Crea una pipeline con Decision Tree

    elif chosen_model == 'random forest':  # Se il modello scelto è Random Forest
        model = make_pipeline(RandomForestClassifier())  # Crea una pipeline con Random Forest

    elif chosen_model == 'naive bayes':  # Se il modello scelto è Naive Bayes
        model = make_pipeline(GaussianNB())  # Crea una pipeline con Naive Bayes

    elif chosen_model == 'gbm':  # Se il modello scelto è GBM
        model = make_pipeline(GradientBoostingClassifier())  # Crea una pipeline con GBM
else:  # Se la modalità amministratore non è attiva
    model = make_pipeline(GaussianNB())  # Utilizza Naive Bayes come modello predefinito

model.fit(X_train_imputed, y_train)  # Addestra il modello con i dati di addestramento

recommend_movies(X)  # Consiglia film all'utente

y_pred = model.predict(X_test_imputed)  # Predice il genere dei film nel set di test

if modalita_Admin:  # Se la modalità amministratore è attiva
    accuracy = accuracy_score(y_test, y_pred)  # Calcola l'accuratezza del modello
    precision = precision_score(y_test, y_pred, average='weighted')  # Calcola la precisione del modello
    recall = recall_score(y_test, y_pred, average='weighted')  # Calcola il richiamo del modello
    f1 = f1_score(y_test, y_pred, average='weighted')  # Calcola il punteggio F1 del modello

    print(f'Performance del {chosen_model.capitalize()}:')  # Stampa la performance del modello scelto
    print(f'Accuracy: {accuracy}')  # Stampa l'accuratezza del modello
    print(f'Precision: {precision}')  # Stampa la precisione del modello
    print(f'Recall: {recall}')  # Stampa il richiamo del modello
    print(f'F1-score: {f1}')  # Stampa il punteggio F1 del modello

    confusion_matrix = confusion_matrix(y_test, y_pred)  # Calcola la matrice di confusione
    plt.figure(figsize=(8, 6))  # Imposta le dimensioni della figura
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # Visualizza la matrice di confusione
    plt.title('Matrice di Confusione')  # Titolo della figura
    plt.colorbar()  # Aggiunge una barra dei colori
    tick_marks = np.arange(len(label_encoder.classes_))  # Crea i segni degli assi
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)  # Imposta le etichette sull'asse x
    plt.yticks(tick_marks, label_encoder.classes_)  # Imposta le etichette sull'asse y
    plt.tight_layout()  # Ottimizza la disposizione
    plt.ylabel('Valore vero')  # Etichetta sull'asse y
    plt.xlabel('Valore predetto')  # Etichetta sull'asse x
    plt.show()  # Mostra la figura

    # Calcola e visualizza le metriche per i singoli generi
    genre_metrics = {}  # Dizionario per le metriche per i singoli generi
    for genre in label_encoder.classes_:  # Loop attraverso i generi
        genre_indices = np.where(
            label_encoder.inverse_transform(y_test) == genre)  # Ottiene gli indici per il genere specifico
        genre_y_test = y_test[genre_indices]  # Etichette di classe per il genere specifico
        genre_y_pred = y_pred[genre_indices]  # Predizioni per il genere specifico
        genre_accuracy = accuracy_score(genre_y_test, genre_y_pred)  # Calcola l'accuratezza per il genere
        genre_precision = precision_score(genre_y_test, genre_y_pred,
                                          average='weighted')  # Calcola la precisione per il genere
        genre_recall = recall_score(genre_y_test, genre_y_pred, average='weighted')  # Calcola il richiamo per il genere
        genre_f1 = f1_score(genre_y_test, genre_y_pred, average='weighted')  # Calcola il punteggio F1 per il genere
        genre_metrics[genre] = {'Accuracy': genre_accuracy, 'Precision': genre_precision, 'Recall': genre_recall,
                                'F1-score': genre_f1}  # Aggiunge le metriche al dizionario

    # Stampa le metriche per i singoli generi
    print("Metriche per i singoli generi:")  # Intestazione
    for genre, metrics in genre_metrics.items():  # Loop attraverso i generi e le relative metriche
        print(f"Genere: {genre}")  # Stampa il genere
        print(f"\tAccuracy: {metrics['Accuracy']}")  # Stampa l'accuratezza del genere
        print(f"\tPrecision: {metrics['Precision']}")  # Stampa la precisione del genere
        print(f"\tRecall: {metrics['Recall']}")  # Stampa il richiamo del genere
        print(f"\tF1-score: {metrics['F1-score']}")  # Stampa il punteggio F1 del genere
        print()  # Linea vuota

