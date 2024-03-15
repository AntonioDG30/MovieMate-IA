import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Carica il dataset dei film
df = pd.read_csv("data/nuovo_dataset.csv")

# Rimuovi eventuali righe con valori mancanti
df.dropna(inplace=True)

# Verifica se ci sono valori mancanti nel dataset
#print("Valori mancanti nel dataset:")
#print(df.isnull().sum())

# Dizionario dei 5 temi più comuni per ogni genere
top_themes_per_genre = {}
for genre in df['genre'].unique():
    genre_df = df[df['genre'] == genre].drop(columns=['id', 'genre'])
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
            if user_input_genre not in top_themes_per_genre:
                print("Il genere inserito non è valido.")
                continue

            genre_top_themes = top_themes_per_genre[user_input_genre]
            user_theme_responses = {}
            for theme in genre_top_themes:
                while True:
                    user_response = input(f"Ti piace il tema '{theme}'? (si/no): ").lower()
                    if user_response in ['si', 'no']:
                        user_theme_responses[theme] = 1 if user_response == 'si' else 0
                        break
                    else:
                        print("Risposta non valida. Si prega di rispondere con 'si' o 'no'.")

            user_input_features = pd.DataFrame(columns=X.columns)
            for col in user_input_features.columns:
                if col == user_input_genre:
                    user_input_features[col] = [1]
                elif col in user_theme_responses:
                    user_input_features[col] = [user_theme_responses[col]]
                else:
                    user_input_features[col] = [0]

            prediction = model.predict(user_input_features)
            recommended_movies = df[(df['genre'] == label_encoder.inverse_transform(prediction).ravel()[0]) & (df[genre_top_themes].sum(axis=1) >= len(genre_top_themes)//2)]['id'].tolist()
            if recommended_movies:
                print("Ti consigliamo i seguenti film:")
                print(recommended_movies)
            else:
                print("Ci dispiace, non abbiamo raccomandazioni per questo genere o i tuoi criteri di selezione.")
            print()

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

