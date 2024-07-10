from datetime import datetime
import os
from turtle import st
import pandas as pd # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import numpy as np # type: ignore
from sklearn.impute import KNNImputer # type: ignore
from sklearn.preprocessing import OrdinalEncoder # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer # type: ignore
import streamlit as st # type: ignore


import json


#########################################################################


# Fonction pour gérer les données manquantes
 
def handle_missing_data(df):
    # Sauvegarde des types de données initiaux
    original_dtypes = df[['gender', 'flight_category', 'cabin', 'membershipType','weekday','month']].dtypes
    
    # Remplacement des valeurs nulles par 'Unknown'
    df[['gender', 'flight_category', 'cabin', 'membershipType','weekday','month']] = df[['gender', 'flight_category', 'cabin', 'membershipType','weekday','month']].fillna('Unknown')
    
    # Suppression sélective des instances avec 3 ou plus d'attributs 'Unknown'
    unknown_count = df[['gender', 'flight_category', 'cabin', 'membershipType','weekday','month']].apply(lambda x: x == 'Unknown', axis=1).sum(axis=1)
    df = df[unknown_count < 3]  # Garder les lignes avec moins de 3 'Unknown'

    # Encoder les variables catégorielles en nombres
    encoder = OrdinalEncoder()
    df_encoded = pd.DataFrame(encoder.fit_transform(df[['gender', 'flight_category', 'cabin', 'membershipType','weekday','month']]),
                            columns=['gender', 'flight_category', 'cabin', 'membershipType','weekday','month'],
                            index=df.index)
    
    # Imputation avec KNN pour les valeurs manquantes
    imputer = KNNImputer(n_neighbors=5)
    df_imputed_encoded = pd.DataFrame(imputer.fit_transform(df_encoded),
                                    columns=['gender', 'flight_category', 'cabin', 'membershipType','weekday','month'],
                                    index=df.index)
    
    # Restauration des valeurs initiales
    df_imputed = pd.DataFrame(encoder.inverse_transform(df_imputed_encoded),
                            columns=['gender', 'flight_category', 'cabin', 'membershipType','weekday','month'],
                            index=df_imputed_encoded.index)
    
    # Remplacer les valeurs imputées dans le DataFrame original
    df[['gender', 'flight_category', 'cabin', 'membershipType','weekday','month']] = df_imputed
    
    # Restaurer les types de données initiaux
    df[['gender', 'flight_category', 'cabin', 'membershipType','weekday','month']] = df[['gender', 'flight_category', 'cabin', 'membershipType','weekday','month']].astype(original_dtypes)
    
    return df






#########################################################################





# Fonction pour vérifier les données chargées
@st.cache
def check_loaded_data(df):
    print(f"Nombre d'instances dans le DataFrame : {len(df)}")
    print("\nStructure du DataFrame :")
    print(df.info())
    # Vérification du nombre de profils distincts
    distinct_profiles = df['id_profile'].nunique()
    print(f"\nNombre de profils distincts : {distinct_profiles}")

    # Vérification du nombre de flight_arrival_code distincts
    distinct_arrival_codes = df['flight_arrival_code'].nunique()
    print(f"\nNombre de flight_arrival_code distincts : {distinct_arrival_codes}")





#########################################################################




# Fonction pour vérifier les valeurs uniques et les valeurs manquantes
@st.cache
def check_data_quality(df):
    print("\nVérification des valeurs uniques :")
    print(df['flight_arrival_code'].value_counts())
    print("\nVérification des valeurs manquantes :")
    print(df.isnull().sum())




#########################################################################



# Fonction pour calculer la similarité
 
def calculate_similarity(profile, flight_features):
    # Calculer la similarité cosinus
    profile_values = profile.values.reshape(1, -1)
    flight_values = flight_features.values.reshape(1, -1)
    similarity = cosine_similarity(profile_values, flight_values)[0][0]
    return similarity



#########################################################################




# Fonction pour filtrage basé sur le contenu
def content_based_filtering(profile_id, flights, num_recommendations=10):
    # Ajouter les nouvelles caractéristiques pour le filtrage basé sur le contenu
    content_features = ['flight_departure_code', 'flight_arrival_code', 'flight_category', 'cabin', 'gender', 'membershipType', 'month', 'weekday']

    # Filtrer le profil utilisateur
    user_profile = flights[flights['id_profile'] == profile_id]
    # Vérifier si le profil utilisateur est vide
    if user_profile.empty:
        print(f"Profil utilisateur avec l'id {profile_id} non trouvé.")
        return []
    #print(user_profile.columns)

    # Récupérer la ville de résidence du voyageur
    user_residence_city = user_profile['addressCityName'].iloc[0]

    # Filtrer les vols où flight_arrival_code est égal à la ville de résidence du voyageur si elle est définie
    if pd.notnull(user_residence_city):
        flights = flights[flights['flight_arrival_code'] != user_residence_city]


    # Encodage des caractéristiques catégorielles
    flights_encoded = pd.get_dummies(flights[content_features])
    user_profile_encoded = pd.get_dummies(user_profile[content_features])

    # Aligner les colonnes des dataframes encodés
    flights_encoded, user_profile_encoded = flights_encoded.align(user_profile_encoded, join='outer', axis=1, fill_value=0)


    
    # Calculer la moyenne des caractéristiques encodées pour le profil utilisateur
    user_profile_vector = user_profile_encoded.mean(axis=0)


    # Calculer la similarité avec tous les vols disponibles
    similarities = []
    for idx, row in flights_encoded.iterrows():
        similarity = calculate_similarity(user_profile_vector, row)
        similarities.append((flights.loc[idx, 'flight_arrival_code'], similarity))

    # Trier par similarité décroissante
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)


    # Supprimer les redondances et garder le score le plus élevé pour chaque destination
    unique_destinations = {}
    for destination, rating in similarities[:num_recommendations]:
        if destination not in unique_destinations:
            unique_destinations[destination] = rating
        else:
            if rating > unique_destinations[destination]:
                unique_destinations[destination] = rating


    # Retourner les recommandations finales
    recommendations = {dest: rating for dest, rating in unique_destinations.items()}

    return recommendations



#########################################################################






def evaluate_content_based_recommendations(profile_id, flights_data,recommendations_content_based):

    # Récupérer les vols réels réservés par l'utilisateur
    #actual_flights_departure = flights_data.loc[flights_data['id_profile'] == profile_id, 'flight_departure_code'].tolist()
    actual_flights_arrival = flights_data.loc[flights_data['id_profile'] == profile_id, 'flight_arrival_code'].tolist()
    actual_flights_departure = flights_data.loc[flights_data['id_profile'] == profile_id, 'flight_departure_code'].tolist()
    # Combiner les codes de départ et d'arrivée des vols sans redondance
    actual_flights = list(set(actual_flights_arrival+actual_flights_departure))


    # Extraire les destinations recommandées
    recommended_flights = list(recommendations_content_based.keys())


    # Trouver toutes les destinations possibles
    all_flights = list(set(actual_flights + recommended_flights))
    
    # Utiliser MultiLabelBinarizer pour créer des vecteurs binaires
    mlb = MultiLabelBinarizer(classes=all_flights)
    actual_flights_binary = mlb.fit_transform([actual_flights])
    recommended_flights_binary = mlb.transform([recommended_flights])


    # Aplatir les vecteurs binaires
    actual_flights_binary = actual_flights_binary.flatten()
    recommended_flights_binary = recommended_flights_binary.flatten()

    # Comparer uniquement les positions communes entre les vecteurs binaires des vols réels et recommandés
    common_indices = np.where((actual_flights_binary + recommended_flights_binary) > 0)[0]
    actual_flights_common = actual_flights_binary[common_indices]
    recommended_flights_common = recommended_flights_binary[common_indices]


    # Calculer la précision, le rappel et le F1-score
    precision_content = precision_score(actual_flights_common, recommended_flights_common)
    recall_content = recall_score(actual_flights_common, recommended_flights_common)
    f1_content = f1_score(actual_flights_common, recommended_flights_common)

    return precision_content, recall_content, f1_content







#########################################################################




# Fonction pour normaliser les scores de recommandations par la norme euclidienne
 
def normalize_scores(recommendations):
    if not recommendations:
        return []

    if isinstance(recommendations, list):
        # Convertir la liste en dictionnaire
        recommendations = {dest: score for dest, score in recommendations}

    # Utilisation de items() pour itérer sur les clés et les valeurs
    scores = np.array([score for _, score in recommendations.items()])

    # Calculer la norme euclidienne
    norm = np.linalg.norm(scores)

    if norm == 0:
        # Attribuer une valeur de 0 si la norme est 0 (cas où tous les scores sont 0)
        normalized_scores = [(destination, 0) for destination, score in recommendations.items()]
    else:
        # Normaliser les scores par la norme euclidienne
        normalized_scores = [(destination, score / norm) for destination, score in recommendations.items()]

    return normalized_scores



#########################################################################




 
def hybrid_recommendations(profile_id,flights_data, recommendations_user_based, recommendations_content_based, num_recommendations=20):
    weight_user_based=0.5
    weight_content_based=0.5

    # Récupérer les données historiques du voyageur
    actual_flights_departure = flights_data.loc[flights_data['id_profile'] == profile_id, 'flight_departure_code'].tolist()
    actual_flights_arrival = flights_data.loc[flights_data['id_profile'] == profile_id, 'flight_arrival_code'].tolist()
    actual_flights = list(set(actual_flights_departure + actual_flights_arrival))

    print("Evaluation hybrid recommendations pour le profile:", profile_id)
    
    # Print recommendations_user_based
    print("\nUser-Based Recommendations (Before Normalization):")
    for rec in recommendations_user_based:
        print(f"Destination: {rec[0]}, Score: {rec[1]}")

    # Print recommendations_content_based
    print("\nContent-Based Recommendations (Before Normalization):")
    for destination, rating in recommendations_content_based.items():  # Utilisation de items() pour itérer sur les clés et les valeurs
        print(f"Destination: {destination}, Score: {rating:.2f}")


    normalized_scores_user_based = normalize_scores(recommendations_user_based)


    print("\nNormalized User-Based Recommendations:")
    for rec in normalized_scores_user_based:
        print(f"Destination: {rec[0]}, Normalized Score: {rec[1]}")


    # Fusion pondérée des recommandations
    hybrid_recommendations = {}
    print("\nFusion pondérée des recommandations:")
    for destination, rating in recommendations_content_based.items():
        score_content_based = float(rating)
        hybrid_recommendations[destination] = score_content_based * weight_content_based
        print(f"Recommendation Content-based pour {destination}: Score = {score_content_based:.2f}, Score pondéré = {hybrid_recommendations[destination]:.2f}")


    for rec in normalized_scores_user_based:
        flight_code = rec[0]
        score_user_based = float(rec[1])
        if flight_code in hybrid_recommendations:
            hybrid_recommendations[flight_code] += score_user_based * weight_user_based
            print(f"Recommendation User-based pour {flight_code}: Score = {score_user_based:.2f}, Score pondéré = {hybrid_recommendations[flight_code]:.2f} (updated)")
        else:
            hybrid_recommendations[flight_code] = score_user_based 
            print(f"Recommendation User-based pour {flight_code}: Score = {score_user_based:.2f}, Score pondéré = {hybrid_recommendations[flight_code]:.2f} (added)")



    # Trier les recommandations par score pondéré décroissant
    sorted_recommendations = sorted(hybrid_recommendations.items(), key=lambda x: x[1], reverse=True)

    # Sélectionner les meilleures recommandations
    top_recommendations = sorted_recommendations[:num_recommendations]
    print("\nTop Hybrid Recommendations:")
    for i, rec in enumerate(top_recommendations):
        print(f"Rank {i + 1}: {rec[0]} - Score: {rec[1]}")


    return top_recommendations




#########################################################################



# Fonction pour obtenir les mois d'intervalle
 
def get_month_interval(month, range_months=1):
    return [(month + i - 1) % 12 + 1 for i in range(-range_months, range_months + 1)]




 #####################################################################



# Filtrage des vols où flight_arrival_code est égal à la ville de résidence de l'utilisateur
# Groupement par profil et application du filtrage
 
def filter_residence_vols(group):
    # Exclure les vols où flight_arrival_code est égal à la ville de résidence du profil
    residence_city = group['addressCityName'].iloc[0]
    if group[group['flight_arrival_code'] == residence_city].empty:
        return group
    else:
        return group[group['flight_arrival_code'] != residence_city]



#########################################################################



 
# Correspondance des recommandations avec les vols disponibles
def match_recommendations_with_flights(profile_id, top_recommendations, df, engine):
    # Charger les vols disponibles
    query_available_flights = "SELECT * FROM available_flight"
    available_flights = pd.read_sql(query_available_flights, engine)

    highlighted_recommendations = []
    for recommendation in top_recommendations:
        flight_arrival_code = recommendation['destination']
        score = recommendation['score']
        found = False
        
        # Vérifier si la recommandation hybride correspond à un vol dans available_flight
        matching_flights = available_flights[available_flights['flight_arrival_code'] == flight_arrival_code]
        
        # Si des vols disponibles existent pour cette destination
        if not matching_flights.empty:
            user_history = df[(df['id_profile'] == profile_id) & (df['flight_arrival_code'] == flight_arrival_code)]
            user_departures = df[df['id_profile'] == profile_id]['flight_departure_code'].unique()
            
            # Si l'utilisateur a déjà voyagé vers cette destination
            if not user_history.empty:
                user_months = user_history['month'].unique()
                user_weekdays = user_history['weekday'].unique()

                for idx, flight in matching_flights.iterrows():
                    flight_departure_code = flight['flight_departure_code']
                    flight_departure_time = pd.to_datetime(flight['flight_departure_time'], utc=True)
                    flight_month = flight_departure_time.month
                    flight_weekday = flight_departure_time.weekday()
                    
                    # Vérifier si le vol de départ correspond à l'historique du voyageur
                    if flight_departure_code in user_departures:
                        for month in user_months:
                            if flight_month in get_month_interval(month, range_months=1):  # Intervalle d'un mois avant et après
                                found = True
                                highlighted_recommendations.append({
                                    'destination': flight_arrival_code,
                                    'score': score,
                                    'departure_code': flight_departure_code,
                                    'departure_time': flight_departure_time
                                })
                                break  # Sortir de la boucle si une correspondance est trouvée
                
                if not found:
                    # Aucune correspondance trouvée pour cette destination malgré l'historique
                    default_flight = matching_flights.iloc[0]
                    highlighted_recommendations.append({
                        'destination': flight_arrival_code,
                        'score': score,
                        'departure_code': default_flight['flight_departure_code'],
                        'departure_time': pd.to_datetime(default_flight['flight_departure_time'], utc=True)
                    })
                    print(f"selon la dispo ")
                   
            else:
                # Pour les nouvelles destinations
                current_month = datetime.now().month
                for idx, flight in matching_flights.iterrows():
                    flight_departure_time = pd.to_datetime(flight['flight_departure_time'], utc=True)
                    flight_month = flight_departure_time.month
                    
                    if flight_month in get_month_interval(current_month, range_months=6):
                        highlighted_recommendations.append({
                            'destination': flight_arrival_code,
                            'score': score,
                            'departure_code': flight['flight_departure_code'],
                            'departure_time': flight_departure_time
                        })
                        found = True
                        break  # Sortir de la boucle si une correspondance est trouvée
                
                if not found:
                    # Si aucune correspondance n'est trouvée dans les habitudes générales, prendre un vol par défaut
                    default_flight = matching_flights.iloc[0]
                    highlighted_recommendations.append({
                        'destination': flight_arrival_code,
                        'score': score,
                        'departure_code': default_flight['flight_departure_code'],
                        'departure_time': pd.to_datetime(default_flight['flight_departure_time'], utc=True)
                    })
                    print(f"selon la dispo")

        # Si la destination n'a pas de vol disponible
        else:
            highlighted_recommendations.append({
                'destination': flight_arrival_code,
                'score': score,
                'departure_code': None,
                'departure_time': None
            })

    # Affichage des recommandations hybrides
    print(f"\nRecommandations hybrides après correspondance avec les vols disponibles, pour l'utilisateur {profile_id}:")
    for rec in highlighted_recommendations:
        flight_departure_code = rec['departure_code']
        flight_arrival_code = rec['destination']
        score = rec['score']
        flight_departure_time = rec['departure_time']
        
        if flight_departure_code:
            print(f"{flight_departure_code} -> {flight_arrival_code} (Score: {score}, Date de départ: {flight_departure_time})")
        else:
            print(f"{flight_arrival_code} (Score: {score})")

    return highlighted_recommendations



#########################################################################


 

def generate_and_evaluate_recommendations(df):
    all_recommendations = []
    all_evaluations = []
    global_precision = []
    global_recall = []
    global_f1 = []

    recommendations_dict = {}  # Dictionnaire pour stocker les recommandations pour chaque profil
    evaluations_dict = {}      # Dictionnaire pour stocker les évaluations pour chaque profil

    # Obtenir tous les profils uniques
    unique_profiles = df['id_profile'].unique()

    # Itérer sur chaque profil unique
    for profile_id in unique_profiles:
        # Obtention des recommandations content-based
        flights_data = df[['id_profile', 'flight_departure_code', 'flight_arrival_code', 'flight_category', 'cabin', 'gender', 'membershipType', 'month', 'weekday', 'addressCityName']].drop_duplicates()
        recommendations_content_based = content_based_filtering(profile_id, flights_data)

        # Ajouter les recommandations pour ce profil au dictionnaire
        recommendations_dict[profile_id] = recommendations_content_based

        # Enregistrer les recommandations
        for destination, rating in recommendations_content_based.items():
            all_recommendations.append({
                'id_profile': profile_id,
                'destination': destination,
                'rating': rating
            })

        # Évaluer les recommandations
        precision_content, recall_content, f1_content = evaluate_content_based_recommendations(profile_id, flights_data, recommendations_content_based)

        # Enregistrer les résultats d'évaluation pour chaque voyageur
        all_evaluations.append({
            'id_profile': profile_id,
            'precision': precision_content,
            'recall': recall_content,
            'f1_score': f1_content
        })
        
        # Ajouter les évaluations au dictionnaire
        evaluations_dict[profile_id] = {
            'precision': precision_content,
            'recall': recall_content,
            'f1_score': f1_content
        }


        # Ajouter les métriques individuelles aux listes globales
        global_precision.append(precision_content)
        global_recall.append(recall_content)
        global_f1.append(f1_content)


    # Convertir les recommandations et les évaluations en DataFrames
    recommendations_df = pd.DataFrame(all_recommendations)
    evaluations_df = pd.DataFrame(all_evaluations)

    # Sauvegarder les recommandations et les évaluations dans des fichiers CSV
    recommendations_df.to_csv('content_based_recommendations300.csv', index=False)
    evaluations_df.to_csv('content_based_evaluations300.csv', index=False)

    print("Recommandations et évaluations enregistrées avec succès.")

    # Sauvegarder les dictionnaires dans des fichiers JSON
    with open('recommendations300.json', 'w') as rec_file:
        json.dump(recommendations_dict, rec_file)
    with open('evaluations300.json', 'w') as eval_file:
        json.dump(evaluations_dict, eval_file)

    return recommendations_dict, evaluations_dict  # Retourner les recommandations et les évaluations


#########################################################################




print(f"Functions Importing Done")

