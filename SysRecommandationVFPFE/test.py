from utils import generate_and_evaluate_recommendations
from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score , accuracy_score # type: ignore
import numpy as np # type: ignore
from surprise.model_selection import GridSearchCV
import matplotlib.pyplot as plt # type: ignore
import joblib # type: ignore

from data_loader import df
import json
import pandas as pd   # type: ignore
import time
import matplotlib.pyplot as plt # type: ignore
import sys
import os
import datetime
import joblib # type: ignore
import streamlit as st # type: ignore

from data_loader import df, engine
from utils import content_based_filtering, evaluate_content_based_recommendations, generate_and_evaluate_recommendations, hybrid_recommendations, get_month_interval, match_recommendations_with_flights




###################################################################################################


def get_actual_choices(profile_id, flights_data):
    # Implémentation fictive pour récupérer les choix réels des utilisateurs
    actual_choices = flights_data[flights_data['id_profile'] == profile_id]['flight_arrival_code'].tolist()
    return actual_choices


###################################################################################################


def load_aggregated_df():
    return pd.read_csv('aggregated_df1000.csv')

#Generation et evaluation des recommandations hybrides
def generate_and_evaluate_hybrid_recommendations(profile_id, model_path, df, recommendations_dict):
    aggregated_df = load_aggregated_df()

    # Chargement du modèle user-based
    loaded_model = joblib.load(model_path)
    print(f"Modèle chargé à partir de {model_path}")

    # Faire des recommandations user-based pour un utilisateur spécifique
    all_items = df['flight_arrival_code'].unique()
    predictions = []
    for item in all_items:
        if not aggregated_df[(aggregated_df['id_profile'] == profile_id) & (aggregated_df['flight_arrival_code'] == item)].empty:
            continue
        prediction = loaded_model.predict(profile_id, item)
        predictions.append(prediction)
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = 10
    predictions_user_based = [(pred.iid, pred.est) for pred in predictions[:top_n]]

    # Faire des recommandations hybrides
    recommendations_content_based = recommendations_dict.get(profile_id, {})
    top_recommendations = hybrid_recommendations(profile_id, df, predictions_user_based, recommendations_content_based)

    return top_recommendations



###################################################################################################


#Recuperation des Recommandation Content Based 
def load_recommendations():
    if os.path.exists('recommendations1000.json') and os.path.exists(
            'evaluations1000.json'):
        with open('recommendations1000.json', 'r') as rec_file:
            recommendations_dict = json.load(rec_file)
        with open('evaluations1000.json', 'r') as eval_file:
            evaluations_dict = json.load(eval_file)
    else:
        recommendations_dict, evaluations_dict = generate_and_evaluate_recommendations(df)
    
    return recommendations_dict, evaluations_dict



###################################################################################################


#Recuperation des recommandation hybrides
def get_hybrid_recommendations(profile_id):
    recommendations_dict, evaluations_dict = load_recommendations()
    
    # Recommendation user based pour un id profile
    model_path = "model_user_based_k100_min_k7_1000.joblib"
    result = generate_and_evaluate_hybrid_recommendations(profile_id, model_path, df, recommendations_dict)

    if isinstance(result, tuple) and len(result) >= 2:
        top_recommendations = result[0]
    else:
        top_recommendations = result

    # Ensure top_recommendations is a list
    if not isinstance(top_recommendations, list):
        top_recommendations = [top_recommendations]

    # Convertir les tuples en  dictionaries
    top_recommendations = [
        {'destination': item[0], 'score': item[1], 'departure_code': None, 'departure_time': None}
        if isinstance(item, tuple) and len(item) == 2 else item
        for item in top_recommendations
    ]

    # Get actual user choices
    actual_choices = get_actual_choices(profile_id, df)

    # Compare hybrid recommendations with actual choices
    correct_recommendations = [dest for dest in top_recommendations if dest.get('destination') in actual_choices]

    # Calculate accuracy
    accuracy = len(correct_recommendations) / len(top_recommendations) if len(top_recommendations) > 0 else 0

    highlighted_recommendations = match_recommendations_with_flights(profile_id, top_recommendations, df, engine)

    return highlighted_recommendations, accuracy


###################################################################################################


 #Recuperation des Vol disponibles
def get_available_flights():
    query = """
    SELECT
        [flight_number],
        [flight_departure_code],
        [flight_arrival_code],
        [flight_departure_time]
    FROM 
        [BD_FFP_AirAlgerie].[dbo].[available_flight]
    """
    available_flights = pd.read_sql(query, engine)
    return available_flights


###################################################################################################

 
# Affichage des informations du voyageur 

def display_traveler_info(profile_id):
    query = """
    SELECT
        customer.FirstName,
        customer.LastName,
        customer.gender,
        flight.flight_departure_code,
        flight.flight_arrival_code,
        flight.flight_category,
        flight.flight_departure_time,
        membership.membershipType,
        
        contact.addressCityName
    FROM 
        profile
    JOIN 
        customer ON profile.id_customer = customer.id_customer
    JOIN 
        membership ON profile.id_profile = membership.id_profile
    JOIN
        ticket ON profile.id_profile = ticket.id_profile 
    JOIN 
        flight ON ticket.id_ticket = flight.id_ticket
    JOIN
        contact ON profile.id_profile = contact.id_profile
    WHERE
        profile.id_profile = ?
    """
    traveler_info = pd.read_sql(query, engine, params=(profile_id,))
    if not traveler_info.empty:
        st.write(traveler_info)
        #miles = traveler_info.milesBalanceValue
    else:
        st.write("Aucune information trouvée pour cet ID de profil.")



###################################################################################################


 
#functionalite de recherche de vols disponible
def display_available_flights():
    st.subheader("Recherche de vols disponibles")
    
    available_flights = get_available_flights()

    departure_code = st.text_input("Code de départ")
    arrival_code = st.text_input("Code de destination")
    
    # Filter the dataframe based on user input
    if departure_code:
        available_flights = available_flights[available_flights['flight_departure_code'] == departure_code]
    
    if arrival_code:
        available_flights = available_flights[available_flights['flight_arrival_code'] == arrival_code]
    
    if not available_flights.empty:
        st.write(available_flights)
    else:
        st.write("Aucun vol disponible trouvé.")


###################################################################################################