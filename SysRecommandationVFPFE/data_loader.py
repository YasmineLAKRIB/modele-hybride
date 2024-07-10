import pandas as pd # type: ignore
from sqlalchemy import create_engine # type: ignore
import time
import streamlit as st # type: ignore

from utils import check_data_quality, check_loaded_data, handle_missing_data, filter_residence_vols


#####################################################################


# Initialisation du moteur de base de données
conn_str = "DRIVER={SQL Server};SERVER=DESKTOP-R16KP0C\\SQLEXPRESS;DATABASE=BD_FFP_AirAlgerie;Trusted_Connection=yes;"
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str}")

# Fonction pour charger les données depuis la base de données
 
def load_data(query):
    start_time = time.time()
    df = pd.read_sql(query, engine)
    end_time = time.time()
    print(f"Temps de chargement des données: {end_time - start_time:.4f} secondes")
    return df

#####################################################################

# Connexion à la base de données et requête SQL
query = """
SELECT 
    profile.id_profile,
    customer.gender,
    membership.membershipType,
    flight.flight_departure_code,
    flight.flight_arrival_code,
    flight.flight_category,
    flight.flight_departure_time,
    flight.cabin,
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
"""




#####################################################################



# Charger les données et les prétraiter
df = load_data(query)


# Convertir flight_departure_time en datetime
df['flight_departure_time'] = pd.to_datetime(df['flight_departure_time'], utc=True)
df['weekday'] = df['flight_departure_time'].dt.weekday
df['month'] = df['flight_departure_time'].dt.month


df = handle_missing_data(df)

# Filtrer les données pour exclure les destinations de résidence
df_filtered = df.groupby('id_profile', group_keys=False).apply(filter_residence_vols, include_groups=False).reset_index(drop=True)

#check_loaded_data(df)
#check_data_quality(df)


#####################################################################


print(f"DataLoading Done")


