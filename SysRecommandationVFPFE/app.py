
from SysRecommandationVFPFE.testt import display_available_flights, display_traveler_info, get_hybrid_recommendations
import streamlit as st # type: ignore

def main():
    st.title("Système de Recommandation de Vol")
    st.sidebar.image("images/air-algerie-logo.svg", width=200)
    st.sidebar.title("Menu")
    option = st.sidebar.selectbox("Choisissez une option", ["Afficher les informations du voyageur", "Afficher les vols disponibles", "Tester le système de recommandation"])
    #st.write(f"Option sélectionnée : {option}")

    if option == "Afficher les informations du voyageur":
        profile_id = st.text_input("Entrez l'ID de profil du voyageur")
        if profile_id:
            st.subheader("Informations sur le voyageur")
            traveler_info = display_traveler_info(profile_id)
            st.write(traveler_info)

    elif option == "Afficher les vols disponibles":
        
        available_flights = display_available_flights()
        st.write(available_flights)

    elif option == "Tester le système de recommandation":
        profile_id = st.text_input("Entrez l'ID de profil pour les recommandations")
        if profile_id:
            top_recommendations, accuracy = get_hybrid_recommendations(profile_id)
            st.write(f"Recommandations hybrides pour l'utilisateur {profile_id}:")
            for rec in top_recommendations:
                if rec['departure_code']:
                    st.write(f"Départ: {rec['departure_code']},Destination: {rec['destination']},  Date de départ: {rec['departure_time']}")
                else:
                    st.write(f"Destination: {rec['destination']}")
            #st.write(f"Accuracy des recommandations hybrides: {accuracy:.4f}")

if __name__ == "__main__":
    main()
