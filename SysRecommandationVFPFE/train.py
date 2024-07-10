import json
import time
import matplotlib.pyplot as plt # type: ignore
import sys
import os
import datetime
import joblib # type: ignore
import streamlit as st # type: ignore

###################################################################################################
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

from utils import  generate_and_evaluate_recommendations


#########################################################################

#Entrainement du modele, 
def train_model():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"output_{timestamp}.txt"
    file_path = os.path.join("D:\\PFE\\code\\PFEMaster\\SysRecommandationVF\\affichages", filename)

    with open(file_path, "w") as file:
        sys.stdout = file
        try:
            print(f"Date et heure d'exécution : {datetime.datetime.now()}")
            print("Entraînement de User Based")

            sim_options = {
                'name': 'cosine',
                'user_based': True
            }

            df['rating'] = df.groupby(['id_profile', 'flight_arrival_code'])['flight_arrival_code'].transform('count')
            aggregated_df = df.groupby(['id_profile', 'flight_arrival_code'])['rating'].mean().reset_index()

            # Sauvegarder aggregated_df dans un fichier CSV
            aggregated_df.to_csv("aggregated_df300.csv", index=False)
            
            reader = Reader(rating_scale=(1, df['rating'].max()))
            data = Dataset.load_from_df(aggregated_df[['id_profile', 'flight_arrival_code', 'rating']], reader)
            trainset, testset = train_test_split(data, test_size=0.2)
            param_grid = {'k': [ 70, 100, 130, 160, 190, 210, 240], 'min_k': [3, 5, 7, 10, 15]}
            grid_search = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=5)
            grid_search.fit(data)

            
            results_train = {}
            results_test = {}

            results = {}
            for k in param_grid['k']:
                for min_k in param_grid['min_k']:
                    algo = KNNWithMeans(k=k, min_k=min_k, sim_options=sim_options)
                    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
                    algo.fit(trainset)
                    model_filename = f"model_user_based_k{k}_min_k{min_k}_1000.joblib"
                    joblib.dump(algo, model_filename)
                    print(f"Modèle sauvegardé dans {model_filename}")


                    # Prédictions sur les données de test
                    predictions = algo.test(testset)
                    true_values = [pred.r_ui for pred in predictions]
                    predicted_values = [pred.est for pred in predictions]
                    threshold = 3
                    true_bools = [true >= threshold for true in true_values]
                    predicted_bools = [pred >= threshold for pred in predicted_values]
                    precision = precision_score(true_bools, predicted_bools)
                    recall = recall_score(true_bools, predicted_bools)
                    f1 = f1_score(true_bools, predicted_bools)
                    #accuracy = accuracy_score(true_bools, predicted_bools)
                    rmse_mean = np.mean(cv_results['test_rmse'])
                    mae_mean = np.mean(cv_results['test_mae'])

                    results[(k, min_k)] = {
                        'rmse': rmse_mean,
                        'mae': mae_mean,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        #'accuracy': accuracy
                    }


                    results_test[(k, min_k)] = {
                        'rmse': rmse_mean,
                        'mae': mae_mean,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        #'accuracy': accuracy
                    }

                    
                    print(f"\nMetrics for k={k}, min_k={min_k}:")
                    print(f"RMSE: {rmse_mean:.4f}")
                    print(f"MAE: {mae_mean:.4f}")
                    print(f"Precision: {precision:.4f}")
                    print(f"Recall: {recall:.4f}")
                    print(f"F1-score: {f1:.4f}")
                    #print(f"Accuracy: {accuracy:.4f}")

                    file_path = 'RsltEntrainement300.txt'
                    with open(file_path, 'a') as file_append:
                        file_append.write(f"k={k}, min_k={min_k}:\n")
                        file_append.write(f"RMSE: {rmse_mean:.4f}\n")
                        file_append.write(f"MAE: {mae_mean:.4f}\n")
                        file_append.write(f"Precision: {precision:.4f}\n")
                        file_append.write(f"Recall: {recall:.4f}\n")
                        file_append.write(f"F1-score: {f1:.4f}\n")
                        #file_append.write(f"Accuracy: {accuracy:.4f}\n\n")


                    # Prédictions sur les données d'entraînement
                    train_predictions = algo.test(trainset.build_testset())
                    train_true_values = [pred.r_ui for pred in train_predictions]
                    train_predicted_values = [pred.est for pred in train_predictions]
                    train_true_bools = [true >= threshold for true in train_true_values]
                    train_predicted_bools = [pred >= threshold for pred in train_predicted_values]
                    train_precision = precision_score(train_true_bools, train_predicted_bools)
                    train_recall = recall_score(train_true_bools, train_predicted_bools)
                    train_f1 = f1_score(train_true_bools, train_predicted_bools)
                    #train_accuracy = accuracy_score(train_true_bools, train_predicted_bools)
                    train_rmse = accuracy.rmse(train_predictions, verbose=False)
                    train_mae = accuracy.mae(train_predictions, verbose=False)



                    results_train[(k, min_k)] = {
                        'rmse': train_rmse,
                        'mae': train_mae,
                        'precision': train_precision,
                        'recall': train_recall,
                        'f1': train_f1,
                        #'accuracy': train_accuracy
                    }


                    print(f"\nMetrics for k={k}, min_k={min_k} (Train):")
                    print(f"RMSE: {train_rmse:.4f}")
                    print(f"MAE: {train_mae:.4f}")
                    print(f"Precision: {train_precision:.4f}")
                    print(f"Recall: {train_recall:.4f}")
                    print(f"F1-score: {train_f1:.4f}")
                    #print(f"Accuracy: {train_accuracy:.4f}")

                    file_path = 'RsltEntrainementAvcTrainset300.txt'
                    with open(file_path, 'a') as file_append:
                        file_append.write(f"Train - k={k}, min_k={min_k}:\n")
                        file_append.write(f"RMSE: {train_rmse:.4f}\n")
                        file_append.write(f"MAE: {train_mae:.4f}\n")
                        file_append.write(f"Precision: {train_precision:.4f}\n")
                        file_append.write(f"Recall: {train_recall:.4f}\n")
                        file_append.write(f"F1-score: {train_f1:.4f}\n")
                        #file_append.write(f"Accuracy: {train_accuracy:.4f}\n\n")



            #########################################



            k_values = param_grid['k']
            min_k_values = param_grid['min_k']
            generate_and_evaluate_recommendations(df)
            recommendations_dict, evaluations_dict = generate_and_evaluate_recommendations(df)
            avg_precision = np.mean([eval_dict['precision'] for eval_dict in evaluations_dict.values()])
            avg_recall = np.mean([eval_dict['recall'] for eval_dict in evaluations_dict.values()])
            avg_f1 = np.mean([eval_dict['f1_score'] for eval_dict in evaluations_dict.values()])

            file_path = 'MétriquesGlobalesContentBased1000.txt'
            with open(file_path, 'a') as file_append:
                file_append.write(f"Métriques globales Content Based 1000: \nPrécision: {avg_precision:.4f}\nRappel: {avg_recall:.4f}\nF1-score: {avg_f1:.4f}\n")

            plt.figure(figsize=(18, 16))

            plt.subplot(3, 2, 1)
            for min_k in min_k_values:
                plt.plot(k_values, [results[(k, min_k)]['rmse'] for k in k_values], marker='o', label=f'min_k={min_k}')
            plt.title('RMSE vs k for different min_k')
            plt.xlabel('k')
            plt.ylabel('RMSE')
            plt.xticks(k_values)
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 2, 2)
            for min_k in min_k_values:
                plt.plot(k_values, [results[(k, min_k)]['mae'] for k in k_values], marker='o', label=f'min_k={min_k}')
            plt.title('MAE vs k for different min_k')
            plt.xlabel('k')
            plt.ylabel('MAE')
            plt.xticks(k_values)
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 2, 3)
            for min_k in min_k_values:
                plt.plot(k_values, [results[(k, min_k)]['precision'] for k in k_values], marker='o', label=f'min_k={min_k}')
            plt.title('Precision vs k for different min_k')
            plt.xlabel('k')
            plt.ylabel('Precision')
            plt.xticks(k_values)
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 2, 4)
            for min_k in min_k_values:
                plt.plot(k_values, [results[(k, min_k)]['recall'] for k in k_values], marker='o', label=f'min_k={min_k}')
            plt.title('Recall vs k for different min_k')
            plt.xlabel('k')
            plt.ylabel('Recall')
            plt.xticks(k_values)
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 2, 5)
            for min_k in min_k_values:
                plt.plot(k_values, [results[(k, min_k)]['f1'] for k in k_values], marker='o', label=f'min_k={min_k}')
            plt.title('F1-score vs k for different min_k')
            plt.xlabel('k')
            plt.ylabel('F1-score')
            plt.xticks(k_values)
            plt.legend()
            plt.grid(True)

            '''
            plt.subplot(3, 2, 6)
            for min_k in min_k_values:
                plt.plot(k_values, [results[(k, min_k)]['accuracy'] for k in k_values], marker='o', label=f'min_k={min_k}')
            plt.title('Accuracy vs k for different min_k')
            plt.xlabel('k')
            plt.ylabel('Accuracy')
            plt.xticks(k_values)
            plt.legend()
            plt.grid(True)
            '''
            
            plt.tight_layout()
            plt.savefig('MetriqueUserbased1000.png')



            labels = ['Precision', 'Recall', 'F1-score']
            plt.figure(figsize=(8, 6))
            plt.bar(labels, [avg_precision, avg_recall, avg_f1], color=['blue', 'green', 'orange'])
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title('Evaluation Globale des Metriques pour le Filtrage basé Contenu ')
            plt.ylim(0, 1)
            plt.grid(True)
            plt.savefig('MetriqueContentbased300.png')




            ####################################### fig entre trainset et testset



            # Création de la figure pour afficher les résultats
            tuples = [(k, min_k) for k in param_grid['k'] for min_k in param_grid['min_k']]
            tuple_labels = [f'({k},{min_k})' for k, min_k in tuples]

            plt.figure(figsize=(18, 16))

            plt.subplot(3, 2, 1)
            plt.plot(tuple_labels, [results_train[tpl]['rmse'] for tpl in tuples], marker='o', label='Train RMSE')
            plt.plot(tuple_labels, [results_test[tpl]['rmse'] for tpl in tuples], marker='o', linestyle='dashed', label='Test RMSE')
            plt.title('RMSE vs (k, min_k)')
            plt.xlabel('(k, min_k)')
            plt.ylabel('RMSE')
            plt.ylim(0, max(max([results_train[tpl]['rmse'] for tpl in tuples]), max([results_test[tpl]['rmse'] for tpl in tuples])) * 1.1)
            plt.xticks(rotation=90)
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 2, 2)
            plt.plot(tuple_labels, [results_train[tpl]['mae'] for tpl in tuples], marker='o', label='Train MAE')
            plt.plot(tuple_labels, [results_test[tpl]['mae'] for tpl in tuples], marker='o', linestyle='dashed', label='Test MAE')
            plt.title('MAE vs (k, min_k)')
            plt.xlabel('(k, min_k)')
            plt.ylabel('MAE')
            plt.ylim(0, max(max([results_train[tpl]['mae'] for tpl in tuples]), max([results_test[tpl]['mae'] for tpl in tuples])) * 1.1)
            plt.xticks(rotation=90)
            plt.legend()
            plt.grid(True)



            plt.subplot(3, 2, 3)
            plt.plot(tuple_labels, [results_train[tpl]['precision'] for tpl in tuples], marker='o', label='Train Precision')
            plt.plot(tuple_labels, [results_test[tpl]['precision'] for tpl in tuples], marker='o', linestyle='dashed', label='Test Precision')
            plt.title('Precision vs (k, min_k)')
            plt.xlabel('(k, min_k)')
            plt.ylabel('Precision')
            plt.ylim(0.2, 1)
            plt.xticks(rotation=90)
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 2, 4)
            plt.plot(tuple_labels, [results_train[tpl]['recall'] for tpl in tuples], marker='o', label='Train Recall')
            plt.plot(tuple_labels, [results_test[tpl]['recall'] for tpl in tuples], marker='o', linestyle='dashed', label='Test Recall')
            plt.title('Recall vs (k, min_k)')
            plt.xlabel('(k, min_k)')
            plt.ylabel('Recall')
            plt.ylim(0.2, 1)
            plt.xticks(rotation=90)
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 2, 5)
            plt.plot(tuple_labels, [results_train[tpl]['f1'] for tpl in tuples], marker='o', label='Train F1-score')
            plt.plot(tuple_labels, [results_test[tpl]['f1'] for tpl in tuples], marker='o', linestyle='dashed', label='Test F1-score')
            plt.title('F1-score vs (k, min_k)')
            plt.xlabel('(k, min_k)')
            plt.ylabel('F1-score')
            plt.ylim(0.2, 1)
            plt.xticks(rotation=90)
            plt.legend()
            plt.grid(True)

            '''
            plt.subplot(2, 2, 4)
            plt.plot(tuple_labels, [results_train[tpl]['accuracy'] for tpl in tuples], marker='o', label='Train Accuracy')
            plt.plot(tuple_labels, [results_test[tpl]['accuracy'] for tpl in tuples], marker='o', linestyle='dashed', label='Test Accuracy')
            plt.title('Accuracy vs (k, min_k)')
            plt.xlabel('(k, min_k)')
            plt.ylabel('Accuracy')
            plt.ylim(0.2, 1)
            plt.xticks(rotation=90)
            plt.legend()
            plt.grid(True)
            '''

            plt.tight_layout()
            plt.savefig('Performance_Train_Test1000.png')





        finally:
            sys.stdout = sys.__stdout__

    print("Training Done")



if __name__ == "__main__":
    train_model()
