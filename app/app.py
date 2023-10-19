# Library Importing
import os
import pickle
import seaborn as sns
import pandas as pd
import streamlit as st
import wfdb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import scipy.io
from util import classify, set_background, PAgeIntro, authors
import numpy as np
from visualization.visualize_ecg  import (
    convert_label,plot_ecg_signals,
    generate_pie_chart,
    plot_class_distribution,
    display_descriptive_statistics)
from css import hide_streamlit_style
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt



#Page layout---------------------------------#

st.set_page_config(
    page_title='Heartbeat Classification',
    # anatomical heart favicon
    page_icon="https://api.iconify.design/openmoji/anatomical-heart.svg?width=500",
    layout='wide'
)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#Formatting ---------------------------------#

# Définir les bases de données et les modèles disponibles
bases_de_donnees = ["Kaggle", "MIT", "PTB"]
modeles = ["WaveNet","RandomForest", "SVM", "KNN", "LogisticRegression"]


# Définition des variables globales
uploaded_files = []
train_csv_data = None
test_csv_data  = None
selected_directory = ""
uploaded_atr  = []
uploaded_hea  = []
uploaded_dat  = []
uploaded_file = []
output_path   = ""
modele_selected = None
bd_selected = None
folder_selected = False
show_intro = True  # Initialisation à True pour afficher l'introduction au début


classes = ['Normal','Atrial Fibrillation','Other','Noise']

# Définition des valeurs et des labels pour chaque base de données
labels = ['Battements normaux (0.0)', 'Battements supraventriculaires (1.0)', 'Battements ventriculaires (2.0)', 'Battements de fusion (3.0)', 'Battements inconnus (4.0)']




# Fonction principale



def ecg_signal_segment(records_dir,atr_file_name, hea_file_name, dat_file_name,atr_contents, hea_contents, dat_contents):
  # Save .atr, .hea, and .dat files
    with open(atr_file_name, "wb") as atr_file:
        atr_file.write(atr_contents)
    with open(hea_file_name, "wb") as hea_file:
        hea_file.write(hea_contents)
    with open(dat_file_name, "wb") as dat_file:
        dat_file.write(dat_contents)

    # Read the ECG record
    record = wfdb.rdrecord(os.path.join(records_dir, os.path.splitext(atr_file_name)[0]))
    annotation = wfdb.rdann(os.path.join(records_dir, os.path.splitext(atr_file_name)[0]), 'atr')

    all_segments = []
    all_labels = []

    # Collect ECG signal segments
    for i in range(len(annotation.symbol)):
        symbol = annotation.symbol[i]
        start = annotation.sample[i]
        end = start + 187  # Segment of 187 points

        # Extract the ECG signal segment
        segment = record.p_signal[start:end, 0]  # Use the first derivation

        # Add the segment and label to the lists
        if segment.shape[0] == 187:  # Ignore segments that are too short
            all_segments.append(segment)
            all_labels.append(symbol)

    # Convert to a DataFrame
    segmented_data = pd.DataFrame(all_segments)
    segmented_data['Label'] = all_labels

    # Use stratified k-fold cross-validation to split the data
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for train_index, test_index in skf.split(segmented_data, segmented_data['Label']):
        train_data = pd.concat([train_data, segmented_data.iloc[train_index]])
        test_data = pd.concat([test_data, segmented_data.iloc[test_index]])



    return train_data, test_data, segmented_data


def ecg_preprocessing():
    if uploaded_files:
        # Ajoutez un bouton "Preprocessing" à votre barre latérale
        if st.sidebar.button("Preprocessing"):
            atr_file_name = None
            hea_file_name = None
            dat_file_name = None

            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == "atr":
                    atr_contents = uploaded_file.read()
                    atr_file_name = uploaded_file.name

                elif file_extension == "hea":
                    hea_contents = uploaded_file.read()
                    hea_file_name = uploaded_file.name
                elif file_extension == "dat":
                    dat_contents = uploaded_file.read()
                    dat_file_name = uploaded_file.name

            # Ensure that all required files are uploaded
            if atr_file_name and hea_file_name and dat_file_name:
                # Get the directory containing the .atr, .hea, and .dat files
                # Obtenir le nom du répertoire contenant les fichiers .atr et .hea
                records_dir = os.path.dirname(atr_file_name)

                # Call the preprocessing function
                train_data, test_data, segmented_data = ecg_signal_segment(records_dir, atr_file_name,
                                                                               hea_file_name,
                                                                               dat_file_name, atr_contents,
                                                                               hea_contents,
                                                                               dat_contents)

                st.sidebar.success("Processing completed!")

                # Créez les boutons de téléchargement pour les données d'entraînement et de test
                with st.sidebar:
                    st.markdown('### Download CSV Files')

                    if st.button("Download Training CSV File"):
                        train_csv = train_data.to_csv(index=False).encode()
                        download_button_str = f"Download Train CSV"
                        st.download_button(label=download_button_str, data=train_csv,
                                           file_name=f"{os.path.splitext(atr_file_name)[0]}_train.csv")

                    if st.button("Download Test CSV File"):
                        test_csv = test_data.to_csv(index=False).encode()
                        download_button_str = f"Download Test CSV"
                        st.download_button(label=download_button_str, data=test_csv,
                                           file_name=f"{os.path.splitext(atr_file_name)[0]}_test.csv")

            else:
                st.warning("Veuillez télécharger tous les fichiers requis : .atr, .hea et .dat")

@st.cache_data
def get_prediction(data, model):
    if model is not None:
        prob = model(data)
        ann = np.argmax(prob)
        return classes[ann], prob
    else:
        return None, None  # Rety_test = df.iloc[:, -1].values




@st.cache_data
def get_model(model_path):
    try:
        model = joblib.load(f'{model_path}')
        return model
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement du modèle : {e}")
        return None

# Fonction pour effectuer la classification des ECG
@st.cache_data

def classification( train_csv_data,test_csv_data, bd_selected, modele_selected):
    model = None

    if train_csv_data is not None and test_csv_data is not None:
        train_data = pd.read_csv(train_csv_data, header=None, encoding='ISO-8859-1')
        test_data = pd.read_csv(test_csv_data, header=None, encoding='ISO-8859-1')

        if train_data.empty or test_data.empty:
            st.warning("The CSV files (train / test) contain no data.")
        else:
            train_data[187] = train_data[187].apply(convert_label)
            test_data[187] = test_data[187].apply(convert_label)

        train_data = train_data.drop([0])
        test_data = test_data.drop([0])

        model_path = f'app/models/{bd_selected}/{modele_selected}/{modele_selected.lower() }.joblib'
        model = get_model(model_path)

    return model, train_data, test_data



@st.cache_data
def load_data(train_csv_data, test_csv_data):
    train_data = pd.DataFrame()  # Initialize as an empty DataFrame
    test_data = pd.DataFrame()   # Initialize as an empty DataFrame

    if train_csv_data is not None and test_csv_data is not None:
        train_data = pd.read_csv(train_csv_data, header=None, encoding='ISO-8859-1')
        test_data = pd.read_csv(test_csv_data, header=None, encoding='ISO-8859-1')

        if train_data.empty or test_data.empty:
            st.warning("The CSV files (train / test) contain no data.")
            train_data = pd.DataFrame()  # Reset to an empty DataFrame
            test_data = pd.DataFrame()   # Reset to an empty DataFrame

    if not train_data.empty and not test_data.empty:
        train_data[187] = train_data[187].apply(convert_label)
        test_data[187] = test_data[187].apply(convert_label)

        train_data = train_data.drop([0])
        test_data = test_data.drop([0])

    return train_data, test_data


    # Your code to work with train_data and test_data here

@st.cache_data
def visualisation(train_csv_data, test_csv_data):
    if train_csv_data is not None and test_csv_data is not None:

        loaded_data = load_data(train_csv_data, test_csv_data)

        if not loaded_data[0].empty and not loaded_data[1].empty:
            train_data, test_data = loaded_data

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<font color="#dbdada"><h4> Visualize ECG - Training Data</h4></font>',
                        unsafe_allow_html=True)

            if train_data is not None:
                st.write(train_data)
            st.markdown('<font color="#dbdada"><h4>Descriptive Statistics - Training Data</h4></font>',
                        unsafe_allow_html=True)
            st.write(display_descriptive_statistics(train_data, "Training Data"))

            st.markdown('<font color="#dbdada"><h4>ECG Signals - Training Data</h4></font>', unsafe_allow_html=True)
            ecg_signals_fig_train = plot_ecg_signals(train_data, "ECG signals - Training Data")
            st.pyplot(ecg_signals_fig_train)

            st.markdown('<font color="#dbdada"><h4>Class Distribution - Training Data</h4></font>',
                        unsafe_allow_html=True)
            plot_class_distribution(train_data, "Class Distribution - Training Data")

        with col2:
            st.markdown('<font color="#dbdada"><h4> Visualize ECG - Test Data</h4></font>', unsafe_allow_html=True)
            if train_data is not None:
                st.write(train_data)

            st.markdown('<font color="#dbdada"><h4>Descriptive Statistics - Test Data</h4></font>',
                        unsafe_allow_html=True)
            st.write(display_descriptive_statistics(test_data, "Test Data"))

            st.markdown('<font color="#dbdada"><h4>ECG Signals - Test Data</h4></font>', unsafe_allow_html=True)
            ecg_signals_fig_test = plot_ecg_signals(test_data, "ECG signals - Test Data")
            st.pyplot(ecg_signals_fig_test)

            st.markdown('<font color="#dbdada"><h4>Class Distribution - Test Data</h4></font>',
                        unsafe_allow_html=True)
            plot_class_distribution(test_data, "Class Distribution - Test Data")


def main():
    global folder_selected, output_path, uploaded_files, train_csv_data,test_csv_data, uploaded_atr, uploaded_hea, uploaded_dat  # Déclarez les variables comme globales
    # Sélectionnez un fichier CSV
    with st.sidebar.header('█▓▒­░⡷⠂ECG Signal⠐⢾░▒▓█'):
        uploaded_files = st.sidebar.file_uploader("Upload ECG Files", type=["atr", "hea", "dat"],
                                                  accept_multiple_files=True)
        ecg_preprocessing()
        # Utilize a button to select the folder
        #if uploaded_files is not None:
            #if not folder_selected:
                #selected_directory = st.sidebar.text_input("Enter the preprocessing folder path:")
                #output_path = selected_directory + '\\'
                #if selected_directory:
                    #folder_selected = True
                    #st.sidebar.text(f"Preprocessing folder selected: {selected_directory}")
                    #ecg_preprocessing()
        #else:
                    #st.sidebar.text("No preprocessing folder selected.")

    with st.sidebar.header('█▓▒­░⡷⠂CSV File⠐⢾░▒▓█'):
        train_csv_data = st.sidebar.file_uploader("Download Training CSV", type=["csv"])
        test_csv_data = st.sidebar.file_uploader("Download Test CSV", type=["csv"])


#Start ---------------------------------#

set_background('app/img/background.gif')#HeartGif.gif


#st.set_option('deprecation.showfileUploaderEncoding', False)


# Partie 1 : Telecharger file csv segmenté ou files .atr( signal)  + preprocessing
# --------------------------------------------------

if __name__ == "__main__":
    main()


# Partie 2 : Visualization/Classification
# --------------------------------------------------
if train_csv_data  is not None and test_csv_data is not None:
    # Afficher l'en-tête dans la barre latérale
    with st.sidebar.header('▓▒­░⡷⠂Choose DB⠐⢾░▒▓█'):
        bd_selected = st.sidebar.selectbox("Base de données", bases_de_donnees)


   # Afficher le choix du modèle si une base de données est sélectionnée
    if bd_selected:
        with st.sidebar.header('▓▒­░⡷⠂Choose Model⠐⢾░▒▓█'):
           modele_selected  = st.sidebar.selectbox("Modèle", modeles)
           # Ajout de l'option oversampling et undersampling
           sampling_option = st.sidebar.radio("Sampling Option", ["None", "Oversampling", "Undersampling"])

           if sampling_option != "None":
               if sampling_option == "Oversampling":
                   modele_selected = modele_selected + "_over"
               elif sampling_option == "Undersampling":
                   modele_selected = modele_selected + "_under"


    # Créer les cases à cocher
    visualization_checkbox = st.sidebar.checkbox("Visualization")
    classification_checkbox = st.sidebar.checkbox("Classification")

    # Mettez à jour show_intro si un fichier CSV est sélectionné
    if classification_checkbox or visualization_checkbox:
        show_intro = False



    # Afficher la page correspondante en fonction des cases cochées
    if visualization_checkbox:
        visualisation(train_csv_data,test_csv_data, )

    # Appel de la fonction
    if classification_checkbox:
        try:
            st.info(f"Prédiction du fichier avec le modèle {modele_selected} basé sur la base de données {bd_selected}")
            model, train_data, test_data = classification(train_csv_data,test_csv_data, bd_selected, modele_selected)

            col1, col2 = st.columns(2)

            with col1:
                with st.spinner(text="Exécution du modèle..."):
                    y_train = train_data.iloc[:, -1].values
                    y_test = test_data.iloc[:, -1].values

                    # Extraire les caractéristiques des données
                    X_train = train_data.iloc[:, :-1].values.reshape((len(train_data), 1, -1))
                    X_test = test_data.iloc[:, :-1].values.reshape((len(test_data), 1, -1))
                    X_notre_train_flat = X_train.reshape((len(X_train), -1))
                    X_notre_test_flat = X_test.reshape((len(X_test), -1))

                    # Réorganiser les données d'entraînement pour que chaque ligne soit un enregistrement et chaque colonne une variable

                    X_train = np.array([x.flatten() for x in X_train])
                    X_test = np.array([x.flatten() for x in X_test])

                    # prédire sur les données selon le model choisi model

                    if model is not None:
                        y_pred = model.predict(X_test)
                        unique_classes = np.unique(y_test)  # Obtenez les classes uniques à partir des données réelles
                        target_names = [f"Classe {i}" for i in unique_classes]  # Créez les noms de classe dynamiquement

                        report = classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names,
                                                       zero_division=0)

                        st.subheader("Rapport de Classification")
                        st.text_area(label="Rapport", value=report, height=350)



                        st.subheader("Analyse des erreurs")
                        if model is not None:
                            y_pred = model.predict(X_test)
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                        xticklabels=["Classe 0", "Classe 1", "Classe 2", "Classe 3", "Classe 4"],
                                        yticklabels=["Classe 0", "Classe 1", "Classe 2", "Classe 3", "Classe 4"])
                            plt.xlabel('Prédictions')
                            plt.ylabel('Vraies étiquettes')
                            plt.title('Matrice de Confusion')
                            st.pyplot(fig)
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            false_positives = cm.sum(axis=0) - np.diag(cm)
                            false_negatives = cm.sum(axis=1) - np.diag(cm)
                            st.subheader("Erreurs de Prédiction")
                            st.write(f"Faux Positifs par Classe : {false_positives}")
                            st.write(f"Faux Négatifs par Classe : {false_negatives}")
                        else:
                            st.error("Le modèle n'a pas été correctement chargé.")

                    else:
                        st.error("Le modèle n'a pas été correctement chargé.")

            with col2:
                st.subheader("Visualisation des prédictions")
                if model is not None:
                    predictions_df = pd.DataFrame({'Vraie Classe': y_test, 'Classe Prédite': y_pred})
                    st.bar_chart(predictions_df['Classe Prédite'].value_counts())

                else:
                    st.error("Le modèle n'a pas été correctement chargé.")

                st.subheader("Résumé des prédictions")

                if model is not None:
                    y_pred = model.predict(X_test)
                    summary_df = pd.DataFrame({'Vraies étiquettes': y_test, 'Prédictions': y_pred})
                    st.text_area(label="Vraies étiquettes/Prédictions", value=summary_df.to_string(), height=480)

                else:
                    st.error("Le modèle n'a pas été correctement chargé.")

        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}")
# PAge Intro
# Contenu de la page d'accueil
if show_intro:
    st.markdown(PAgeIntro, unsafe_allow_html=True)
    st.markdown(authors, unsafe_allow_html=True)
    st.image('app/img/model_learing.gif')






