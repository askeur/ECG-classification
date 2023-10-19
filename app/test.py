import os
import pickle
import seaborn as sns

# Import des bibliothèques
import pandas as pd
import streamlit as st
import wfdb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import scipy.io
from util import classify, set_background, PAgeIntro
import numpy as np
from visualization.visualize_ecg  import (
    convert_label,plot_ecg_signals,
    generate_pie_chart,
    plot_class_distribution,
    display_descriptive_statistics)
from css import hide_streamlit_style
import easygui

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


#from src.visualization.visualize_ecg import plot_ecg_signals

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(
    page_title='Heartbeat Classification',
    # anatomical heart favicon
    page_icon="https://api.iconify.design/openmoji/anatomical-heart.svg?width=500",
    #page_icon="❤️",
    layout='wide'
)


#Formatting ---------------------------------#



st.markdown(hide_streamlit_style, unsafe_allow_html=True)



# Définition des variables globales

# Définir les bases de données et les modèles disponibles
bases_de_donnees = ["Kaggle", "MIT", "PTB"]
modeles = ["Random Forest", "SVM", "KNN", "Logistic Regression"]


uploaded_files = None
selected_directory = ""
uploaded_atr = []
uploaded_hea = []
uploaded_dat = []
uploaded_file = []
output_path = ""
model_option = None
folder_selected = False

# Fonction principale
def main():
    global folder_selected, output_path, uploaded_files, uploaded_atr, uploaded_hea, uploaded_dat  # Déclarez les variables comme globales

    # Sélectionnez un fichier CSV
    with st.sidebar.header('█▓▒­░⡷⠂CSV File⠐⢾░▒▓█'):
        uploaded_files = st.sidebar.file_uploader("Téléchargez un fichier CSV", type=["csv"])

    st.sidebar.header('█▓▒­░⡷⠂ECG Signal⠐⢾░▒▓█')

    # Utilisez un bouton pour sélectionner le dossier
    if not folder_selected:
        if st.sidebar.button("Sélectionner un dossier de sortie"):
            selected_directory = easygui.diropenbox()
            output_path = selected_directory + '\\'
            if selected_directory:
                folder_selected = True

    # Affichez le texte en fonction de l'état de la sélection du dossier
    if folder_selected:
        st.sidebar.text(f"Dossier de sortie sélectionné : {selected_directory}")
    else:
        st.sidebar.text("Aucun dossier sélectionné.")

    # Si output_path est défini, mettez à jour les fichiers téléchargés
    if output_path:
        uploaded_files = st.sidebar.file_uploader("Téléchargez les fichiers ECG", type=["atr", "hea", "dat"],
                                                  accept_multiple_files=True)
        if uploaded_files is not None:
            for uploadedfile in uploaded_files:
                folder_selected = True
                # Obtenir l'extension du fichier
                file_extension = uploadedfile.name.split('.')[-1].lower()
                if file_extension == "atr":
                    uploaded_atr.append(uploadedfile)
                elif file_extension == "hea":
                    uploaded_hea.append(uploadedfile)
                elif file_extension == "dat":
                    uploaded_dat.append(uploadedfile)

# Choisissez un modèle si le dossier de sortie est sélectionné
if uploaded_files is not None:

    # Afficher l'en-tête dans la barre latérale
    with st.sidebar.header('Choisissez une base de données'):
        base_de_donnees_selectionnee = st.sidebar.selectbox("Base de données", bases_de_donnees)

    # Afficher le choix du modèle si une base de données est sélectionnée
    if base_de_donnees_selectionnee:
        with st.sidebar.header('Choisissez un modèle'):
            modele_selectionne = st.sidebar.selectbox("Modèle", modeles)

if __name__ == "__main__":
    main()


set_background('ezgif.com-gif-maker.gif')#HeartGif.gif
# Affichez du contenu Streamlit
# PAge Intro

# Contenu de la page d'accueil
st.markdown( PAgeIntro, unsafe_allow_html=True)

# Titre à droite
# st.markdown('<h1>Heartbeat Classification</h1>', unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding', False)


# Lire les fichiers et segmenter les signaux ECG
all_segments = []
all_labels = []



model_path = 'models/weights-best.hdf5'
classes = ['Normal','Atrial Fibrillation','Other','Noise']

# Définition des valeurs et des labels pour chaque base de données
labels = ['Battements normaux (0.0)', 'Battements supraventriculaires (1.0)', 'Battements ventriculaires (2.0)', 'Battements de fusion (3.0)', 'Battements inconnus (4.0)']



@st.cache_data  # 👈 Add the caching decorator
def get_model(model_path):
    model = load_model(f'{model_path}')
    return model


st.sidebar.markdown("---------------")

#---------------------------------#
# Data preprocessing and Model building

@st.cache
def read_ecg_preprocessing(records_dir,atr_file_name, hea_file_name, dat_file_name,atr_contents, hea_contents, dat_contents):
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

# Check if .atr, .hea, and .dat files are uploaded

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
            train_data, test_data, segmented_data = read_ecg_preprocessing(records_dir, atr_file_name, hea_file_name,
                                                                           dat_file_name, atr_contents, hea_contents,
                                                                           dat_contents)

            # Save the training and test sets as CSV files
            train_csv_path = os.path.join(output_path, f"{os.path.splitext(atr_file_name)[0]}_segmented_train.csv")
            test_csv_path = os.path.join(output_path, f"{os.path.splitext(atr_file_name)[0]}_segmented_test.csv")
            train_data.to_csv(train_csv_path, index=False)
            test_data.to_csv(test_csv_path, index=False)

            # Display success messages
            st.success(f"Enregistrement terminé ! Le fichier segmenté train a été enregistré sous {train_csv_path}")
            st.success(f"Enregistrement terminé ! Le fichier segmenté test a été enregistré sous {test_csv_path}")

            # Create the download button for segmented data
            csv_data = segmented_data.to_csv(index=False).encode()
            st.download_button(
                label="Télécharger le fichier CSV segmenté",
                data=csv_data,
                file_name=f"{os.path.splitext(atr_file_name)[0]}_segmented.csv",
                key='segmented_csv'
            )
        else:
            st.warning("Veuillez télécharger tous les fichiers requis : .atr, .hea et .dat")


# Rest of your Streamlit app

@st.cache_data  # 👈 Add the caching decorator
def get_prediction(data,model):
    prob = model(data)
    ann = np.argmax(prob)
    #true_target =
    #print(true_target,ann)

    #confusion_matrix[true_target][ann] += 1
    return classes[ann],prob #100*prob[0,ann]

# Visualization --------------------------------------
# Partie 2 : Classification des battements de cœur
# --------------------------------------------------
# Charger les modèles pré-entraînés



model_file_path = 'models/RandomForest.pkl'

# Ouvrez le fichier en mode lecture binaire ('rb') et chargez le modèle
#with open(model_file_path, 'rb') as model_file:
    #random_forest_model = pickle.load(model_file)

# ---------------------------------#
# Main panel

# model = get_model(f'{model_path}')
# Si output_path est défini, mettez à jour les fichiers téléchargés

# Si output_path est défini, mettez à jour les fichiers téléchargés
if uploaded_files is not None:
    for uploadedfile in uploaded_files:
     # Charger le fichier CSV
        df = pd.read_csv(uploaded_file, header=None, encoding='ISO-8859-1')

        # Vérifier si le fichier CSV est vide
        if df.empty:
            st.warning("Le fichier CSV ne contient aucune donnée.")
        else:
            # Appliquer la fonction de conversion aux étiquettes des données df
            df[187] = df[187].apply(convert_label)

            # Supprimer la première ligne de chaque fichier CSV si nécessaire
            df = df.drop([0])
            # Diviser la page en deux colonnes avec une largeur égale
            col1, col2 = st.columns(2)

            with col1:  # Colonne 1 - Visualisation des données
                st.markdown('<font color="#dbdada"><h4> Visualize ECG</h4></font>', unsafe_allow_html=True)
                st.write(df)

                # Afficher les statistiques descriptives dans la colonne 1
                st.markdown('<font color="#dbdada"><h4>Statistiques descriptives</h4></font>',
                        unsafe_allow_html=True)
                st.write(display_descriptive_statistics(df, "df"))

            with col2:  # Colonne 2 - Visualisation des ECG et résultats numériques
                st.markdown('<font color="#dbdada"><h4>Visualiser les ECG et résultats numériques</h4></font>', unsafe_allow_html=True)
                # Afficher les signaux ECG dans la colonne 2
                ecg_signals_fig = plot_ecg_signals(df, "Signaux ECG")
                st.pyplot(ecg_signals_fig)

                st.markdown('<font color="#dbdada"><h4>Répartition des classes</h4></font>',
                        unsafe_allow_html=True)
                # Afficher la répartition des classes dans la colonne 2

                plot_class_distribution(df, "Répartition des classes")


    # st.line_chart(np.concatenate(ecg).ravel().tolist())



