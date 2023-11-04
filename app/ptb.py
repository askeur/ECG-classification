import numpy as np
import streamlit as st
import pandas as pd
import wfdb
import os
from matplotlib import pyplot as plt
import visualization.visualize_ecg as vis
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns


def read_ecg_data(dat_file_path, hea_file_path, start_sample, end_sample):
    record = wfdb.rdrecord(os.path.splitext(dat_file_path)[0], sampfrom=start_sample, sampto=end_sample, channels=[0, 1])
    ecg_signals = pd.DataFrame(record.p_signal, columns=[f"Signal {i+1}" for i in range(record.p_signal.shape[1])])
    return ecg_signals



def display_ecg_section(ecg_signals, fs):
    st.markdown('<h4>ECG Signals</h4>', unsafe_allow_html=True)

    time_seconds = np.arange(0, len(ecg_signals)) / fs

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in ecg_signals.columns:
        ax.plot(time_seconds, ecg_signals[col], label=col)
    ax.set_xlabel('Temps (secondes)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    st.pyplot(fig)

def segment_ecg_signal(dat_file_path, hea_file_path, segment_length=187):
    # Lire l'enregistrement complet
    record = wfdb.rdrecord(os.path.splitext(dat_file_path)[0])

    # Segmenter le signal ECG
    total_samples = record.p_signal.shape[0]
    num_segments = total_samples // segment_length
    segments = []

    # Segmenter le signal en morceaux continus de 186 points
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = record.p_signal[start:end, 0]
        segments.append(segment)

    # Convertir en DataFrame
    segmented_data = pd.DataFrame(segments)
    return segmented_data

@st.cache_data
def load_jb_model(model_path):
    try:
        model = joblib.load(f'{model_path}')
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None


@st.cache_resource
def load_wavenet_model(model_path):
    """Charge le modèle WaveNet avec l'optimiseur Adam."""
    # Charger le modèle
    model = tf.keras.models.load_model(model_path)

    return model


def predict(model, data):
    """Effectue une prédiction avec le modèle WaveNet."""
    return model.predict(data)


def display_important_patient_info(hea_file_path):
    # Read metadata
    metadata = wfdb.rdheader(os.path.splitext(hea_file_path)[0])
    patient_info = metadata.comments

    # Define a list of keys that represent important information
    important_keys = [
        "age:",
        "sex:",
        "Reason for admission:",
        "Number of coronary vessels involved:",
        "Medication pre admission:",
        "Medication after discharge:"
    ]

    # Display only the important information
    st.subheader("Important Patient Information")
    for info in patient_info:
        for key in important_keys:
            if key in info:
                formatted_info = f'<p style="font-style: italic; color: #c9621e;">{info}</p>'
                st.markdown(formatted_info, unsafe_allow_html=True)


def preprocess_ecg_files(uploaded_files):
    xyz_file_name = None
    hea_file_name = None
    dat_file_name = None


    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == "xyz":
            xyz_contents = uploaded_file.read()
            xyz_file_name = uploaded_file.name

        elif file_extension == "hea":
            hea_contents = uploaded_file.read()
            hea_file_name = uploaded_file.name


        elif file_extension == "dat":
            dat_contents = uploaded_file.read()
            dat_file_name = uploaded_file.name


    # Ensure that all required files are uploaded
    if xyz_file_name and hea_file_name and dat_file_name:
        folder = os.path.join(os.path.expanduser("~"), "Downloads", f"{os.path.splitext(xyz_file_name)[0]}")

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Get the directory containing the .atr, .hea, and .dat files
        # Obtenir le nom du répertoire contenant les fichiers .atr et .hea
        records_dir = os.path.dirname(xyz_file_name)
        dat_path = dat_file_name
        hea_path = hea_file_name
        xyz_path =xyz_file_name
        with open(dat_path, 'wb') as dat_writer:
            dat_writer.write(dat_contents)
        with open(hea_path, 'wb') as hea_writer:
            hea_writer.write(hea_contents)
        with open(xyz_path, 'wb') as atr_writer:
            atr_writer.write(xyz_contents)

        # Display patient metadata
        display_important_patient_info(hea_path)
        record = wfdb.rdheader(os.path.splitext(dat_path)[0])
        fs = record.fs  # fréquence d'échantillonnage

        # Specify the time range in seconds
        start_time = st.sidebar.number_input('Start Time (in seconds)', 0.0)
        end_time = st.sidebar.number_input('End Time (in seconds)', 1.0)

        # Convert the start and end times to sample points
        start_sample = int(start_time * record.fs)
        end_sample = int(end_time * record.fs)

        # Ensure that end_sample is greater than start_sample
        if end_sample <= start_sample:
            st.error("The value of 'End time' must be greater than 'Start time'.")
        else:
            ecg_signals = read_ecg_data(dat_path, hea_path, start_sample, end_sample)


        display_ecg_section(ecg_signals, fs)

        if st.sidebar.button("Segment ECG Signal"):
            segments = segment_ecg_signal(dat_path, hea_path)
            st.sidebar.success(f"{len(segments)}segments created successfully!")

            segmented_data = pd.DataFrame(segments)
            segmented_data.to_csv(os.path.join(folder, "segmented_data.csv"), index=False)

            # Diviser l'écran en deux colonnes
            col1, col2 = st.columns(2)

            # Colonne 1
            with col1:
                # Section Header: Training Data
                st.markdown('<h4>Data segmted</h4>', unsafe_allow_html=True)

                # Description: This section displays information about the training data.
                st.write("This section displays information about the segmented data.")
                print(segmented_data)
                st.write(segmented_data)

            with col2:
                print("colonnne")
                print(segmented_data.columns)
                # Section Header: Class Distribution for Training Data
                st.markdown('<h4>Class Distribution - segmented Data</h4>', unsafe_allow_html=True)

                with st.expander("This section visualizes the distribution of classes within the PTBDB data.",
                                 expanded=True):
                    st.write(
                        "The 'plot_class_distribution' function generates two visualizations for class distribution:")
                    st.write("1. A count plot that shows the distribution of classes in the dataset.")

                    st.write("This helps in understanding the balance or imbalance of classes in the dataset.")

                    # Call the plot_class_distribution function to visualize class distribution for PTBDB
                    vis.plot_ptbdb_class_distribution(segmented_data.head(10), "Class Distribution - segmented Data")





        if st.sidebar.button("Clean files"):
            os.unlink(dat_path)
            os.unlink(hea_path)
            os.unlink(xyz_path)
            st.sidebar.success('Files cleaned successfully!')

def run_ptb_visualize(uploaded_file):
    if uploaded_file:
        preprocess_ecg_files(uploaded_file)

def Classification(uploaded_file,model,isWaveNet):
    segmented_data = pd.read_csv(uploaded_file)

    with st.spinner(text="Running  Model..."):

        if isWaveNet:
            X = segmented_data.values

            y_test = segmented_data.values
            predictions = predict(model, X)
        else:
            # Préparer les ensembles de train et de test
            train_size = int(0.8 * len(segmented_data))

            X_train = np.array(segmented_data.iloc[:train_size].values)
            X_test = np.array(segmented_data.iloc[train_size:].values)

            y_train= np.array(segmented_data.iloc[train_size:].values)
            y_test = np.array(segmented_data.iloc[train_size:].values)
            predictions = predict(model, X_test)


        # Diviser l'écran en deux colonnes
        col1, col2 = st.columns(2)

        # Colonne 1
        with col1:
            if isWaveNet== False:
                if len(predictions.shape) == 1:
                    # Réorganisez les prédictions pour les structurer correctement
                    predictions = np.column_stack((1 - predictions, predictions))
            predicted_categories = ["Normal" if pred[0] > 0.5 else "Anormal" for pred in predictions]

            st.markdown('<h4>Prediction results</h4>', unsafe_allow_html=True)
            st.write(pd.Series(predicted_categories).value_counts())
            predictions_df = pd.DataFrame({
            'Probabilité_Normal': [pred[0] for pred in predictions],
            'Catégorie': predicted_categories
            })

        if isWaveNet:

            st.download_button("Download predictions", predictions_df.to_csv(index=False), "predictions.csv")

        with col2:
            with st.spinner(text="Creating Prediction Chart..."):

                if isWaveNet:

                    plt.figure(figsize=(8, 6))
                    colors = ['green' if cat == "Normal" else 'red' for cat in predicted_categories]
                    plt.bar(range(len(predicted_categories)), [pred[0] for pred in predictions], color=colors)
                    plt.xlabel("Sample")
                    plt.ylabel("Probability of Normal")
                    st.pyplot(plt)
                else:
                    plt.figure(figsize=(8, 6))

                    # Assurez-vous que predictions est un tableau 2D
                    if len(predictions.shape) == 1:
                        predictions = predictions.reshape(-1, 1)

                    # Maintenant, vous pouvez accéder à l'index 1 (deuxième colonne)
                    probabilities_anormal = predictions[:, 1]
                    probabilities_normal = predictions[:, 0]

                    width = 0.5  # Largeur des barres

                    # Créez un graphique à barres empilées
                    p1 = plt.bar(range(len(predicted_categories)), probabilities_normal, width)
                    p2 = plt.bar(range(len(predicted_categories)), probabilities_anormal, width,
                             bottom=probabilities_normal)

                    plt.xlabel("Sample")
                    plt.ylabel("Probability")
                    plt.legend((p1[0], p2[0]), ('Normal', 'Anormal'))

                    st.pyplot(plt)


def run_ptb_classification(uploaded_file, model,isWaveNet):
    if uploaded_file:
        Classification(uploaded_file,model,isWaveNet)



#run_ptb_ui()

