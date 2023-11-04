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
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from tensorflow.keras.utils import to_categorical

classes = ['Catégorie_0', 'Catégorie_1', 'Catégorie_2', 'Catégorie_3',
                                                   'Catégorie_4']


def convert_label(label):
    if label in ['N']:
        return 0
    elif label in ['L', 'R', 'A', 'a', 'J', 'S', 'j']:
        return 1
    elif label in ['V', 'E']:
        return 2
    elif label in ['F']:
        return 3
    else:
        return 4


def read_ecg_data(dat_file_path, hea_file_path, atr_file_path, start_sample, end_sample):
    record = wfdb.rdrecord(os.path.splitext(dat_file_path)[0], sampfrom=start_sample, sampto=end_sample)
    annotation = wfdb.rdann(os.path.splitext(dat_file_path)[0], 'atr', sampfrom=start_sample, sampto=end_sample)
    ecg_signals = pd.DataFrame(record.p_signal, columns=['Signal 1', 'Signal 2'])
    rythm_annotations = annotation.symbol
    beat_locations = annotation.sample
    rythm_categories = [convert_label(label) for label in rythm_annotations]
    return ecg_signals, rythm_annotations, beat_locations, rythm_categories


def display_ecg_section(ecg_signals, fs):
    st.markdown('<h4>ECG Signals</h4>', unsafe_allow_html=True)
    # Convert sample indices to minutes
    time_minutes = np.arange(0, len(ecg_signals)) / fs / 60

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_minutes, ecg_signals['Signal 1'], label='Signal 1')
    ax.plot(time_minutes, ecg_signals['Signal 2'], label='Signal 2')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    st.pyplot(fig)


def display_initial_rythm_annotations(rythm_annotations):
    st.markdown('<h4>Statistical Analysis of Initial Annotations</h4>', unsafe_allow_html=True)
    st.write(pd.DataFrame({'Rythme': rythm_annotations}).groupby('Rythme').size())



def display_other_annotations(rythm_annotations, beat_locations, fs):
    st.markdown('<h4>Other Annotations (excluding N)</h4>', unsafe_allow_html=True)
    other_annotations = []
    other_locations = []

    for i, annotation in enumerate(rythm_annotations):
        if annotation != 'N':
            other_annotations.append(annotation)
            other_locations.append(beat_locations[i])

    # Convertir les positions en secondes
    other_locations_seconds = [loc / fs for loc in other_locations]


    st.write(pd.DataFrame({'Annotation': other_annotations, 'Position (secondes)': other_locations_seconds}))

    # Si vous voulez également la position en heures, ajoutez :
    # other_locations_hours = [loc / fs / 3600 for loc in other_locations]
    # Et ajoutez 'Position (heures)': other_locations_hours au DataFrame

def categorize_and_segment_ecg(records_dir, atr_file_name, hea_file_name, dat_file_name, atr_contents, hea_contents, dat_contents):
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

    # Categorize and segment the ECG signals
    for i in range(len(annotation.symbol)):
        symbol = annotation.symbol[i]
        label = convert_label(symbol)
        start = annotation.sample[i]
        end = start + 187  # Segment of 187 points

        # Extract the ECG signal segment
        segment = record.p_signal[start:end, 0]  # Use the first derivation

        # Add the segment and label to the lists
        if segment.shape[0] == 187:  # Ignore segments that are too short
            all_segments.append(segment)
            all_labels.append(label)

    # Convert to a DataFrame
    segmented_data = pd.DataFrame(all_segments)
    segmented_data['Label'] = all_labels

    return segmented_data

def display_patient_metadata(hea_file_path):
    # Read metadata
    metadata = wfdb.rdheader(os.path.splitext(hea_file_path)[0])
    patient_info = metadata.comments

    # Display metadata in a centered and italicized format
    st.subheader("Patient Information")
    for info in patient_info:
        formatted_info = f'<p style="font-style: italic; color: #c9621e;">{info}</p>'
        st.markdown(formatted_info, unsafe_allow_html=True)

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
    """Charge le modèle WaveNet."""
    model = tf.keras.models.load_model(model_path)

    #model = load_model(model_path)
    return model


def predict(model, data):
    """Effectue une prédiction avec le modèle WaveNet."""
    return model.predict(data)


@st.cache_data ()
def get_prediction(data, model):
    prob = model(data)
    predictions = model.predict(data)
    predicted_class = np.argmax(prob)

    if 0 <= predicted_class < prob.shape[1]:  # Check if predicted_class is within bounds
        confidence = prob[0, predicted_class] * 100
    else:
        predicted_class = -1  # Set a default value for invalid predicted_class
        confidence = 0  # Set confidence to 0 for invalid predicted_class

    return classes[predicted_class], confidence, predictions



def preprocess_ecg_files(uploaded_files):
    atr_file_name = None
    hea_file_name = None
    dat_file_name = None
    atr_file_value = None
    hea_file_value = None
    dat_file_value = None

    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == "atr":
            atr_contents = uploaded_file.read()
            atr_file_name = uploaded_file.name
            atr_file_value = uploaded_file.getvalue()
        elif file_extension == "hea":
            hea_contents = uploaded_file.read()
            hea_file_name = uploaded_file.name
            hea_file_value = uploaded_file.getvalue()

        elif file_extension == "dat":
            dat_contents = uploaded_file.read()
            dat_file_name = uploaded_file.name
            dat_file_value = uploaded_file.getvalue()

    # Ensure that all required files are uploaded
    if atr_file_name and hea_file_name and dat_file_name:
        folder = os.path.join(os.path.expanduser("~"), "Downloads", f"{os.path.splitext(atr_file_name)[0]}")

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Get the directory containing the .atr, .hea, and .dat files
        # Obtenir le nom du répertoire contenant les fichiers .atr et .hea
        records_dir = os.path.dirname(atr_file_name)
        dat_path = dat_file_name
        hea_path = hea_file_name
        atr_path = atr_file_name
        with open(dat_path, 'wb') as dat_writer:
            dat_writer.write(dat_contents)
        with open(hea_path, 'wb') as hea_writer:
            hea_writer.write(hea_contents)
        with open(atr_path, 'wb') as atr_writer:
            atr_writer.write(atr_contents)

        # Display patient metadata
        display_patient_metadata(hea_path)
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
            ecg_signals, rythm_annotations, beat_locations, rythm_categories = read_ecg_data(dat_path, hea_path,
                                                                                             atr_path, start_sample,
                                                                                             end_sample)

        # Diviser l'écran en deux colonnes
        col1, col2 = st.columns(2)

        # Colonne 1
        with col1:
            # Section Header: Training Data

            display_ecg_section(ecg_signals, fs)
        with col2:
            display_initial_rythm_annotations(rythm_annotations)
            display_other_annotations(rythm_annotations, beat_locations, fs)
            rythm_categories = [convert_label(label) for label in rythm_annotations]
            st.markdown('<h4>Cardiac Anomalies Statistical Analysis</h4>', unsafe_allow_html=True)
            st.write(pd.DataFrame({'Catégorie': rythm_categories}).groupby('Catégorie').size())

        if st.sidebar.button('Save Categorized Data'):
            df = pd.DataFrame({'Catégorie': rythm_categories})
            df.to_csv(os.path.join(folder, "categorized_data.csv"), index=False)

            st.sidebar.success("Categorized data saved successfully!")


        if st.sidebar.button("Categorize and Segment ECG Signal"):
            with st.expander("Preprocessing Details", expanded=True):
                st.write("The preprocessing includes the following steps:")
                st.write("1. Saving .atr, .hea, and .dat files.")
                st.write("2. Reading the ECG record and annotations.")
                st.write("3. Detecting R-peaks and segmenting the ECG signal.")
                st.write("4. Performing stratified k-fold cross-validation")
            segmented_data = categorize_and_segment_ecg(os.getcwd(), atr_path, hea_path, dat_path,
                                                           atr_file_value,
                                                           hea_file_value, dat_file_value)
            segmented_data.rename(columns={"Label": "187"}, inplace=True)

            segmented_data.to_csv(os.path.join(folder, "segmented_data.csv"), index=False)
            st.sidebar.success('ECG Signal categorized and segmented successfully!')
            # Diviser l'écran en deux colonnes
            col1, col2 = st.columns(2)

            # Colonne 1
            with col1:
                # Section Header: Training Data
                st.markdown('<h4>Data segmted</h4>', unsafe_allow_html=True)

                # Description: This section displays information about the training data.
                st.write("This section displays information about the training data.")
                st.write(segmented_data)

            with col2:
                # Section Header: Class Distribution for Training Data
                st.markdown('<h4>Class Distribution segmented Data</h4>', unsafe_allow_html=True)

                # Description: Visualizes the distribution of classes within the training data.
                # st.write("This section visualizes the distribution of classes within the training data.")
                with st.expander("This section visualizes the distribution of classes within the segmented data.",
                                 expanded=True):
                    st.write(
                        "The 'plot_class_distribution' function generates two visualizations for class distribution:")
                    st.write("1. A count plot that shows the distribution of classes in the dataset.")
                    st.write("2. A pie chart that visualizes the percentage distribution of classes.")
                    st.write("This helps in understanding the balance or imbalance of classes in the dataset.")
                vis.plot_class_distribution(segmented_data, "Class Distribution - segmented Data")

        if st.sidebar.button("Save signals and annotations"):
            ecg_signals.to_csv(os.path.join(folder, "signals.csv"), index=False)
            annotations_df = pd.DataFrame({'Rhythm': rythm_annotations, 'Location': beat_locations})
            annotations_df.to_csv(os.path.join(folder, "annotations.csv"), index=False)
            st.sidebar.success('Files signals.csv and annotations.csv saved successfully!')

        if st.sidebar.button("Clean files"):
            os.unlink(dat_path)
            os.unlink(hea_path)
            os.unlink(atr_path)
            st.sidebar.success('Files cleaned successfully!')

        # Call the preprocessing function
        #train_data, test_data, segmented_data = ecg_signal_segment(records_dir, atr_file_name,
                                                                       #hea_file_name,
                                                                       #dat_file_name, atr_contents,
                                                                       #hea_contents,
                                                                       #dat_contents)

        #st.sidebar.success("Processing completed!")
        # Next, update the expander with preprocessing completion details.
        #with st.expander("Preprocessing Details", expanded=True):
            #st.success("Training and testing data have been successfully generated.")
        #download_csv_files(train_data, test_data, atr_file_name)

        #else:
        #st.warning("Please upload all the required files: .atr, .hea, and .dat")


def Classification(uploaded_files,model,isWaveNet):
    segmented_data = pd.read_csv(uploaded_files)

    counts = None  # Initialisation par défaut

    with st.spinner(text="Running  Model..."):
        if isWaveNet:
            X = segmented_data.drop(columns=["187"]).values
            y_test = segmented_data["187"].values


            predictions = predict(model, X)

        else:
            # Préparer les ensembles de train et de test
            train_size = int(0.8 * len(segmented_data))

            y_train = segmented_data.iloc[:, -1].values
            y_test = segmented_data.iloc[:, -1].values

            # Extraire les caractéristiques des données
            X_train = segmented_data.iloc[:, :-1].values.reshape((len(segmented_data), 1, -1))
            X_test = segmented_data.iloc[:, :-1].values.reshape((len(segmented_data), 1, -1))
            X_train_flat = X_train.reshape((len(X_train), -1))
            X_test_flat = X_test.reshape((len(X_test), -1))

            # Réorganiser les données d'entraînement pour que chaque ligne soit un enregistrement et chaque colonne une variable

            X_train = np.array([x.flatten() for x in X_train])
            X_test = np.array([x.flatten() for x in X_test])
            predictions = predict(model, X_test)


        # Diviser l'écran en deux colonnes
        col1, col2 = st.columns(2)

        # Colonne 1
        with col1:
            # 2. Affichage des résultats
            st.markdown('<h4>Prediction results</h4>', unsafe_allow_html=True)
            if isWaveNet :
                # Convertir les prédictions en catégories (si nécessaire)
                predicted_categories = [np.argmax(pred) for pred in predictions]

                # Compter le nombre d'annotations par catégorie
                counts = pd.Series(predicted_categories).value_counts()
                st.write(counts)

                # Génération d'un rapport de classification
                st.markdown('<h4>Classification Report</h4>', unsafe_allow_html=True)
                y_true = np.round(y_test).astype(int)

                y_pred = np.argmax(predictions, axis=1)
                unique_classes = np.unique(y_true)

                target_names = [f"Classe {i}" for i in unique_classes]

                report = classification_report(y_true, y_pred, labels=unique_classes, target_names=target_names,
                                               zero_division=0)

                st.text_area(label="Report", value=report, height=350)


                # Ajouter d'autres visualisations ou métriques de performance si nécessaire

                # 3. Téléchargement des résultats
                if len(predictions.shape) == 1:
                    predictions = predictions.reshape(-1, 1)

                # Créez un DataFrame en utilisant les prédictions et les noms de colonnes
                column_names = ['Catégorie_0', 'Catégorie_1', 'Catégorie_2', 'Catégorie_3', 'Catégorie_4']

                predictions_df = pd.DataFrame(predictions, columns=column_names)

                # Créez le texte que vous voulez afficher à l'intérieur du bouton de téléchargement
                button_text = '<span style="color: black;">Download predictions</span>'

                # Utilisez st.markdown pour afficher le texte personnalisé
                st.markdown(button_text, unsafe_allow_html=True)

                # Ajoutez le bouton de téléchargement
                st.download_button("Download predictions", predictions_df.to_csv(index=False), "predictions.csv")

            else:
                st.markdown('<h4>Classification Report</h4>', unsafe_allow_html=True)
                unique_classes = np.unique(y_test)
                target_names = [f"Classe {i}" for i in unique_classes]

                report = classification_report(y_test, predictions, labels=unique_classes, target_names=target_names,
                                               zero_division=0)
                Accuracy_train = round(model.score(X_train_flat, y_train) * 100, 2)
                model_accuracy = round(accuracy_score(predictions, y_test) * 100, 2)

                st.text_area(label="Report", value=report, height=350)

                st.write(f"Training Accuracy: {Accuracy_train}%")
                st.write(f"Model Accuracy Score: {model_accuracy}%")
                # Ajouter d'autres visualisations ou métriques de performance si nécessaire



        with col2:
            if isWaveNet:
                st.markdown('<h4>Prediction Chart</h4>', unsafe_allow_html=True)
                # Create a chart to display the predictions
                plt.figure(figsize=(8, 6))
                plt.bar(range(len(predicted_categories)), predicted_categories)
                plt.xlabel("Sample")
                plt.ylabel("Prediction Category")
                st.pyplot(plt)  # Display the chart in Streamlit
            else:
                with st.spinner(text="Prédictions Model..."):
                    st.markdown('<h4>Prediction Visualization</h4>', unsafe_allow_html=True)
                    predictions_df = pd.DataFrame({'True Class': y_test, 'Predicted Class': predictions})
                    st.bar_chart(predictions_df['Predicted Class'].value_counts())
                    st.markdown("---")

                    st.markdown('<h4>Confusion Matrix</h4>', unsafe_allow_html=True)

                    cm = confusion_matrix(y_test, predictions)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"],
                                    yticklabels=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"])
                    plt.xlabel('Predictions')
                    plt.ylabel('True Labels')
                    plt.title('Confusion Matrix')
                    st.pyplot(fig)
                    st.set_option('deprecation.showPyplotGlobalUse', False)





def run_mit_visualize(uploaded_files):
    if uploaded_files:
        preprocess_ecg_files(uploaded_files)

def run_mit_classification(uploaded_file, model,isWaveNet):
    if uploaded_file:
        Classification(uploaded_file,model,isWaveNet)


#run_mit_ui(choice)