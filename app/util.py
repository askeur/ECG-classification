import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np

PAgeIntro  =""" # Classification des Battements Cardiaques

Dans le domaine médical, l'interprétation manuelle des enregistrements d'activité électrique cardiaque (ECG) peut être une tâche laborieuse et sujette à des erreurs. Pour relever ce défi, notre projet s'est concentré sur la classification automatisée des rythmes cardiaques dans les signaux ECG. Nous avons exploré deux approches principales : les méthodes traditionnelles de machine learning et l'utilisation de l'architecture WaveNet.

Nous avons évalué nos modèles sur deux bases de données distinctes : notre propre collection interne d'ECG et une base de données Kaggle basée sur les ensembles de données MIT-BIH et PTB Diagnostic ECG. Les objectifs du projet étaient multiples :

- Développer des modèles de classification précis pour identifier les rythmes cardiaques dans les ECG, en utilisant à la fois des techniques de machine learning classiques et l'architecture WaveNet.
- Évaluer la capacité de ces modèles à généraliser à partir de sources de données variées en les testant sur les deux bases de données.
- Analyser en profondeur les performances obtenues, en évaluant leur pertinence clinique et en examinant les implications potentielles pour l'adoption de ces techniques dans le domaine médical.

Notre projet met en lumière la valeur ajoutée de l'architecture WaveNet dans la classification des rythmes cardiaques tout en soulignant le potentiel des techniques de machine learning pour l'analyse automatisée des ECG. Une classification précise des rythmes cardiaques est essentielle pour des diagnostics plus justes, améliorant ainsi les soins aux patients et la prévention des complications cardiaques.
"""
authors  =""" 
**Auteurs :**
- Nabila ASKEUR, Abdelkader DEBBAGHI, Hocine DRIOUECHE,Miryam KUETE
"""

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    # index = np.argmax(prediction)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

