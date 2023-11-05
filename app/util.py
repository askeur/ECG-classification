import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np

PAgeIntro = """
<h1>Heartbeat Classification</h1>
<p>In the medical field, manual interpretation of electrocardiogram (ECG) recordings can be a labor-intensive and error-prone task. To address this challenge, our project colab focused on the automated classification of cardiac rhythms in ECG signals. We explored two primary approaches: traditional machine learning methods and the use of the WaveNet architecture.</p>
<h2>Objectives</h2>
<ul>
  <li>Develop accurate classification models to identify cardiac rhythms in ECGs, using both classical machine learning techniques and the WaveNet architecture.</li>
  <li>Assess the ability of these models to generalize from diverse data sources by testing them on both databases.</li>
  <li>Conduct a detailed analysis of the achieved performances, evaluating their clinical relevance and examining potential implications for the adoption of these techniques in the medical field.</li>
</ul>
<p>Our project colab highlights the added value of the WaveNet architecture in cardiac rhythm classification while emphasizing the potential of machine learning techniques for automated ECG analysis. Accurate classification of cardiac rhythms is crucial for more precise diagnoses, thereby improving patient care and preventing cardiac complications.</p>
"""
authors  =""" 
**Authors :**
    [Nabila ASKEUR](https://www.linkedin.com/in/nabila-askeur-b120334a/) , KUETE DONGMO Miryam , [Abdelkader DEBBAGHI](https://www.linkedin.com/in/debbaghi-abdelkader-84840810/) , [Hocine DRIOUECHE](https://www.linkedin.com/in/hocine-drioueche-93b65b27/)
"""

PAgeMl = """
<ul>
  <h3>The Modeling Architecture</h3>
  <p>For the Kaggle PTB MIH ECG dataset, we employed a machine learning approach to diagnose cardiac arrhythmias.</p>
</ul>

<ul>
  <h4>Splitting the Dataset</h4>
  <p>We divided the dataset into training (80%) and testing (20%) sets to assess the model's effectiveness.</p>
</ul>

<ul>
  <li><h4>Feature Selection</h4></li>
  <p>Segmentation was performed to remove spikes and noise, reducing the dimensionality of the dataset while preserving vital information.</p>
</ul>



 <h3>Models in General</h3>
<p>We employed several models to classify the ECG data, each with its unique strengths:</p>
<ul>
  <li>
    <h4>SVM (Support Vector Machine)</h4>
    <p>A linear SVM model was used to separate classes within the dataset, with fine-tuned C and gamma hyperparameters for optimal performance.</p>
  </li>
  <li>
    <h4>KNN (K-Nearest Neighbors)</h4>
    <p>The K-nearest neighbors algorithm found the k nearest neighbors of each data point, starting with k=5 and adapting for better performance.</p>
  </li>
  <li>
    <h4>WaveNet</h4>
    <p>We employed the WaveNet architecture, a neural network designed for signal processing, and trained it on the ECG data. We evaluated its performance by comparing results with actual labels.</p>
  </li>
  <li>
    <h4>LogisticRegression</h4>
    <p>Logistic regression was employed as a classification model, providing valuable insights into the dataset's patterns and relationships.</p>
  </li>
</ul>


<h3>Model Evaluation</h3>
<ul>
  <li><h4>Cross-Validation</h4></li>
  <p>To assess model performance, we applied cross-validation by dividing the data into 10 subsets, training and testing each model on every subset, providing insights into generalization capabilities.</p>

  <li><h4>Model Performance</h4></li>
  <p>We assessed model performance using essential metrics, including precision, recall, and the F1 score.</p>

  <li><h4>Accuracy</h4></li>
  <p>We measured accuracy as the number of correct predictions compared to the total number of predictions.</p>
</ul>

  <li><h4>Results</h4></li>
  <p>We presented the results of each approach in a table, emphasizing precision, recall, and the F1 score. Notably, the KNN and WaveNet models achieved the best performance on this dataset.</p>
</ul>

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

