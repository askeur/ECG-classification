# ECG-classification

(https://github.com/askeur/ECG-classification)

This Streamlit application is designed to classify heart rhythms in ECG (Electrocardiogram) signals. It provides the following features:

- Uploading and preprocessing ECG files in the .atr, .hea, and .dat formats.
- Classification of ECG signals using different machine learning models.
- Visualizations of ECG signal segments, class distributions, and classification results.

## Getting Started

### Prerequisites

Before running the application, make sure you have the required Python packages installed. You can install them using `pip`:
- pip install streamlit
- pip install pandas
- pip install seaborn
- pip install wfdb
- pip install numpy
- pip install scikit-learn
- pip install tensorflow
- pip install matplotlib



### Running the Application

To run the ECG Classification Application, open your terminal and navigate to the application's directory. Then, run the following command:

streamlit run app/mains.py  


This will start the Streamlit development server, and you can access the application in your web browser.

## Usage

1. Upload ECG files in the .atr, .hea, and .dat formats -- > csv_train, csv_test
2. Choose a contect ( MIT, PTB) and a classification model (WaveNet, SVM, KNN, LogisticRegression).
3. Perform preprocessing on the uploaded files.
4. Visualize ECG signals, class distributions, and descriptive statistics.
5. Perform ECG signal classification.
6. View classification reports, confusion matrices, and error analysis.

## Model Information
The models used in this application are provided in zip format due to their large size. You need to unzip them before using them.

## Built With

- Python
- Streamlit
- scikit-learn
- TensorFlow
- Pandas
- WFDB (WaveForm Database)

## Authors

[Nabila ASKEUR](https://www.linkedin.com/in/nabila-askeur-b120334a/) , Myriam KUTES DONGMO , [Abdelkader DEBBAGHI](https://www.linkedin.com/in/debbaghi-abdelkader-84840810/) , [Hocine DRIOUECHE](https://www.linkedin.com/in/hocine-drioueche-93b65b27/)


## Acknowledgments

- Icons and gif (Adob photoshop)  

## Additional Information

For more information, please refer to the [GitHub repository](https://github.com/askeur/ECG-classification).



