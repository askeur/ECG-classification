# ECG-classification

![Heartbeat Gif](https://github.com/askeur/ECG-classification)

This Streamlit application is designed to classify heart rhythms in ECG (Electrocardiogram) signals. It provides the following features:

- Uploading and preprocessing ECG files in the .atr, .hea, and .dat formats.
- Classification of ECG signals using different machine learning models.
- Visualizations of ECG signal segments, class distributions, and classification results.

## Getting Started

### Prerequisites

Before running the application, make sure you have the required Python packages installed. You can install them using `pip`:
pip install streamlit
pip install pandas
pip install seaborn
pip install wfdb
pip install numpy
pip install scikit-learn
pip install tensorflow
pip install matplotlib



### Running the Application

To run the ECG Classification Application, open your terminal and navigate to the application's directory. Then, run the following command:

streamlit run app/mains.py  


This will start the Streamlit development server, and you can access the application in your web browser.

## Usage

1. Upload ECG files in the .atr, .hea, and .dat formats -- > csv_train, csv_test
2. Choose a database (Kaggle, MIT, PTB) and a classification model (WaveNet, RandomForest, SVM, KNN, LogisticRegression).
3. Perform preprocessing on the uploaded files.
4. Visualize ECG signals, class distributions, and descriptive statistics.
5. Perform ECG signal classification.
6. View classification reports, confusion matrices, and error analysis.

## Model Information
The models used in this application are provided in zip format due to their large size. You need to unzip them before using them.

Instructions for Extracting the Models
To extract the models, you can follow these steps:

Download the zip file containing the models from [insert the download link here].

Use Python and the zipfile library to extract the models to the destination directory of your choice. You can use the following code:

import zipfile
import os

# Set the path to the zip file
zip_file_path = 'path_to_the_file.zip'

# Set the destination directory for extracting the files
extracted_folder_path = 'destination_directory_path'

# Create the destination directory if it doesn't exist
os.makedirs(extracted_folder_path, exist_ok=True)

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)
Make sure to follow the above steps to have the models ready for use in the application.

## Built With

- Python
- Streamlit
- scikit-learn
- TensorFlow
- Pandas
- WFDB (WaveForm Database)

## Authors

 Nabila ASKEUR, Abdelkader DEBBAGHI, Hocine DRIOUECHE,Miryam KUETE

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Icons and gif (Adob photoshop)  

## Additional Information

For more information, please refer to the [GitHub repository](https://github.com/askeur/ECG-classification).



