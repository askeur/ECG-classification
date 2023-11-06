import streamlit as st
import mit as mit  # Import mit_module and ptb_module if they are Python modules in your project colab
import ptb as ptb
from css import hide_streamlit_style
import warnings
from util import set_background, PAgeIntro, authors, PAgeMl

# Set the page configuration
st.set_page_config(
    page_title='Heartbeat Classification',
    page_icon="https://api.iconify.design/openmoji/anatomical-heart.svg?width=500",
    layout='wide'
)
# Add custom CSS to hide Streamlit's default styles
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
set_background('app/img/background.gif')  # HeartGif.gif

modeles = ["WaveNet", "SVM", "KNN", "LogisticRegression"]



def main():

    # Set the default value for show_intro
    show_intro = True
    st.sidebar.image('app/img/logo.png')
    # Sidebar header for selecting the context
    st.sidebar.header("Select the Context")
    selected_context = st.sidebar.radio("", ["Training MIH", "Training PTB"])

    # Choice for ECG visualization or classification
    choice = st.sidebar.selectbox("Select an Action", ["Visualize ECG", "Classification"])

    # File uploaders
    uploaded_files = None
    uploaded_file = None

    if choice == "Visualize ECG":
        st.sidebar.header("▓▒­░⡷⠂ECG Signal⠐⢾░▒▓")
        if selected_context == "Training MIH":
            uploaded_files = st.sidebar.file_uploader("Upload ECG Files", type=["atr", "hea", "dat"],
                                                      accept_multiple_files=True)
        elif selected_context == "Training PTB":
            uploaded_files = st.sidebar.file_uploader("Select ECG Files (XYZ, HEA, DAT)", type=["xyz", "hea", "dat"],
                                                      accept_multiple_files=True)

    if uploaded_files:
        show_intro = False

    modele_selected = None
    # Handle different actions based on the choice
    if selected_context == "Training MIH" and choice:
        if choice == "Visualize ECG":
            mit.run_mit_visualize(uploaded_files)
        elif choice == "Classification":
            uploaded_file = None
            with st.sidebar.header('█▓▒­░⡷⠂CSV File⠐⢾░▒▓█'):
                uploaded_file = st.sidebar.file_uploader("Download asegmented file CSV", type=["csv"])
            if uploaded_file is not None:
                # Afficher l'en-tête dans la barre latérale
                with st.sidebar.header('▓▒­░⡷⠂Choose Model⠐⠐⢾░▒▓█'):
                    modele_selected = st.sidebar.selectbox("Modèle", modeles)
                    # Ajout de l'option oversampling et undersampling
                    sampling_option = st.sidebar.radio("Sampling Option", ["None", "Oversampling", "Undersampling"])

            model_path = None
            model = None

            if modele_selected:
                if st.sidebar.button("Perform the prediction"):
                    show_intro = False
                    st.markdown("<h1 style='text-align: center; font-size: 36px;'>ECG Data Prediction</h1>",
                                unsafe_allow_html=True)
                    st.markdown(
                        f"<h4 style='text-align: center;'>Using {modele_selected} model based on {selected_context} database</h4>",
                        unsafe_allow_html=True)

                    if modele_selected != 'WaveNet':
                        isWaveNet = False
                        if sampling_option != "None":
                            if sampling_option == "Oversampling":
                                model_path = f'app/models/{modele_selected}/{modele_selected.lower() + "_over_mit"}.joblib'
                            elif sampling_option == "Undersampling":

                                model_path = f'app/models/{modele_selected}/{modele_selected.lower() + "_under_mit"}.joblib'
                        else:
                            model_path = f'app/models/{modele_selected}/{modele_selected.lower() + "_mit"}.joblib'
                        model = mit.load_jb_model(model_path)
                    else:
                        isWaveNet = True
                        model_path = f'app/models/{modele_selected}/{modele_selected.lower() + "_mit"}.h5'

                        model = mit.load_wavenet_model(model_path)
                    if model_path and model:
                        mit.run_mit_classification(uploaded_file, model, isWaveNet)
                    else:
                        st.warning("Please select a valid model before running the classification.")

    elif selected_context == "Training PTB" and choice:
        if choice == "Visualize ECG":
            ptb.run_ptb_visualize(uploaded_files)
        elif choice == "Classification":
            with st.sidebar.header('█▓▒­░⡷⠂CSV File⠐⢾░▒▓█'):
                uploaded_file = st.sidebar.file_uploader("Download asegmented file CSV", type=["csv"])
            if uploaded_file is not None:
                # Afficher l'en-tête dans la barre latérale
                with st.sidebar.header('▓▒­░⡷⠂Choose Model⠐⠐⢾░▒▓█'):
                    modele_selected = st.sidebar.selectbox("Modèle", modeles)
                    # Ajout de l'option oversampling et undersampling
                    sampling_option = st.sidebar.radio("Sampling Option", ["None", "Oversampling", "Undersampling"])

            model_path = None
            model = None

            if modele_selected:
                isWaveNet = False
                if st.sidebar.button("Perform the prediction"):
                    show_intro = False
                    st.markdown("<h1 style='text-align: center; font-size: 36px;'>ECG Data Prediction</h1>",
                                unsafe_allow_html=True)
                    st.markdown(
                        f"<h4 style='text-align: center;'>Using {modele_selected} model based on {selected_context} database</h4>",
                        unsafe_allow_html=True)
                    if modele_selected != 'WaveNet':

                        isWaveNet = False
                        if sampling_option != "None":
                            if sampling_option == "Oversampling":

                                model_path = f'app/models/{modele_selected}/{modele_selected.lower() + "_over_ptb"}.joblib'
                            elif sampling_option == "Undersampling":

                                model_path = f'app/models/{modele_selected}/{modele_selected.lower() + "_under_ptb"}.joblib'
                        else:
                            model_path = f'app/models/{modele_selected}/{modele_selected.lower() + "_ptb"}.joblib'

                        model = ptb.load_jb_model(model_path)
                    else:
                        isWaveNet = True
                        model_path = f'app/models/{modele_selected}/{modele_selected.lower() + "_ptb"}.h5'

                        model = ptb.load_wavenet_model(model_path)
                    if model_path and model:
                        ptb.run_ptb_classification(uploaded_file, model, isWaveNet)
                    else:
                        st.warning("Please select a valid model before running the classification.")

    # Content for the introductory page
    if show_intro:
        st.markdown(PAgeIntro, unsafe_allow_html=True)
        st.markdown(authors, unsafe_allow_html=True)
        with st.expander("Click to see more details - The modeling architecture -", expanded=False):
            st.write(PAgeMl, unsafe_allow_html=True)
        st.image('app/img/ML.gif')


if __name__ == "__main__":
    # Ignore all warnings
    warnings.filterwarnings("ignore")

    # Start the Streamlit application
    main()