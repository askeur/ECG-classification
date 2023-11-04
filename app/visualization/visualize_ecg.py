from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ecg(uploaded_ecg, FS):
    '''
    Visualize the ECG signal.

    Parameters
    ----------
    uploaded_ecg : numpy.ndarray
        The ECG signal as a numpy array.
    FS : int
        The sampling frequency of the ECG signal.

    Returns
    -------
        The figure object created by matplotlib of the ECG signal.
    '''
    ecg_1d = uploaded_ecg.reshape(-1)
    N = len(ecg_1d)
    time = np.arange(N) / FS

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, ecg_1d)
    ax.set_title('ECG Signal', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('ECG (mV)', fontsize=12)
    ax.grid(which='both', color='#CCCCCC', linestyle='--')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(labelsize=10)

    plt.tight_layout()

    return fig


def plot_ecg_signals(df, title, n_samples=3):
    labels = df.iloc[:, -1].unique()
    fig, axs = plt.subplots(len(labels), n_samples, figsize=(15, 2 * len(labels)))

    for i, label in enumerate(labels):
        sample_df = df[df.iloc[:, -1] == label]
        n_samples_actual = min(n_samples, sample_df.shape[0])
        sample_df = sample_df.sample(n_samples_actual, random_state=42)
        for j, idx in enumerate(sample_df.index):
            signal = sample_df.loc[idx, :].values[:-1]
            axs[i, j].plot(signal)
            axs[i, j].set_title(f"Classe {label}")
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

#"Normal Beats" correspond à la catégorie 0.0 (N)
#"Supraventricular Beats" correspond aux catégories 1.0 (L, R, A, a, J, S)
#"Ventricular Beats" correspond aux catégories 2.0 (V, E)
#"Fusion Beats" correspond à la catégorie 3.0 (F)
#"Unknown Beats" correspond à toutes les autres catégories, donc 4.0 pour toutes les autres valeurs.


# Génère le diagramme en cercle

def generate_pie_chart(ax, class_counts, labels, title):
    values = class_counts.values
    labels = class_counts.index  # Utilisez les index comme étiquettes
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=plt.cm.Paired(range(len(labels))))
    circle = plt.Circle((0, 0), 0.7, color='white')
    ax.add_artist(circle)
    ax.set_title(title, fontsize=16)


import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st





def plot_class_distribution(df, title):
    plt.rcParams['legend.fontsize'] = 10

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Sous-graphique 1 : sns.countplot
    sns.countplot(x=df.iloc[:, -1], ax=ax[0])
    ax[0].set_title(title)
    ax[0].set_xlabel('Classes')
    ax[0].set_ylabel('Nombre de battements')

    # Ajout de la légende
    class_legend = {
        0.0: 'Battements normaux',
        1.0: 'Battements supraventriculaires',
        2.0: 'Battements ventriculaires',
        3.0: 'Battements de fusion',
        4.0: 'Battements inconnus'
    }
    ax[0].legend(title='Classes', labels=class_legend.values())

    # Sous-graphique 2 : Pie chart
    class_counts = df.iloc[:, -1].value_counts()
    num_segments = len(class_counts)
    explode = [0.1] * num_segments  # Liste d'explosions avec la même longueur que le nombre de segments
    class_counts.plot.pie(explode=explode, autopct='%1.2f%%', shadow=True, ax=ax[1])
    ax[1].set_title(title, fontsize=20, color='Red', font='Lucida Calligraphy')
    ax[1].legend(title='Classes', labels=class_legend.values())
    ax[1].axis('off')

    # Ajout de commentaires à côté de chaque barre
    for i, count in enumerate(class_counts):
        ax[0].annotate(f'{count}', (i, count), ha='center', va='bottom')

    st.pyplot(fig)

def plot_ptbdb_class_distribution(df, title):
    class_legend = {
        'Normal': 'Normal Beats',
        'Abnormal': 'Abnormal Beats'
    }
    plt.figure(figsize=(8, 4))
    sns.countplot(x=df.iloc[:, -1])
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Nombre de battements')
    plt.legend(class_legend.values(), title='Classes')
    st.pyplot(plt)

# Example usage:





# Fonction pour afficher les statistiques descriptives

def display_descriptive_statistics(df, name):
    st.write(df.describe())


# Fonction pour convertir les étiquettes en classes numériques
def convert_label(label):
    if label in ['N']:
        return 0.0
    elif label in ['L', 'R', 'A', 'a', 'J', 'S']:
        return 1.0
    elif label in ['V', 'E']:
        return 2.0
    elif label in ['F']:
        return 3.0
    else:
        return 4.0