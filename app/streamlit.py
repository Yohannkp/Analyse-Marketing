

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

import os

# Trouver le chemin absolu du fichier en fonction du répertoire actuel
chemin_data = os.path.join(os.path.dirname(__file__), '../data/marketing_campaign.csv')

# Charger les données
@st.cache_data
def load_data():
    
    df = pd.read_csv(chemin_data, sep=';')
    
    # Création des variables utiles
    df["Total_Campaigns_Accepted"] = (
        df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"] +
        df["AcceptedCmp4"] + df["AcceptedCmp5"] + df["Response"]
    )
    df["Total_Spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["Segment_Spending"] = pd.qcut(df["Total_Spending"], q=3, labels=["Bas", "Moyen", "Élevé"])
    df["Segment_Income"] = pd.qcut(df["Income"].fillna(df["Income"].median()), q=3, labels=["Bas", "Moyen", "Élevé"])
    df["Engagement_Campagnes"] = pd.cut(df["Total_Campaigns_Accepted"], bins=[0, 1, 3, 6], labels=["Faible", "Moyen", "Élevé"])
    
    return df

df = load_data()

# Titre de l'application
st.title("📊 Analyse Exploratoire et Segmentation des Clients")

# Sélection d'un type d'analyse
option = st.sidebar.selectbox("Sélectionner une analyse", [
    "Aperçu des données", 
    "Analyse univariée", 
    "Analyse bivariée", 
    "Détection des valeurs aberrantes", 
    "Segmentation des clients"
])

if option == "Aperçu des données":
    st.subheader("🔍 Aperçu des données")
    st.write(df.head())
    st.write(df.describe())
    
elif option == "Analyse univariée":
    st.subheader("📊 Distribution des variables")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_column = st.selectbox("Choisissez une variable numérique", numeric_columns)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_column], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution de {selected_column}")
    st.pyplot(fig)

elif option == "Analyse bivariée":
    st.subheader("📌 Corrélations entre variables")
    
    # Filtrer uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Matrice de corrélation des variables numériques")
    st.pyplot(fig)

elif option == "Détection des valeurs aberrantes":
    st.subheader("🔎 Détection des valeurs aberrantes (Z-score)")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_column = st.selectbox("Choisissez une variable pour détecter les outliers", numeric_columns)
    df["Zscore"] = zscore(df[selected_column])
    outliers = df[df["Zscore"].abs() > 3]
    st.write(f"Nombre de valeurs aberrantes détectées dans {selected_column} : {len(outliers)}")
    st.write(outliers[[selected_column, "Zscore"]].head())

elif option == "Segmentation des clients":
    st.subheader("📊 Segmentation des clients et engagement marketing")
    segmentation_option = st.radio("Choisissez une analyse de segmentation", [
        "Distribution des dépenses", 
        "Engagement et Dépenses", 
        "Répartition des revenus", 
        "Corrélation Engagement / Revenu"])
    
    if segmentation_option == "Distribution des dépenses":
        fig, ax = plt.subplots()
        sns.histplot(df["Total_Spending"], bins=30, kde=True, color="blue", ax=ax)
        ax.set_title("Distribution des dépenses des clients")
        st.pyplot(fig)
    
    elif segmentation_option == "Engagement et Dépenses":
        fig, ax = plt.subplots()
        sns.boxplot(x=df["Engagement_Campagnes"], y=df["Total_Spending"], palette="coolwarm", ax=ax)
        ax.set_title("Dépenses moyennes par engagement avec les campagnes")
        st.pyplot(fig)
    
    elif segmentation_option == "Répartition des revenus":
        fig, ax = plt.subplots()
        sns.boxplot(x=df["Engagement_Campagnes"], y=df["Income"], palette="coolwarm", ax=ax)
        ax.set_title("Répartition des revenus par engagement avec les campagnes")
        st.pyplot(fig)
    
    elif segmentation_option == "Corrélation Engagement / Revenu":
        fig, ax = plt.subplots()
        sns.countplot(x=df["Engagement_Campagnes"], hue=df["Segment_Income"], palette="coolwarm", ax=ax)
        ax.set_title("Répartition des segments de revenu par engagement avec les campagnes")
        st.pyplot(fig)

st.sidebar.write("📌 Sélectionnez une analyse pour explorer les données et segments clients.")