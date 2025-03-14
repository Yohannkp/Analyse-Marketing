import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import zscore
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

# Trouver le chemin absolu du fichier en fonction du répertoire actuel
chemin_data = os.path.join(os.path.dirname(__file__), '../data/marketing_campaign.csv')

print(chemin_data)

# Chargement des données
@st.cache_data
def load_data():
    file_path = chemin_data
    df = pd.read_csv(file_path, sep=';')
    
    # Nettoyage des données
    df.drop_duplicates(inplace=True)
    df["Income"].fillna(df["Income"].median(), inplace=True)
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    
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
    df["Segment_Income"] = pd.qcut(df["Income"], q=3, labels=["Bas", "Moyen", "Élevé"])
    df["Engagement_Campagnes"] = pd.cut(df["Total_Campaigns_Accepted"], bins=[0, 1, 3, 6], labels=["Faible", "Moyen", "Élevé"])
    
    return df

df = load_data()

# Standardisation des données
features = ["Income", "Total_Spending", "Total_Campaigns_Accepted"]
# Création des nouvelles variables
df["Total_Spending"] = df[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum(axis=1)
df["Total_Purchases"] = df[["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases"]].sum(axis=1)

# Score d'engagement = (nombre d'achats) / (récence d’achat)
df["Engagement_Score"] = df["Total_Purchases"] / (df["Recency"] + 1)  # Ajout de +1 pour éviter la division par 0

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])


# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(df_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=15)
df["DBSCAN_Cluster"] = dbscan.fit_predict(df_scaled)

agglo = AgglomerativeClustering(n_clusters=3)
df["Agglo_Cluster"] = agglo.fit_predict(df_scaled)

# PCA pour visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)
df["PCA_1"], df["PCA_2"] = X_pca[:, 0], X_pca[:, 1]




# Interface Streamlit
st.title("📊 Analyse Exploratoire et Segmentation des Clients")

# Sélection d'un type d'analyse
option = st.sidebar.selectbox("Sélectionner une analyse", [
    "Aperçu des données", 
    "Analyse Exploratoire", 
    #"Détection des valeurs aberrantes", 
    "Segmentation des clients", 
    "Visualisation des Clusters", 
    "Évaluation des Modèles", 
    "Conclusion"
])

if option == "Aperçu des données":
    st.subheader("📌 Aperçu des Données")
    st.write(df.head())
    st.write(df.describe())

elif option == "Analyse Exploratoire":
    st.subheader("📌 Analyse Exploratoire")
    fig, ax = plt.subplots()
    sns.histplot(df["Income"], bins=30, kde=True, color="blue", ax=ax)
    ax.set_title("Distribution des Revenus")
    st.pyplot(fig)
    
    
    # Boxplot pour détecter les valeurs aberrantes de 'Income'
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df["Income"])
    plt.title("Boxplot des revenus des clients")
    plt.show()
    
    fig, ax = plt.subplots()
    sns.boxplot(df["Income"], ax=ax)
    ax.set_title("Distribution des Revenus")
    st.pyplot(fig)
    
    # Répartition des niveaux d'éducation
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(y=df["Education"], palette="coolwarm", ax=ax)
    ax.set_title("Répartition des niveaux d'éducation")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(y=df["Marital_Status"], palette="coolwarm",ax=ax)
    ax.set_title("Répartition des statuts matrimoniaux")
    st.pyplot(fig)
    
    df["Total_Spending"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=df["Income"], y=df["Total_Spending"], alpha=0.5)
    ax.set_title("Revenu vs Dépenses totales")
    ax.set_xlabel("Revenu")
    ax.set_ylabel("Dépenses totales")
    st.pyplot(fig)
    
    # Boxplot des dépenses en fonction du statut matrimonial
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=df["Marital_Status"], y=df["Total_Spending"], palette="coolwarm")
    ax.set_title("Dépenses totales en fonction du statut matrimonial")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Nombre de campagnes marketing acceptées par catégorie d'éducation
    df["Total_Campaigns_Accepted"] = df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"] + df["AcceptedCmp4"] + df["AcceptedCmp5"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df["Education"], y=df["Total_Campaigns_Accepted"], palette="coolwarm")
    ax.set_title("Nombre de campagnes acceptées en fonction du niveau d'éducation")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# elif option == "Détection des valeurs aberrantes":
#     st.subheader("📌 Détection des Valeurs Aberrantes")
#     df["Income_Zscore"] = zscore(df["Income"])
#     df["Spending_Zscore"] = zscore(df["Total_Spending"])
#     st.write(df[["Income", "Income_Zscore", "Total_Spending", "Spending_Zscore"]].head())

elif option == "Segmentation des clients":
    st.subheader("📌 Segmentation des Clients")
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Engagement_Campagnes"], y=df["Total_Spending"], palette="coolwarm", ax=ax)
    st.pyplot(fig)
    
    
    
    # Répartition des clients rentables (dépenses élevées) par engagement avec les campagnes
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=df["Engagement_Campagnes"], hue=df["Segment_Spending"], palette="coolwarm")
    ax.set_title("Répartition des segments de dépenses par engagement avec les campagnes")
    ax.set_xlabel("Engagement avec les campagnes")
    ax.set_ylabel("Nombre de clients")
    st.pyplot(fig)

elif option == "Visualisation des Clusters":
    st.subheader("📌 Visualisation des Clusters")
    
    def plot_clusters(df, cluster_col, title):
        fig, ax = plt.subplots()
        sns.scatterplot(x=df["PCA_1"], y=df["PCA_2"], hue=df[cluster_col], palette="coolwarm", alpha=0.7, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)
    
    plot_clusters(df, "KMeans_Cluster", "Clusters avec K-Means")
    plot_clusters(df, "DBSCAN_Cluster", "Clusters avec DBSCAN")
    plot_clusters(df, "Agglo_Cluster", "Clusters avec Clustering Hiérarchique")

    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df["PCA_1"],y=df["PCA_2"], hue=df["Agglo_Cluster"], palette="coolwarm",alpha=0.7)
    ax.set_title("Visualisation des Clusters avec ACP (PCA)")
    ax.set_xlabel("Composante Principale 1")
    ax.set_ylabel("Composante Principale 2")
    ax.legend(title="Cluster")
    st.pyplot(fig)
    
    
    features = ["Income", "Total_Spending", "Total_Purchases", "Engagement_Score", "Recency"]
    X = df[features]
    # Appliquer t-SNE sur les données
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Ajouter les coordonnées t-SNE dans le DataFrame
    df["tSNE_1"] = X_tsne[:, 0]
    df["tSNE_2"] = X_tsne[:, 1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df["tSNE_1"],y=df["tSNE_2"], hue=df["Agglo_Cluster"], palette="coolwarm",alpha=0.7)
    ax.set_title("Visualisation des Clusters avec t-SNE")
    ax.set_xlabel("t-SNE Composante 1")
    ax.set_ylabel("t-SNE Composante 2")
    ax.legend(title="Cluster")
    st.pyplot(fig)
    
elif option == "Évaluation des Modèles":
    st.subheader("📌 Évaluation des Modèles de Clustering")
    scores = {
        "K-Means": silhouette_score(df_scaled, df["KMeans_Cluster"]),
        "DBSCAN": silhouette_score(df_scaled, df["DBSCAN_Cluster"]) if len(set(df["DBSCAN_Cluster"])) > 1 else -1,
        "Agglomerative": silhouette_score(df_scaled, df["Agglo_Cluster"])
    }
    st.write(pd.DataFrame(scores, index=["Silhouette Score"]).T)

elif option == "Conclusion":
    st.subheader("📝 Conclusion")
    st.write("""
    - Le clustering K-Means a permis d'identifier 3 groupes de clients avec des comportements différents.
    - DBSCAN a détecté certains points aberrants, ce qui montre qu'il est plus sensible aux outliers.
    - Le clustering hiérarchique donne des résultats similaires mais plus denses.
    - Les clients ayant des revenus élevés ont tendance à dépenser plus en vin et viande.
    - Le statut matrimonial et le niveau d'éducation influencent également les dépenses des clients.
    - Les résultats de t-SNE confirment que nos clusters sont bien séparés et significatifs.
    - L’ACP nous aide à voir que la séparation pourrait être améliorée avec d'autres transformations de variables.
    - K-Means reste le modèle de clustering optimal pour notre segmentation des clients.
    - Cluster 0 : Clients avec revenus et dépenses faibles → Ils pourraient être ciblés avec des offres attractives pour les inciter à acheter plus.
    - Cluster 1 : Clients avec revenus intermédiaires et dépenses variables → Un segment instable qui pourrait être converti en clients fidèles avec des offres personnalisées.
    - Cluster 2 : Clients avec revenus élevés et fortes dépenses → Fidéliser ces clients avec des programmes VIP et des offres premium.
    """)

st.sidebar.write("📌 Sélectionnez une analyse pour explorer les données et segments clients.")
