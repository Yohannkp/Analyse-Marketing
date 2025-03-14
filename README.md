# 📊 Projet : Analyse Exploratoire et Segmentation des Clients
Application : https://analyse-marketing-zap5appepxyqkmjhtt3ovqi.streamlit.app/

## 🎯 Objectif
Ce projet vise à analyser un dataset marketing afin d'identifier les segments de clients les plus rentables en fonction de leur engagement avec les campagnes marketing et leurs dépenses.

## 🛠️ Technologies Utilisées
- **Python** (pandas, numpy, seaborn, matplotlib, scikit-learn)
- **Streamlit** (pour l'application interactive)
- **Machine Learning** (K-Means, DBSCAN, Clustering Hiérarchique, PCA, t-SNE)

---

## 🔍 Étapes du Projet

### **1️⃣ Chargement et Préparation des Données**
- Importation du dataset `marketing_campaign.csv`
- Nettoyage des valeurs manquantes et transformation des variables
- Création de nouvelles variables :
  - **Total_Spending** : Somme des achats par catégorie
  - **Total_Campaigns_Accepted** : Nombre total de campagnes marketing acceptées
  - **Engagement_Campagnes** : Segmentation selon le nombre de campagnes acceptées
  - **Segment_Spending** et **Segment_Income** : Catégorisation des clients en fonction de leurs dépenses et revenus

### **2️⃣ Analyse Exploratoire des Données (EDA)**
#### **Analyse Univariée**
- **Distribution des revenus des clients** (Histogramme et KDE)
- **Boxplot des revenus** (Détection des valeurs aberrantes)
- **Répartition des niveaux d’éducation** (Countplot)
- **Répartition des statuts matrimoniaux** (Countplot)

#### **Analyse Bivariée**
- **Corrélation entre le revenu et les dépenses** (Scatterplot)
- **Dépenses moyennes en fonction du statut matrimonial** (Boxplot)
- **Nombre de campagnes acceptées en fonction du niveau d’éducation** (Boxplot)
- **Heatmap des corrélations entre les variables numériques**

### **3️⃣ Détection des Valeurs Aberrantes**
- Utilisation du **Z-score** pour détecter les valeurs extrêmes sur `Income` et `Total_Spending`
- Suppression des points aberrants pour améliorer la qualité des clusters

### **4️⃣ Segmentation des Clients**
- **Segmentation des clients selon leurs revenus et dépenses**
- **Répartition des dépenses par engagement avec les campagnes** (Boxplot)
- **Répartition des revenus par engagement avec les campagnes** (Boxplot)
- **Corrélation entre engagement marketing et revenu** (Countplot)

### **5️⃣ Clustering des Clients**
- **Détermination du nombre optimal de clusters** :
  - **Méthode du Coude (Elbow Method)**
  - **Score de Silhouette**
- **Application des Algorithmes de Clustering** :
  - **K-Means** (Méthode centralisée)
  - **DBSCAN** (Détection d’anomalies)
  - **Agglomerative Clustering** (Clustering hiérarchique)
- **Comparaison des modèles** à l’aide des indices de Silhouette et de Davies-Bouldin

### **6️⃣ Visualisation des Clusters**
- **Réduction de dimension avec ACP (PCA)** pour visualiser les clusters en 2D
- **t-SNE** pour mieux comprendre la structure interne des groupes
- **Affichage des clusters avec Seaborn (Scatterplot colorisé par cluster)**

### **7️⃣ Évaluation et Conclusion**
- **K-Means a permis d'identifier 3 groupes de clients avec des comportements différents.**
- **DBSCAN a détecté certains points aberrants, ce qui montre qu'il est plus sensible aux outliers.**
- **Le clustering hiérarchique donne des résultats similaires mais plus denses.**
- **Les clients ayant des revenus élevés ont tendance à dépenser plus en vin et viande.**
- **Le statut matrimonial et le niveau d’éducation influencent également les dépenses des clients.**
- **Les résultats de t-SNE confirment que nos clusters sont bien séparés et significatifs.**
- **L’ACP nous aide à voir que la séparation pourrait être améliorée avec d'autres transformations de variables.**
- **K-Means reste le modèle de clustering optimal pour notre segmentation des clients.**

### **Synthèse des Segments**
- **Cluster 0** : Clients avec revenus et dépenses faibles → Ils pourraient être ciblés avec des offres attractives pour les inciter à acheter plus.
- **Cluster 1** : Clients avec revenus intermédiaires et dépenses variables → Un segment instable qui pourrait être converti en clients fidèles avec des offres personnalisées.
- **Cluster 2** : Clients avec revenus élevés et fortes dépenses → Fidéliser ces clients avec des programmes VIP et des offres premium.

---

## 🏆 Résultats et Recommandations
- Les segments les plus rentables sont ceux ayant un engagement marketing élevé et des revenus moyens ou élevés.
- Les clients peu engagés sont moins rentables et dépensent moins.
- Recommandation : **Cibler les clients à revenu moyen/élevé avec des campagnes adaptées et analyser les freins des clients à fort revenu qui acceptent peu de campagnes.**

---

## 📌 Lancer l'Application Streamlit
```bash
streamlit run app.py
