# 📊 Projet : Analyse Exploratoire et Segmentation des Clients
Application : https://analyse-marketing-zap5appepxyqkmjhtt3ovqi.streamlit.app/

## 🎯 Objectif
Ce projet vise à analyser un dataset marketing afin d'identifier les segments de clients les plus rentables en fonction de leur engagement avec les campagnes marketing et leurs dépenses.
Une **application Streamlit interactive** a été développée pour explorer les données et visualiser les analyses.

## 📂 Contenu du projet
- **app.py** : Code de l'application Streamlit permettant d'explorer les données.
- **marketing_campaign.csv** : Dataset utilisé pour l'analyse.
- **README.md** : Documentation du projet.

## 🛠️ Installation & Exécution
### 1️⃣ Prérequis
Assurez-vous d'avoir **Python 3.7+** installé ainsi que les dépendances suivantes :
```bash
pip install streamlit pandas matplotlib seaborn scipy
```

### 2️⃣ Lancer l'application Streamlit
Exécutez la commande suivante pour démarrer l'interface interactive :
```bash
streamlit run app.py
```

## 📌 Fonctionnalités de l'application
✅ **Aperçu des données** : Affichage des premières lignes et statistiques générales.  
✅ **Analyse univariée** : Distribution des variables numériques.  
✅ **Analyse bivariée** : Corrélation entre les variables numériques.  
✅ **Détection des valeurs aberrantes** : Identification des outliers via le Z-score.  
✅ **Segmentation des clients** : Étude de l'engagement marketing et des profils de dépenses.  

## 📊 Dataset utilisé
Le dataset **marketing_campaign.csv** contient des informations sur 2 240 clients et 29 variables, dont :
- `Income` : Revenu du client.  
- `MntWines`, `MntFruits`, `MntMeatProducts`... : Dépenses des clients.  
- `NumWebPurchases`, `NumStorePurchases` : Nombre d'achats en ligne et en magasin.  
- `AcceptedCmp1` à `AcceptedCmp5`, `Response` : Indique si le client a accepté une campagne marketing.  

## 🔍 Résultats clés de l'analyse
- **Les clients engagés dans plusieurs campagnes dépensent davantage.**  
- **Les revenus influencent l'engagement marketing** : les clients à revenu moyen/élevé acceptent plus de campagnes.  
- **Certains clients à revenus élevés acceptent peu de campagnes** : possibilité d'optimisation marketing.  

## 🚀 Améliorations possibles
🔹 Analyser les facteurs influençant l'engagement des clients fortunés.  
🔹 Optimiser les campagnes pour cibler les segments les plus rentables.  
🔹 Développer un modèle prédictif pour anticiper les réponses aux campagnes marketing.  

---
✨ **Auteur** : [Ton Nom]  
📅 **Date** : [Aujourd'hui]  
