# 🫀 Modèle de Prédiction du Risque Cardiovasculaire

> Composant IA de la plateforme **CardioDiab Predict** — CHU Hassan II de Fès  
> Modèle : **Régression Logistique** | Pipeline : scikit-learn | Explicabilité : SHAP

---

## 📋 Vue d'ensemble

Ce modèle prédit la **probabilité de présence d'une maladie cardiovasculaire** chez un patient adulte à partir de ses constantes cliniques mesurées en consultation. Il renvoie un score entre 0 et 100 %, accompagné d'une explication variable par variable via la méthode SHAP.

| Attribut | Valeur |
|---|---|
| Fichier final | `pipeline_cardio_reglog.pkl` |
| Algorithme | Régression Logistique (`sklearn.linear_model`) |
| Variable cible | `cardio` (0 = sain, 1 = maladie cardiovasculaire) |
| Dataset source | `cardio_train.csv` |
| Observations initiales | 70 000 patients |
| Observations après nettoyage | ~67 000 |
| Équilibre des classes | ~50 % / 50 % |

---

## 📁 Structure des fichiers

```
cardio/
  EDA.ipynb                    # Exploration, nettoyage, feature engineering
  Prepro_train_tuning.ipynb    # Pipeline, comparaison modèles, sélection finale
  pipeline_cardio_reglog.pkl   # Pipeline complet sauvegardé (préprocesseur + modèle)
  cardio_train.csv             # Dataset source (non inclus — voir section données)
  README.md                    # Ce fichier
```

---

## 🔬 Dataset — `cardio_train.csv`

### Variables brutes

| Variable brute | Renommée | Type | Signification clinique |
|---|---|---|---|
| `age` | `age` | Continue | Âge en **jours** → converti en années |
| `gender` | `gender` | Binaire | 1 = Femme, 2 = Homme |
| `height` + `weight` | `bmi` | Continue | IMC calculé : poids / taille² (m) |
| `ap_hi` | `systolic_pressure` | Continue | Pression artérielle systolique |
| `ap_lo` | `diastolic_pressure` | Continue | Pression artérielle diastolique |
| `cholesterol` | `cholesterol` | Ordinale | 1=Normal, 2=Élevé, 3=Très élevé |
| `gluc` | `glucose` | Ordinale | 1=Normal, 2=Élevé, 3=Très élevé |
| `smoke` | `smoking_status` | Binaire | 0=Non, 1=Oui |
| `alco` | `alcohol_status` | Binaire | 0=Non, 1=Oui |
| `active` | `physical_activity` | Binaire | 1=Actif, 0=Sédentaire |
| `cardio` | `cardio` (cible) | Binaire | 0=Sain, 1=Maladie cardiovasculaire |

---

## 🔍 Phase 1 — Analyse Exploratoire (EDA.ipynb)

### 1.1 Statistiques générales

- Aucune valeur manquante dans le dataset brut
- Classes équilibrées (~50%/50%) → métriques classiques utilisables sans correction
- Doublons détectés et supprimés avant toute analyse

### 1.2 Feature Engineering

Trois transformations clés, toutes justifiées cliniquement :

```python
# 1. Conversion âge en années (stocké en jours dans le dataset brut)
df['age'] = df['age'] // 365.25

# 2. Création de l'IMC (remplace les variables brutes poids et taille)
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
df.drop(['weight', 'height'], axis=1, inplace=True)

# 3. Création de la pression différentielle (variable engineered)
# Mesure la "dureté" des artères — information absente de chaque pression seule
df['diff_PS_PD'] = df['systolic_pressure'] - df['diastolic_pressure']
```

### 1.3 Traitement des valeurs aberrantes

Les pressions artérielles contenaient des erreurs de saisie manifestes (valeurs négatives, inversions physiopathologiques, valeurs incompatibles avec la vie) :

```python
# Correction des erreurs de signe
df['systolic_pressure']  = df['systolic_pressure'].abs()
df['diastolic_pressure'] = df['diastolic_pressure'].abs()

# Plages physiologiquement réalistes
df = df[(df['systolic_pressure']  >= 70) & (df['systolic_pressure']  <= 250)]
df = df[(df['diastolic_pressure'] >= 40) & (df['diastolic_pressure'] <= 140)]

# Règle physique fondamentale : systolique > diastolique
df = df[df['systolic_pressure'] > df['diastolic_pressure']]

# Pression différentielle minimale (< 15 mmHg = cas physiologiquement douteux)
df = df[df['diff_PS_PD'] >= 15]

# IMC extrême
df = df[df['bmi'] < 70]
```

> Résultat : ~4 % des observations supprimées — suppressions ciblées, pas excessives.

### 1.4 Variable supprimée avant modélisation

`systolic_pressure` a été retirée du jeu de features car sa corrélation avec `diff_PS_PD` est de **0.84**, introduisant de la multicolinéarité qui déstabilise les coefficients de la régression logistique.

---

## ⚙️ Phase 2 — Prétraitement & Entraînement (Prepro_train_tuning.ipynb)

### 2.1 Séparation train / test

```python
X = df.drop('cardio', axis=1)
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

`stratify=y` garantit la même proportion de cas positifs dans le train et le test.

### 2.2 Pipeline de prétraitement

Trois sous-pipelines selon la nature des variables :

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Variables numériques continues
numeriques = ['age', 'diff_PS_PD', 'diastolic_pressure', 'bmi']
pipe_num = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# Variables catégorielles ordinales
ordinales = ['cholesterol', 'glucose']
pipe_cat = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False)),
])

# Variables binaires (déjà en 0/1)
binaires = ['gender', 'smoking_status', 'alcohol_status', 'physical_activity']
pipe_bin = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
])

preprocessor = ColumnTransformer([
    ('num', pipe_num, numeriques),
    ('cat', pipe_cat, ordinales),
    ('bin', pipe_bin, binaires),
])
```

### 2.3 Comparaison des algorithmes

11 algorithmes testés dans les mêmes conditions (même pipeline, même test set), évalués à un seuil de décision fixé à **0.3** (plutôt que le seuil par défaut de 0.5, afin de privilégier le rappel — critique en contexte de dépistage clinique, où un faux négatif est plus coûteux qu'un faux positif) :

| Algorithme | Recall @seuil 0.3 | F1 @seuil 0.3 | Observations |
|---|---|---|---|
| **Régression Logistique** | 0.96 | 0.70 | ✅ **Retenu** — stable, interprétable, coefficients lisibles pour le SHAP linéaire |
| Random Forest | 0.82 | 0.71 | Bon F1, mais lent à l'inférence |
| Decision Tree | 0.63 | 0.63 | Surapprentissage marqué (overfitting) |
| Gradient Boosting | 0.88 | 0.73 | Performances proches de LightGBM |
| KNeighbors | 0.84 | 0.71 | Lent, pas adapté au déploiement |
| SVM (SVC) | 0.80 | 0.62 | Très lent sur 50 000 observations |
| XGBoost | 0.87 | 0.73 | Bon compromis vitesse/performance |
| LightGBM | 0.87 | 0.73 | Rapide, performant |
| HistGradientBoosting | 0.87 | 0.73 | Bon sur grandes données |
| AdaBoost (+ tuning GridSearch) | 0.93 | 0.71 | Performant après tuning, Recall améliorable |
| Stacking (LightGBM + GBM + AdaBoost + XGBoost, méta-modèle = Régression Logistique) | 0.87 | 0.73 | Meilleur Recall parmi les ensembles, mais bien plus complexe à maintenir et à expliquer |

**Lecture du tableau :** malgré un F1-score légèrement inférieur à celui des modèles d'ensemble (0.70 contre 0.71–0.73), la régression logistique obtient le **meilleur rappel du comparatif (0.96)** — c'est-à-dire qu'elle manque le moins de vrais cas de maladie cardiovasculaire, un critère jugé prioritaire dans un contexte de dépistage médical où un faux négatif peut retarder une prise en charge. Combiné à son interprétabilité native et à sa compatibilité avec un calcul SHAP exact (non approximé), ce compromis a motivé le choix final.

### 2.4 Modèle final et sauvegarde

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

pipeline_cardio = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=10000, class_weight='balanced')
)

pipeline_cardio.fit(X_train, y_train)

# Sauvegarde du pipeline complet
joblib.dump(pipeline_cardio, 'models/pipeline_cardio_reglog.pkl')
```

---

## 🧠 Justification du choix — Régression Logistique

1. **Meilleur rappel du comparatif (0.96 @ seuil 0.3)** — critère prioritaire pour un outil de dépistage, où manquer un cas à risque est plus coûteux qu'une fausse alerte
2. **Compatibilité native SHAP linéaire** — `shap.Explainer` avec `masker.Independent` : contributions exactes, pas d'approximation
3. **Vitesse d'inférence** — une simple multiplication matricielle : réponse < 10 ms en production
4. **Principe de parcimonie** — en contexte médical, un modèle interprétable et simple est préférable à un modèle opaque légèrement plus précis sur le F1, surtout lorsque l'écart reste marginal (0.70 contre 0.71–0.73)

---

## 📊 Explicabilité SHAP — Intégration backend

```python
import shap
import numpy as np

# Extraction des étapes du pipeline
preprocessor_c = pipeline_cardio.named_steps['columntransformer']
model_lr        = pipeline_cardio.named_steps['logisticregression']

# Background dataset de référence (ligne de zéros)
num_features  = model_lr.n_features_in_
background    = np.zeros((1, num_features))

# Explainer adapté aux modèles linéaires
explainer_cardio = shap.Explainer(
    lambda X: model_lr.predict_proba(X)[:, 1],
    masker=shap.maskers.Independent(background)
)

# Calcul des valeurs SHAP pour une consultation
X_transformed = preprocessor_c.transform(df_patient)
shap_values   = explainer_cardio.shap_values(X_transformed)

# Regroupement des colonnes one-hot → variables cliniques d'origine
# (voir regrouper_valeurs_shap() dans main.py)
```

---

## 🚀 Utilisation depuis le backend FastAPI

```python
import joblib
import pandas as pd

pipeline = joblib.load('models/pipeline_cardio_reglog.pkl')

df_patient = pd.DataFrame([{
    'age':              45,
    'diff_PS_PD':       60,    # ap_hi - ap_lo
    'diastolic_pressure': 80,
    'bmi':              27.4,
    'cholesterol':      2,
    'glucose':          1,
    'gender':           2,
    'smoking_status':   0,
    'alcohol_status':   0,
    'physical_activity': 1,
}])

# Score entre 0 et 100 %
score_cardio = round(pipeline.predict_proba(df_patient)[0][1] * 100, 2)
```

---

## 📦 Dépendances

```
scikit-learn>=1.5.0
pandas>=2.2.0
numpy>=1.26.0
joblib>=1.4.0
shap>=0.45.0
matplotlib>=3.8.0   # pour les graphiques EDA uniquement
seaborn>=0.13.0     # pour les graphiques EDA uniquement
```

---

## ⚠️ Limites et précautions

- Ce modèle est un **outil d'aide à la décision** — il ne remplace pas le jugement clinique du médecin
- Les données d'entraînement sont issues d'un dataset public international — une recalibration sur des données locales du CHU Hassan II améliorerait la précision
- Le modèle calcule l'âge par différence d'années entières — une date de naissance précise donnerait un résultat plus fin
- Le seuil de décision de 0.3 (plutôt que 0.5) privilégie volontairement le rappel au détriment de la précision — à réévaluer si l'usage clinique cible évolue vers un contexte moins orienté dépistage
