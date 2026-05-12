# README — Projet Data Science 2025-2026
## Accidents corporels de la circulation routière en France (2024)

---

## Structure des fichiers

```
donnees/
├── accidents_2024.csv          # Dataset brut original
└── clean/
    ├── X_lab_encoded.csv       # Features encodées en Label Encoding
    ├── X_oh_encoding.csv       # Features encodées en One-Hot Encoding
    └── y_gravite.csv           # Variable cible encodée
```

---

## Description des datasets

### Variable cible — `y_gravite.csv`
| Valeur | Signification |
|--------|--------------|
| 0 | Blessé léger |
| 1 | Blessé hospitalisé |
| 2 | Tué |

**Distribution :**
- Blessé léger : 34 744 (63.9%)
- Blessé hospitalisé : 16 432 (30.2%)
- Tué : 3 226 (5.9%)

⚠️ **Classes déséquilibrées** — ratio max/min ≈ 11x. Tenir compte dans tous les modèles.

### Features — colonnes communes aux deux datasets
| Colonne | Type | Description |
|--------|------|-------------|
| Agglomération | catégorielle | En/Hors agglomération |
| Luminosité | catégorielle | Conditions de luminosité |
| Météo (conditions atmos.) | catégorielle | Conditions météo |
| Type de collision | catégorielle | Nature de la collision |
| Catégorie de route | catégorielle | Type de route |
| Régime de circulation | catégorielle | Sens de circulation |
| État de la surface | catégorielle | État de la chaussée |
| Infrastructure | catégorielle | Infrastructure particulière |
| Situation de l'accident | catégorielle | Position de l'accident |
| Région | catégorielle | Région française (regroupement des départements) |
| Vitesse_num | numérique (int) | Vitesse max autorisée en km/h |
| Voies_num | numérique (int) | Nombre de voies |

---

## Instructions par étudiant

### Issam - Régression Logistique
- **Fichiers à utiliser** : `X_oh_encoding.csv` + `y_gravite.csv`
- `X_oh_encoding` contient 81 colonnes après One-Hot Encoding
- ⚠️ Appliquer `StandardScaler` **après** le train/test split, jamais avant
- Utiliser `class_weight='balanced'` pour gérer le déséquilibre de classes

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Noé — Random Forest
- **Fichiers à utiliser** : `X_lab_encoded.csv` + `y_gravite.csv`
- `X_lab_encoded` contient 12 colonnes en Label Encoding
- Pas de standardisation nécessaire
- Utiliser `class_weight='balanced'`

### Enzo — XGBoost
- **Fichiers à utiliser** : `X_lab_encoded.csv` + `y_gravite.csv`
- Pas de standardisation nécessaire
- Gérer le déséquilibre via `scale_pos_weight` ou `sample_weight`

---

## Décisions de nettoyage importantes à mentionner dans le rapport

- **Doublons** : conservés intentionnellement — deux accidents peuvent avoir les mêmes caractéristiques sans être le même événement
- **Département → Région** : regroupement en 14 modalités pour éviter l'explosion dimensionnelle en OHE (107 → 14)
- **Vitesse max autorisée** : valeurs > 130 km/h considérées comme erreurs de saisie, passées en NaN — imputation par la médiane déléguée au pipeline de chaque binôme après le train/test split
- **"Non renseigné"** : remplacé par "Inconnu" dans toutes les colonnes catégorielles — l'absence d'information est conservée comme signal potentiel
- **#VALEURMULTI** : artifact du système source, traité comme valeur manquante et imputé par la médiane

---

## Protocole commun à tous les modèles

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

⚠️ Utiliser `stratify=y` pour conserver la proportion des classes dans train et test.  
⚠️ Toujours fixer `random_state=42` pour la reproductibilité des résultats.

---

## Métriques d'évaluation attendues pour chaque modèle
- Accuracy
- Précision par classe
- Rappel par classe
- F1-score par classe
- Matrice de confusion
```
