# Pipeline_LLM

Alignement de variables cliniques vers l'ontologie **HPO** à l'aide de LLM locaux
(servis par [Ollama](https://ollama.com)), puis construction de variables
ontologiques pour la prédiction. Cas d'usage : jeu de données *heart disease* (UCI).

## Contenu

| Fichier | Rôle |
|---|---|
| `llm_mapping.py` | Alignement colonnes → classes HPO par LLM, en deux conditions : `zero-shot` (le modèle répond de mémoire) et `grounded` (le modèle choisit parmi des candidats réellement extraits de `hp.owl`). Réponses mises en cache dans `llm_mappings/`. |
| `threshold_llm.py` | Règles cliniques (seuils numériques, modalités catégorielles) proposées par le LLM, puis seuil choisi par le LLM parmi des candidats évalués sur les patients d'entraînement de chaque fold (critère AUC ou Gini). Cache dans `llm_thresholds/`. |
| `heart_ontology_2.ipynb` | Notebook d'expérimentation et d'évaluation. |
| `heart.csv` | Jeu de données *heart disease* (UCI). |
| `llm_mappings/` | Alignements mis en cache, par modèle et par condition. |
| `llm_thresholds/` | Règles et seuils mis en cache, par modèle, critère et fold. |

Modèles utilisés : `gpt-oss:20b`, `qwen3:14b`, `gemma4:12b`, `llama3.1:8b`
(graine fixée à 42, contexte de 32 768 tokens).

Aucune donnée patient individuelle n'est envoyée aux modèles : uniquement des
descriptions de colonnes et des statistiques agrégées.

## Ontologie HPO

`hp.owl` (~76 Mo) **n'est pas versionné** dans ce dépôt. Il faut le télécharger
à la racine de `Pipeline_LLM/` avant d'exécuter les scripts :

```bash
curl -L -o hp.owl https://purl.obolibrary.org/obo/hp.owl
```

## Prérequis

- Ollama en écoute sur `http://127.0.0.1:11434`, avec les modèles ci-dessus téléchargés ;
- Python avec `pandas`, `numpy`, `scikit-learn`, `rdflib`, `owlready2`, `catboost`, `xgboost`.
