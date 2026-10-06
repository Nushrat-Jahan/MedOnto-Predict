# MedOnto-Predict

**MedOnto-Predict** is a research project investigating the integration of medical ontology knowledge into machine-learning pipelines for clinical prediction.

The project focuses on comparing conventional machine-learning approaches based on clinical data with models enriched by semantic knowledge derived from a medical ontology.

The initial use case concerns **rheumatoid arthritis prediction**.

---

## Overview

Conventional clinical prediction models mainly rely on tabular patient data and make limited use of explicit medical knowledge represented through ontologies.

MedOnto-Predict explores how knowledge represented in a medical ontology can be integrated into a machine-learning prediction pipeline.

The project focuses on the **use of a medical ontology for prediction rather than on ontology construction**.

---

## Pipeline

```text
Clinical Data
     |
     v
Clinical Variable–Ontology Mapping
     |
     v
Ontology / Knowledge Graph
     |
     +-------------------------+
     |                         |
     v                         v
Graph Embeddings          Graph Representation
(RDF2Vec, OWL2Vec*,       for future GNN use
 Node2Vec)
     |
     v
Machine-Learning Models
(RF, XGBoost, MLP)
     |
     v
Evaluation
```

---

## Project Components

### Ontology Processing

Tools for loading and exploring medical ontologies, including:

- OWL ontology loading;
- class and property exploration;
- RDFLib / SPARQL querying;
- retrieval of ontology concepts associated with clinical variables.

### Clinical Data–Ontology Mapping

Clinical variables are mapped to corresponding concepts represented in the ontology.

Example:

```text
CRP          → Inflammatory Biomarker
Smoking      → Environmental Exposure
Methotrexate → DMARD
Joint Pain   → Symptom
DAS28        → Disease Activity
```

The objective is to create a structured representation linking patient data to concepts in the ontology.

### Knowledge Representation

Ontology and knowledge-graph information can be transformed into representations suitable for machine learning.

Approaches considered include:

- RDF2Vec;
- OWL2Vec*;
- Node2Vec.

Graph representations suitable for future Graph Neural Network experiments may also be investigated.

### Prediction Models

Reference machine-learning models include:

- Random Forest;
- XGBoost;
- Multilayer Perceptron (MLP).

The models can be evaluated using conventional clinical variables as well as ontology-enhanced representations.

### Evaluation

The project evaluates and compares prediction models using conventional clinical data and ontology-enhanced representations.  
Evaluation metrics may include:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC.

---

## Research Objective

The project investigates the contribution of ontology-based knowledge to clinical prediction by comparing conventional machine-learning models with models using ontology-enhanced representations.

---

## Data

Clinical or patient-level datasets are **not distributed in this repository**.

This public repository is intended to contain source code, documentation, configuration files, and ontology-related resources that can legally be redistributed.

Users must obtain any restricted clinical datasets independently and comply with their corresponding access and usage requirements.

---

## Status

 **Work in progress**

This repository supports ongoing research on ontology-enhanced machine learning for clinical prediction.

---

## License

This project is licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file for details.
