"""Seuils cliniques proposés par un LLM, puis ajustés par le LLM sur les données d'entraînement.

Étape 1 (une seule fois, sans données) : le LLM (grounded sur hp.owl) propose des règles
    - numériques   : variable, opérateur, seuil clinique de départ, classe HPO
    - catégorielles : variable, modalités, classe HPO
Étape 2 (dans chaque fold de la validation croisée) : pour chaque règle numérique, le code
    calcule sur les patients d'ENTRAÎNEMENT du fold une table de seuils candidats autour du
    seuil clinique, avec leur score selon un critère (AUC ou impureté de Gini). Le LLM choisit
    le seuil. Il ne voit jamais les patients de test, ni aucun patient individuel.

Méthode fixée à l'avance (avant de regarder les résultats) :
    - fenêtre des candidats : seuil clinique ± 1 écart-type de la variable (entraînement) ;
    - candidats : milieux entre valeurs observées consécutives dans la fenêtre + seuil clinique,
      dédoublonnés (deux seuils qui annotent les mêmes patients ne comptent qu'une fois),
      au plus MAX_CANDIDATES ;
    - critères : AUC de la colonne binaire (séparation, max(AUC, 1 - AUC)) ou impureté de Gini
      pondérée après découpage (plus faible = meilleur).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import llm_mapping as llm

MODEL = "gemma4:12b"
CACHE_DIR = Path(__file__).parent / "llm_thresholds"
OPERATORS = [">=", ">", "<=", "<"]
CRITERIA = ["auc", "gini"]
MAX_CANDIDATES = 15
WINDOW_SD = 1.0

# --------------------------------------------------------------------------------------
# Étape 1 : règles + seuils cliniques (une fois)
# --------------------------------------------------------------------------------------
RULES_SCHEMA = {
    "type": "object",
    "properties": {
        "numeric_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string"},
                    "operator": {"type": "string", "enum": OPERATORS},
                    "clinical_threshold": {"type": "number"},
                    "hp_id": {"type": "string"},
                    "hp_label": {"type": "string"},
                    "match_type": {"type": "string", "enum": llm.MATCH_TYPES},
                    "justification": {"type": "string"},
                },
                "required": ["variable", "operator", "clinical_threshold", "hp_id", "hp_label",
                             "match_type", "justification"],
                "additionalProperties": False,
            },
        },
        "categorical_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "string"}},
                    "hp_id": {"type": "string"},
                    "hp_label": {"type": "string"},
                    "match_type": {"type": "string", "enum": llm.MATCH_TYPES},
                    "justification": {"type": "string"},
                },
                "required": ["variable", "values", "hp_id", "hp_label", "match_type", "justification"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["numeric_rules", "categorical_rules"],
    "additionalProperties": False,
}

RULES_TASK = """\
We have a tabular dataset of patients referred for coronary angiography (Cleveland heart disease dataset).
Columns:
{columns}

Task: align the columns of this dataset with classes of the Human Phenotype Ontology (HPO).
Each HPO class will become an additional binary column (1 if the patient has the abnormality).

- For a numeric column, give a `numeric_rule`: `variable <operator> clinical_threshold` is true when the
  value is abnormal. The threshold must come from established clinical guidelines or definitions,
  not from this specific dataset. It will later be fine-tuned on training data, so give the most
  standard clinical cut-off.
- For a categorical column, give a `categorical_rule`: the patient has the class if the value is one
  of `values` (use the exact values listed above).
- HPO describes abnormalities: normal values get no class. Columns that are not phenotypes may get no rule.
- A column may yield several rules (several classes), and several rules may point to the same class.
- `match_type`: "exact" if the class describes the finding exactly, "broader" if you had to use a more
  general class, "approximate" if it is a clinical interpretation.

Use ONLY HPO classes from the following candidate list (retrieved from the HPO release we use).
If no candidate fits a column, give that column no rule.
{candidates}
"""


def rules_path() -> Path:
    return CACHE_DIR / f"{MODEL.replace(':', '-')}__rules.json"


def get_rules(df, num_feats, cat_feats, hpo_index, client=None) -> dict:
    """Étape 1 (grounded) : requêtes de recherche -> candidats HPO -> règles. Mis en cache."""
    path = rules_path()
    if path.exists():
        return json.loads(path.read_text())
    import ollama
    client = client or ollama.Client(host=llm.OLLAMA_HOST)
    columns = llm.describe_columns(df, num_feats, cat_feats)

    q = llm._call(client, MODEL, llm.QUERIES_TASK.format(columns=columns), llm.QUERIES_SCHEMA)
    queries = {e["variable"]: e["search_terms"] for e in q["data"]["queries"]}
    candidates = hpo_index.format_candidates(queries)
    prompt = RULES_TASK.format(columns=columns, candidates=candidates)
    r = llm._call(client, MODEL, prompt, RULES_SCHEMA)

    record = {"model": MODEL, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "queries": queries,
              "candidates": candidates, "prompt": prompt, "calls": [q, r], **r["data"]}
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return record


def validate(record, df, hpo_index, num_feats, cat_feats):
    """Valide les règles de l'étape 1. Retourne (règles numériques, règles catégorielles) utilisables."""
    num, cat = [], []
    for r in record["numeric_rules"]:
        ok_col = r["variable"] in num_feats
        rule = f"{r['variable']} {r['operator']} {r['clinical_threshold']}" if ok_col else "invalid"
        v = llm.validate_rules([{**r, "rule": rule}], df, hpo_index, num_feats + cat_feats).iloc[0]
        num.append({**r, "hp_id": v.hp_id, "hp_label_hpo": v.hp_label_hpo, "status": v.status,
                    "usable": bool(v.usable and ok_col)})
    for r in record["categorical_rules"]:
        ok_col = r["variable"] in cat_feats and set(r["values"]) <= set(df[r["variable"]].unique())
        rule = f"{r['variable']} in {list(r['values'])!r}" if ok_col else "invalid"
        v = llm.validate_rules([{**r, "rule": rule}], df, hpo_index, num_feats + cat_feats).iloc[0]
        cat.append({**r, "hp_id": v.hp_id, "hp_label_hpo": v.hp_label_hpo, "status": v.status,
                    "usable": bool(v.usable and ok_col)})
    num, cat = pd.DataFrame(num), pd.DataFrame(cat)
    num.insert(0, "rule_id", [f"N{i}" for i in range(len(num))])
    cat.insert(0, "rule_id", [f"C{i}" for i in range(len(cat))])
    # Règles numériques identiques (même variable, opérateur et seuil) : un seul seuil à ajuster,
    # partagé par toutes leurs classes HPO
    keys = list(zip(num.variable, num.operator, num.clinical_threshold))
    groups = {k: f"G{i}" for i, k in enumerate(dict.fromkeys(keys))}
    num.insert(1, "group", [groups[k] for k in keys])
    return num, cat


# --------------------------------------------------------------------------------------
# Étape 2 : seuils candidats sur l'entraînement + choix du LLM (dans chaque fold)
# --------------------------------------------------------------------------------------
def apply_op(x, op, t):
    return {">=": x >= t, ">": x > t, "<=": x <= t, "<": x < t}[op]


def gini_impurity(flag, y):
    """Impureté de Gini pondérée après découpage par `flag` (0 = pur, 0.5 = maximal)."""
    total = 0.0
    for g in (flag, ~flag):
        if g.sum() == 0:
            continue
        p = y[g].mean()
        total += g.mean() * (1 - p ** 2 - (1 - p) ** 2)
    return total


def candidate_table(x_train, y_train, op, t_clin):
    """Seuils candidats autour du seuil clinique, avec leurs scores sur l'entraînement."""
    x, y = np.asarray(x_train, float), np.asarray(y_train)
    lo, hi = t_clin - WINDOW_SD * x.std(), t_clin + WINDOW_SD * x.std()
    u = np.unique(x)
    mids = (u[:-1] + u[1:]) / 2
    cands = sorted(set(np.round(mids[(mids >= lo) & (mids <= hi)], 3)) | {float(t_clin)})
    rows, seen = [], {}
    for t in cands:
        flag = apply_op(x, op, t)
        if flag.all() or not flag.any():
            continue
        key = flag.tobytes()
        if key in seen and t != t_clin:      # même découpage : on garde un seul représentant
            continue
        seen[key] = t
        a = roc_auc_score(y, flag)
        rows.append({"threshold": float(t), "flagged_%": round(100 * flag.mean(), 1),
                     "auc": round(max(a, 1 - a), 4), "gini": round(gini_impurity(flag, y), 4),
                     "is_clinical": t == t_clin})
    tab = pd.DataFrame(rows).drop_duplicates("threshold")
    if len(tab) > MAX_CANDIDATES:                   # sous-échantillonnage régulier, clinique conservé
        idx = set(np.linspace(0, len(tab) - 1, MAX_CANDIDATES).round().astype(int))
        tab = tab[[i in idx or c for i, c in enumerate(tab.is_clinical)]]
    return tab.reset_index(drop=True)


CRITERION_TEXT = {
    "auc": "`auc` = AUC of the binary column for separating the two outcome classes on the training "
           "patients (0.5 = no separation, 1 = perfect separation; HIGHER is better)",
    "gini": "`gini` = weighted Gini impurity of the outcome after splitting the training patients with "
            "the rule (0 = pure groups, 0.5 = no information; LOWER is better)",
}

ADJUST_TASK = """\
We enrich a tabular dataset of patients referred for coronary angiography with binary columns derived
from HPO classes. Each rule below starts from a clinical threshold. We now fine-tune each threshold on
the TRAINING patients only.

For each rule, a table lists candidate thresholds around the clinical one, with:
- `flagged_%` = percentage of training patients for whom the rule is true,
- {criterion_text}.

Choose, for each rule, the threshold from its table that you consider best: favour the data criterion,
but keep the threshold clinically plausible (do not pick an extreme value for a negligible gain).
`chosen_threshold` must be one of the thresholds listed in the rule's table.

{tables}
"""

ADJUST_SCHEMA = {
    "type": "object",
    "properties": {
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string"},
                    "chosen_threshold": {"type": "number"},
                    "justification": {"type": "string"},
                },
                "required": ["rule_id", "chosen_threshold", "justification"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["choices"],
    "additionalProperties": False,
}


def fold_path(criterion: str, fold: int) -> Path:
    return CACHE_DIR / f"{MODEL.replace(':', '-')}__{criterion}__fold{fold:02d}.json"


def adjust_fold(num_rules, df_train, y_train, criterion, fold, client=None) -> dict:
    """Le LLM choisit les seuils d'un fold à partir des tables calculées sur l'entraînement. Mis en cache."""
    path = fold_path(criterion, fold)
    if path.exists():
        return json.loads(path.read_text())

    tables, cand = [], {}
    cols = ["threshold", "flagged_%", criterion]
    for g, rules in num_rules.groupby("group", sort=False):
        r = rules.iloc[0]
        tab = candidate_table(df_train[r.variable], y_train, r.operator, r.clinical_threshold)
        cand[g] = tab
        if len(tab) <= 1:                            # rien à choisir : pas envoyé au LLM
            continue
        labels = ", ".join(dict.fromkeys(rules.hp_label_hpo))
        tables.append(f"Rule {g}: `{r.variable} {r.operator} threshold` -> {labels} "
                      f"(clinical threshold: {r.clinical_threshold:g})\n" + tab[cols].to_string(index=False))

    import ollama
    client = client or ollama.Client(host=llm.OLLAMA_HOST)
    prompt = ADJUST_TASK.format(criterion_text=CRITERION_TEXT[criterion], tables="\n\n".join(tables))
    res = llm._call(client, MODEL, prompt, ADJUST_SCHEMA)

    chosen = {c["rule_id"]: c for c in res["data"]["choices"]}
    thresholds = {}
    clinical = dict(zip(num_rules.group, num_rules.clinical_threshold))
    for rid, tab in cand.items():
        c = chosen.get(rid)
        if tab.empty:                                # aucun découpage possible -> seuil clinique
            t, status = float(clinical[rid]), "no_candidates"
        elif len(tab) == 1:
            t, status = float(tab.threshold.iloc[0]), "single_candidate"
        elif c is None:                              # règle oubliée par le LLM -> seuil clinique
            t, status = float(clinical[rid]), "missing->clinical"
        else:                                        # valeur hors table -> candidat le plus proche
            t = float(tab.threshold.iloc[(tab.threshold - c["chosen_threshold"]).abs().argmin()])
            status = "ok" if np.isclose(t, c["chosen_threshold"]) else "snapped"
        thresholds[rid] = {"threshold": t, "status": status}

    record = {"model": MODEL, "criterion": criterion, "fold": fold, "prompt": prompt,
              "thresholds": thresholds, "call": res,
              "tables": {k: v.to_dict("records") for k, v in cand.items()}}
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return record


# --------------------------------------------------------------------------------------
# Construction des colonnes binaires
# --------------------------------------------------------------------------------------
def build_features(df_part, num_rules, cat_rules, thresholds=None):
    """Colonnes binaires par classe HPO (union des règles d'une même classe).
    `thresholds` : {groupe: seuil}. None = seuils cliniques de l'étape 1."""
    hits = {}
    for r in num_rules.itertuples():
        t = r.clinical_threshold if thresholds is None else thresholds[r.group]
        hits.setdefault(r.hp_id, []).append(apply_op(df_part[r.variable].to_numpy(float), r.operator, t))
    for r in cat_rules.itertuples():
        hits.setdefault(r.hp_id, []).append(df_part[r.variable].isin(r.values).to_numpy())
    return pd.DataFrame({k.replace(":", "_"): np.any(v, axis=0).astype(int) for k, v in hits.items()},
                        index=df_part.index)
