"""Alignement dataset -> HPO par des LLM locaux (Ollama) : approche proposée par le tuteur.

Deux conditions expérimentales :
  - "zero-shot" : le LLM ne reçoit que la description des colonnes et donne les IDs HPO de mémoire ;
  - "grounded"  : le LLM propose d'abord des requêtes de recherche, on interroge hp.owl,
                  puis le LLM choisit ses termes parmi les candidats réels retournés.

Chaque réponse est mise en cache dans llm_mappings/ : le notebook ne réinterroge pas les
modèles si le fichier existe déjà.
Seules des descriptions de colonnes et des statistiques agrégées sont envoyées au LLM,
jamais de données patient ni la cible.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "llm_mappings"
OLLAMA_HOST = "http://127.0.0.1:11434"

# Modèles locaux (Ollama) : réglage du raisonnement (`think`, None = défaut du modèle) et
# paramètres d'échantillonnage recommandés par chaque éditeur. On n'utilise PAS une
# température de 0 : avec un décodage glouton, les modèles à raisonnement tournent en boucle
# (observé avec gpt-oss et gemma4, qui répétaient la même phrase jusqu'à saturer le contexte).
MODELS = {
    "gpt-oss:20b": {"think": "high", "sampling": {"temperature": 1.0, "top_p": 1.0}},                # OpenAI
    "qwen3:14b":   {"think": True,   "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}},  # Alibaba
    "gemma4:12b":  {"think": None,   "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}},  # Google
    "llama3.1:8b": {"think": None,   "sampling": {"temperature": 0.6, "top_p": 0.9}},                # Meta
}
# Graine fixe : à matériel et version d'Ollama identiques, les réponses sont reproductibles
COMMON_OPTIONS = {"seed": 42, "num_ctx": 32768}
CONDITIONS = ["zero-shot", "grounded"]
MATCH_TYPES = ["exact", "broader", "approximate"]

# --------------------------------------------------------------------------------------
# Description du dataset (ce que le LLM « voit »)
# --------------------------------------------------------------------------------------
COLUMN_DOCS = {
    "age": "Age of the patient (years).",
    "sex": "Sex of the patient.",
    "chest_pain_type": "Type of chest pain reported by the patient.",
    "resting_blood_pressure": "Resting systolic blood pressure on admission to the hospital (mmHg).",
    "cholesterol": "Serum total cholesterol (mg/dl).",
    "fasting_blood_sugar": "Fasting blood sugar, dichotomised at 120 mg/dl.",
    "resting_ecg": "Resting electrocardiographic result.",
    "max_heart_rate": "Maximum heart rate achieved during an exercise stress test (beats per minute).",
    "exercise_induced_angina": "Angina induced by the exercise stress test.",
    "st_depression": "ST depression induced by exercise relative to rest (mm).",
    "st_slope": "Slope of the peak exercise ST segment.",
    "num_major_vessels": "Number of major coronary vessels (0-3) colored by fluoroscopy during angiography.",
    "thalassemia": "Result of the thallium-201 myocardial perfusion scintigraphy stress test "
                   "(the column name is historical; it is NOT the blood disorder thalassemia).",
}


def describe_columns(df: pd.DataFrame, num_feats, cat_feats) -> str:
    """Table texte des colonnes : sens clinique, type, plage observée ou modalités exactes."""
    lines = []
    for col in list(num_feats) + list(cat_feats):
        if col in num_feats:
            s = df[col]
            info = f"numeric, observed range {s.min():g}-{s.max():g}, median {s.median():g}"
        else:
            cats = ", ".join(repr(v) for v in sorted(df[col].dropna().unique()))
            info = f"categorical, values: {cats}"
        lines.append(f"- `{col}` ({info}): {COLUMN_DOCS[col]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Schémas de sortie structurée
# --------------------------------------------------------------------------------------
MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string", "description": "Main dataset column used by the rule."},
                    "rule": {"type": "string", "description": "pandas DataFrame.eval boolean expression."},
                    "hp_id": {"type": "string", "description": "HPO identifier, format HP:0000000."},
                    "hp_label": {"type": "string", "description": "Exact HPO label of the class."},
                    "match_type": {"type": "string", "enum": MATCH_TYPES},
                    "justification": {"type": "string"},
                },
                "required": ["variable", "rule", "hp_id", "hp_label", "match_type", "justification"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rules"],
    "additionalProperties": False,
}

QUERIES_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string"},
                    "search_terms": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["variable", "search_terms"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
SYSTEM = (
    "You are an expert in clinical informatics and biomedical ontologies, in particular the "
    "Human Phenotype Ontology (HPO)."
)

TASK = """\
We have a tabular dataset of patients referred for coronary angiography (Cleveland heart disease dataset).
Columns:
{columns}

Task: align the columns of this dataset with classes of the Human Phenotype Ontology (HPO).
For each column, write rules that map values (or value ranges) to the HPO class describing the
abnormality those values indicate. Each HPO class will become an additional binary column
(1 if the rule is true for the patient, 0 otherwise) used to train a machine-learning model.

Constraints:
- HPO describes abnormalities: normal values get no class. Columns that are not phenotypes may get no rule.
- Choose thresholds from established clinical knowledge, not from this specific dataset.
- Each `rule` must be a valid pandas `DataFrame.eval` boolean expression using only the column names
  above and, for categorical columns, the exact values listed (e.g. `col == 'value'`, `col >= 140`,
  `col in ['a', 'b']`). A rule may combine several columns.
- Several rules may point to the same HPO class, and one column may yield several classes.
- `hp_id` must be a real HPO identifier (format HP:0000000) and `hp_label` its exact label.
- `match_type`: "exact" if the class describes the finding exactly, "broader" if you had to use a
  more general class, "approximate" if it is a clinical interpretation.
{grounding}"""

GROUNDING = """
Use ONLY HPO classes from the following candidate list (retrieved from the HPO release we use).
If no candidate fits a column, give that column no rule.
{candidates}
"""

QUERIES_TASK = """\
We have a tabular dataset of patients referred for coronary angiography (Cleveland heart disease dataset).
Columns:
{columns}

We want to align these columns with classes of the Human Phenotype Ontology (HPO).
For each column, give 1 to 5 short English search terms (as they could appear in HPO labels or
synonyms) to look up the HPO classes describing abnormal values of that column."""


# --------------------------------------------------------------------------------------
# Recherche dans HPO (pour la condition "grounded" et la validation)
# --------------------------------------------------------------------------------------
class HPOIndex:
    """Index léger des termes HPO actifs : labels, synonymes exacts, parents."""

    def __init__(self, onto):
        from owlready2 import ThingClass

        self.terms = {}
        for c in onto.classes():
            if not c.name.startswith("HP_"):
                continue
            hp_id = c.name.replace("_", ":")
            label = c.label.first()
            # IAO_0100001 = "term replaced by" (IRI ou entité selon owlready2)
            replaced = [m.group(0).replace("_", ":") for x in getattr(c, "IAO_0100001", [])
                        if (m := re.search(r"HP[_:]\d+", str(getattr(x, "name", x))))]
            self.terms[hp_id] = {
                "label": str(label) if label else "",
                "synonyms": [str(s) for s in c.hasExactSynonym],
                "deprecated": bool(c.deprecated),
                "replaced_by": replaced[0] if replaced else None,
                "parents": [p.label.first() for p in c.is_a
                            if isinstance(p, ThingClass) and p.name.startswith("HP_")],
            }
        active = {i: t for i, t in self.terms.items() if not t["deprecated"]}
        self.by_label = {t["label"].lower(): i for i, t in active.items()}
        self.by_synonym = {s.lower(): i for i, t in active.items() for s in t["synonyms"]}

    def lookup_label(self, label: str):
        """ID du terme actif dont le label (ou à défaut un synonyme exact) est `label`."""
        key = label.strip().lower()
        return self.by_label.get(key) or self.by_synonym.get(key)

    def search(self, query: str, k: int = 8):
        """Termes actifs dont le label ou un synonyme contient tous les mots de la requête."""
        words = re.findall(r"[a-z0-9]+", query.lower())
        if not words:
            return []
        hits = []
        for hp_id, t in self.terms.items():
            if t["deprecated"]:
                continue
            names = [t["label"]] + t["synonyms"]
            if any(all(w in n.lower() for w in words) for n in names):
                exact = any(n.lower() == query.lower() for n in names)
                hits.append((not exact, len(t["label"]), hp_id))
        return [h[2] for h in sorted(hits)[:k]]

    def format_candidates(self, queries: dict[str, list[str]], k: int = 8) -> str:
        blocks = []
        for var, terms in queries.items():
            ids = []
            for q in terms:
                ids += [i for i in self.search(q, k) if i not in ids]
            lines = [f"  - {i} | {self.terms[i]['label']} | parents: {', '.join(self.terms[i]['parents'])}"
                     for i in ids]
            blocks.append(f"`{var}`:\n" + ("\n".join(lines) if lines else "  (no candidate found)"))
        return "\n".join(blocks)


# --------------------------------------------------------------------------------------
# Appels aux modèles locaux (avec cache)
# --------------------------------------------------------------------------------------
def _call(client, model: str, prompt: str, schema: dict) -> dict:
    """Un appel Ollama avec sortie JSON contrainte par le schéma. Retourne le JSON + les métadonnées."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    think = MODELS[model]["think"]
    options = {**MODELS[model]["sampling"], **COMMON_OPTIONS}
    t0 = time.time()
    response = client.chat(model=model, messages=messages, format=schema, options=options, think=think)
    meta = {
        "model": model,
        "think": think,
        "options": options,
        "done_reason": response.done_reason,
        "prompt_tokens": response.prompt_eval_count,
        "output_tokens": response.eval_count,
        "seconds": round(time.time() - t0, 1),
    }
    if response.done_reason == "length":
        raise RuntimeError(f"{model} : réponse tronquée (contexte de {COMMON_OPTIONS['num_ctx']} tokens atteint)")
    return {"data": json.loads(response.message.content), "thinking": response.message.thinking, "meta": meta}


def cache_path(model: str, condition: str, run: int = 1) -> Path:
    return CACHE_DIR / f"{model.replace(':', '-')}__{condition}__run{run}.json"  # pas de ':' dans un nom de fichier


def get_llm_mapping(model, condition, df, num_feats, cat_feats, hpo_index=None, run=1, client=None):
    """Retourne le mapping (et ses métadonnées) d'un LLM ; appelle l'API seulement si absent du cache."""
    path = cache_path(model, condition, run)
    if path.exists():
        return json.loads(path.read_text())

    import ollama
    client = client or ollama.Client(host=OLLAMA_HOST)
    columns = describe_columns(df, num_feats, cat_feats)
    record = {"model": model, "condition": condition, "run": run,
              "date": time.strftime("%Y-%m-%d %H:%M:%S"), "calls": []}

    grounding = ""
    if condition == "grounded":
        q = _call(client, model, QUERIES_TASK.format(columns=columns), QUERIES_SCHEMA)
        record["calls"].append({"step": "queries", **q})
        queries = {e["variable"]: e["search_terms"] for e in q["data"]["queries"]}
        record["queries"] = queries
        record["candidates"] = hpo_index.format_candidates(queries)
        grounding = GROUNDING.format(candidates=record["candidates"])

    prompt = TASK.format(columns=columns, grounding=grounding)
    m = _call(client, model, prompt, MAPPING_SCHEMA)
    record["calls"].append({"step": "mapping", **m})
    record["prompt"] = prompt
    record["rules"] = m["data"]["rules"]

    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return record


# --------------------------------------------------------------------------------------
# Validation d'un mapping produit par un LLM
# --------------------------------------------------------------------------------------
def validate_rules(rules: list[dict], df: pd.DataFrame, hpo_index: HPOIndex, allowed_cols) -> pd.DataFrame:
    """Vérifie chaque règle et décide si elle est utilisable.

    Un LLM retient mieux les libellés que les numéros : quand l'ID et le libellé se
    contredisent, on se fie au libellé s'il désigne un vrai terme HPO.

    Statuts :
      ok              ID valide et libellé correct
      obsolete_id     ID obsolète -> remplacé par le terme qui le remplace dans HPO
      id_repaired     ID inexistant ou contredit par le libellé, mais le libellé est un vrai
                      terme HPO (label ou synonyme exact) -> on prend l'ID du libellé
      label_mismatch  ID réel, libellé inconnu de HPO -> on garde l'ID (signe d'hallucination)
      invalid_id      ni l'ID ni le libellé n'existent dans HPO -> règle rejetée
      invalid_rule    expression non exécutable ou colonne inconnue -> règle rejetée
    """
    rows = []
    allowed = set(allowed_cols)
    for r in rules:
        row = dict(r)
        hp_id = r["hp_id"].strip().replace("_", ":")
        term = hpo_index.terms.get(hp_id)
        label_id = hpo_index.lookup_label(r["hp_label"])
        active = term is not None and not term["deprecated"]

        if active and label_id == hp_id:
            row["status"] = "ok"
        elif label_id:
            hp_id, row["status"] = label_id, "id_repaired"
        elif term and term["deprecated"] and term["replaced_by"]:
            hp_id, row["status"] = term["replaced_by"], "obsolete_id"
        elif active:
            row["status"] = "label_mismatch"
        else:
            row["status"] = "invalid_id"

        used = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", re.sub(r"'[^']*'|\"[^\"]*\"", "", r["rule"])))
        used -= {"and", "or", "not", "in", "True", "False"}
        if row["status"] != "invalid_id":
            if not used <= allowed:
                row["status"] = "invalid_rule"
            else:
                try:
                    res = df.eval(r["rule"])
                    if res.dtype != bool:
                        raise ValueError("expression non booléenne")
                except Exception:
                    row["status"] = "invalid_rule"

        row["hp_id"] = hp_id
        row["hp_label_hpo"] = hpo_index.terms[hp_id]["label"] if hp_id in hpo_index.terms else None
        row["usable"] = row["status"] in ("ok", "obsolete_id", "id_repaired", "label_mismatch")
        rows.append(row)
    return pd.DataFrame(rows)
