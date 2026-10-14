"""
Pipeline Modèle A vs Modèle B sur MIMIC-IV Demo (OMOP) avec la VRAIE hiérarchie SNOMED CT
(vocabulaire Athena). Maladie test : le diabète.

Étapes :
  1. Charger le vocabulaire Athena (CONCEPT.csv), seulement les concepts SNOMED standards
  2. Charger les diagnostics MIMIC (condition_concept_id = concept SNOMED, déjà mappé par OMOP)
  3. Vérifier le mapping (statistiques à mettre dans le mémoire)
  4. Construire le graphe de l'ontologie : concepts des patients + leurs ancêtres sous la
     racine RACINE, jusqu'à PROFONDEUR_MAX niveaux (comme KEEP). Les concepts plus
     profonds sont rattachés à leurs ancêtres situés à PROFONDEUR_MAX.
  5. Définir la cible (CIBLE_NOMS + descendants + mots-clés) et la retirer des variables
  6. Variables : A (un concept = une colonne), B1 roll-up, B2 node2vec, B3 KEEP, contrôle aléatoire
  7. Évaluation : validation croisée répétée, 3 modèles, AUPRC/AUC, tests appariés

Les liens directs "est un" viennent de CONCEPT_ANCESTOR.csv (min_levels_of_separation = 1).
CONCEPT_RELATIONSHIP.csv sert seulement à remapper ("Maps to") les concepts devenus non
standards depuis la conversion des données en OMOP.

Utilisation (Colab) : vérifier les deux chemins ci-dessous, puis
    %run pipeline_snomed.py
Dépendances : numpy, pandas, scipy, scikit-learn.
"""

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOSSIER_MIMIC = "/content/drive/MyDrive/Youssef/StageLirmm/mimic-iv-demo-data-in-the-omop-common-data-model-0.9/1_omop_data_csv"
DOSSIER_ATHENA = "/content/drive/MyDrive/Youssef/StageLirmm/vocabulary_download_v5_{fa4cccb1-e82c-4ae5-bc68-563863e183ee}_1791532254461"   # dossier qui contient CONCEPT.csv

# La cible = ces concepts ET tous leurs descendants. Dans SNOMED, les complications
# ("Chronic kidney disease due to type 2 diabetes mellitus"...) ne descendent PAS toutes de
# "Diabetes mellitus" : elles sont dans une branche voisine, d'où les deux autres racines.
# Les noms absents du vocabulaire sont simplement ignorés.
CIBLE_NOMS = ["Diabetes mellitus", "Disorder due to diabetes mellitus",
              "Complication due to diabetes mellitus"]
# Filet de sécurité : tout concept des patients dont le nom contient un de ces mots est
# aussi compté comme cible (sauf s'il contient un mot exclu). Pour la PR : ["rheumatoid"], [].
MOTS_CIBLE = ["diabet"]
MOTS_EXCLUS = ["insipidus", "prediabet", "pre-diabet", "without diabet", "non-diabet", "nondiabet"]
RACINE = "Disease"                # comme KEEP. "Clinical finding" garde aussi les symptômes
                                  # et les relevés de monitoring (rythme sinusal...)
PROFONDEUR_MAX = 5                # comme KEEP

# Ne garder, pour les malades, que les diagnostics posés AVANT le premier diagnostic de la
# cible. Indispensable sur MIMIC-IV complet. Sur la démo, l'historique avant le diabète est
# presque toujours vide (diagnostic dès le premier séjour) : laisser False pour tester le pipeline.
COUPURE_TEMPORELLE = False

DIM = 32
MIN_PATIENTS = 3
N2V = dict(walks_par_noeud=10, longueur=20, fenetre=5, negatifs=5, epochs=3, lr=0.025)
KEEP = dict(lam=1e-3, epochs=300, lr=0.05, alpha=0.75)
CV = dict(n_splits=5, n_repeats=5)
SEED = 42
TAILLE_MORCEAU = 5_000_000        # lignes lues à la fois dans les gros fichiers Athena


def lire_athena(chemin, **kw):
    """Les fichiers Athena sont séparés par des tabulations, sans guillemets."""
    return pd.read_csv(chemin, sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines="skip", **kw)


# ---------------------------------------------------------------------------
# Étape 1 : vocabulaire SNOMED
# ---------------------------------------------------------------------------
def charger_snomed(dossier):
    t = time.time()
    morceaux = []
    for ch in lire_athena(Path(dossier) / "CONCEPT.csv",
                          usecols=["concept_id", "concept_name", "domain_id",
                                   "vocabulary_id", "standard_concept"],
                          dtype={"concept_id": "int64"}, chunksize=TAILLE_MORCEAU):
        morceaux.append(ch[(ch["vocabulary_id"] == "SNOMED") & (ch["standard_concept"] == "S")]
                        [["concept_id", "concept_name", "domain_id"]])
    snomed = pd.concat(morceaux).set_index("concept_id")
    print(f"[1] Vocabulaire : {len(snomed):,} concepts SNOMED standards ({time.time() - t:.0f} s)")
    return snomed


def trouver_concept(snomed, nom, obligatoire=True):
    exact = snomed[snomed["concept_name"].str.lower() == nom.lower()]
    if len(exact) >= 1:
        if len(exact) > 1:   # préférer le domaine Condition
            cond = exact[exact["domain_id"] == "Condition"]
            exact = cond if len(cond) else exact
        return int(exact.index[0])
    if not obligatoire:
        return None
    proches = snomed[snomed["concept_name"].str.contains(nom, case=False, na=False, regex=False)]
    raise SystemExit(f"Concept '{nom}' introuvable. Noms proches :\n"
                     f"{proches['concept_name'].head(15).to_string()}")


# ---------------------------------------------------------------------------
# Étapes 2-3 : diagnostics et vérification du mapping
# ---------------------------------------------------------------------------
def remapper_non_standards(ids, dossier_athena):
    """Concepts non standards -> concepts standards actuels via la relation "Maps to".
    Nécessaire quand les données ont été converties avec une version plus ancienne du
    vocabulaire que celle téléchargée sur Athena (cas de MIMIC-IV Demo 0.9)."""
    ids = pd.Index(list(ids))
    t = time.time()
    morceaux = []
    for ch in lire_athena(Path(dossier_athena) / "CONCEPT_RELATIONSHIP.csv",
                          usecols=["concept_id_1", "concept_id_2", "relationship_id"],
                          dtype={"concept_id_1": "int64", "concept_id_2": "int64"},
                          chunksize=TAILLE_MORCEAU):
        morceaux.append(ch.loc[(ch["relationship_id"] == "Maps to") & ch["concept_id_1"].isin(ids),
                               ["concept_id_1", "concept_id_2"]])
    corr = pd.concat(morceaux)
    print(f"    CONCEPT_RELATIONSHIP lu ({time.time() - t:.0f} s)")
    return corr.rename(columns={"concept_id_1": "ancien", "concept_id_2": "nouveau"})


def charger_diagnostics(dossier, snomed, dossier_athena):
    d = Path(dossier)
    cond = pd.read_csv(d / "condition_occurrence.csv",
                       usecols=["person_id", "condition_concept_id", "condition_start_date",
                                "condition_source_value"])
    person = pd.read_csv(d / "person.csv")
    print(f"[2] Diagnostics : {len(cond):,} lignes, {cond['person_id'].nunique()} patients")

    non_mappe = cond["condition_concept_id"] == 0
    hors_snomed = ~cond["condition_concept_id"].isin(snomed.index) & ~non_mappe
    print(f"[3] Mapping :")
    print(f"    sans concept (concept_id = 0)      : {non_mappe.mean():.1%} des lignes")
    print(f"    concept hors SNOMED standard       : {hors_snomed.mean():.1%} des lignes "
          f"({cond.loc[hors_snomed, 'person_id'].nunique()} patients)")

    # Remappage des concepts non standards vers leur remplaçant standard actuel
    if hors_snomed.any():
        corr = remapper_non_standards(cond.loc[hors_snomed, "condition_concept_id"].unique(), dossier_athena)
        corr = corr[corr["nouveau"].isin(snomed.index)]
        a_remapper = cond[hors_snomed].merge(corr, left_on="condition_concept_id", right_on="ancien")
        a_remapper = a_remapper.assign(condition_concept_id=a_remapper["nouveau"]).drop(columns=["ancien", "nouveau"])
        recuperes = set(cond.index[hors_snomed]) & set(cond[hors_snomed].reset_index()
                    .merge(corr, left_on="condition_concept_id", right_on="ancien")["index"])
        print(f"    remappés via 'Maps to'             : {len(recuperes) / hors_snomed.sum():.0%} des lignes hors standard "
              f"-> {a_remapper['condition_concept_id'].nunique()} concepts standards")
        exemples = (cond[hors_snomed].merge(corr, left_on="condition_concept_id", right_on="ancien")
                    .assign(nouveau_nom=lambda x: snomed["concept_name"].reindex(x["nouveau"]).to_numpy())
                    .groupby("condition_source_value")["nouveau_nom"].first().head(6))
        for src, n in exemples.items():
            print(f"      ex. {src} -> {n}")
        perdus = hors_snomed.sum() - len(recuperes)
        print(f"    toujours sans concept standard     : {perdus / len(cond):.1%} des lignes (écartées)")
        ok = pd.concat([cond[~non_mappe & ~hors_snomed], a_remapper], ignore_index=True)
    else:
        ok = cond[~non_mappe]

    print(f"    {cond.loc[~non_mappe, 'condition_source_value'].nunique():,} valeurs sources distinctes "
          f"-> {ok['condition_concept_id'].nunique():,} concepts SNOMED standards")
    return ok, person


# ---------------------------------------------------------------------------
# Étape 4 : graphe de l'ontologie
# ---------------------------------------------------------------------------
def lire_ancetres(dossier, concepts_patients, racine_id, snomed_ids):
    """Une seule lecture de CONCEPT_ANCESTOR.csv, par morceaux. On garde :
       - les ancêtres des concepts des patients ;
       - la profondeur de chaque concept sous la racine ;
       - les liens directs "est un" (niveau 1) entre concepts SNOMED standards."""
    t = time.time()
    P = pd.Index(concepts_patients)
    S = pd.Index(snomed_ids)
    anc, prof, liens = [], [], []
    for ch in lire_athena(Path(dossier) / "CONCEPT_ANCESTOR.csv",
                          usecols=["ancestor_concept_id", "descendant_concept_id",
                                   "min_levels_of_separation"],
                          dtype="int64", chunksize=TAILLE_MORCEAU):
        a, d, lv = ch["ancestor_concept_id"], ch["descendant_concept_id"], ch["min_levels_of_separation"]
        anc.append(ch[d.isin(P) & (lv > 0)])
        prof.append(ch.loc[a == racine_id, ["descendant_concept_id", "min_levels_of_separation"]])
        liens.append(ch.loc[(lv == 1) & d.isin(S) & a.isin(S), ["ancestor_concept_id", "descendant_concept_id"]])
    anc = pd.concat(anc)
    profondeur = pd.concat(prof).groupby("descendant_concept_id")["min_levels_of_separation"].min()
    profondeur.loc[racine_id] = 0
    liens = pd.concat(liens)
    print(f"[4] CONCEPT_ANCESTOR lu ({time.time() - t:.0f} s)")
    return anc, profondeur, liens


def construire_ontologie(concepts_patients, anc, profondeur, liens, snomed):
    candidats = set(concepts_patients) | set(anc["ancestor_concept_id"])
    garde = {c for c in candidats if c in profondeur.index and profondeur[c] <= PROFONDEUR_MAX}

    # Représentation de chaque concept patient dans le graphe tronqué
    anc_par_concept = anc.groupby("descendant_concept_id")["ancestor_concept_id"].apply(set).to_dict()
    rep, hors_racine, remontes = {}, [], 0
    for p in concepts_patients:
        if p not in profondeur.index:
            hors_racine.append(p)
        elif profondeur[p] <= PROFONDEUR_MAX:
            rep[p] = {p}
        else:
            rep[p] = {a for a in anc_par_concept.get(p, ()) if profondeur.get(a, -1) == PROFONDEUR_MAX}
            remontes += 1

    # Liens directs entre noeuds gardés -> parents, puis ancêtres par parcours
    l = liens[liens["ancestor_concept_id"].isin(garde) & liens["descendant_concept_id"].isin(garde)]
    parents = l.groupby("descendant_concept_id")["ancestor_concept_id"].apply(list).to_dict()
    memo = {}

    def ancetres(x):
        if x not in memo:
            s = set()
            for p in parents.get(x, []):
                s.add(p)
                s |= ancetres(p)
            memo[x] = s
        return memo[x]

    for x in garde:
        ancetres(x)

    noms = sorted(garde)
    noeuds = {n: i for i, n in enumerate(noms)}
    voisins = {n: set() for n in noms}
    for a, d in zip(l["ancestor_concept_id"], l["descendant_concept_id"]):
        voisins[a].add(d)
        voisins[d].add(a)
    adj = [np.array([noeuds[v] for v in voisins[n]] or [noeuds[n]]) for n in noms]

    print(f"    {len(concepts_patients):,} concepts patients : {len(hors_racine):,} hors de '{RACINE}' (écartés), "
          f"{remontes:,} plus profonds que {PROFONDEUR_MAX} niveaux (rattachés à leurs ancêtres)")
    if hors_racine:
        ex = [snomed["concept_name"].get(c, str(c)) for c in hors_racine[:8]]
        print(f"    exemples écartés : {ex}")
    print(f"    Graphe : {len(noeuds):,} noeuds, {len(l):,} liens 'est un'")
    return noeuds, adj, rep, memo


# ---------------------------------------------------------------------------
# node2vec et KEEP (identiques à pipeline_test_diabete.py)
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def node2vec(adj, dim, walks_par_noeud, longueur, fenetre, negatifs, epochs, lr, rng):
    n = len(adj)
    walks = []
    for _ in range(walks_par_noeud):
        for depart in rng.permutation(n):
            w = [depart]
            for _ in range(longueur - 1):
                w.append(rng.choice(adj[w[-1]]))
            walks.append(w)
    centres, contextes = [], []
    for w in walks:
        for i, c in enumerate(w):
            for j in range(max(0, i - fenetre), min(len(w), i + fenetre + 1)):
                if j != i:
                    centres.append(c)
                    contextes.append(w[j])
    centres, contextes = np.array(centres), np.array(contextes)
    freq = np.bincount(np.concatenate(walks), minlength=n) ** 0.75
    p_neg = freq / freq.sum()
    W = (rng.random((n, dim)) - 0.5) / dim
    C = np.zeros((n, dim))
    taille_lot = max(16, min(1024, 2 * n))
    for ep in range(epochs):
        pas = lr * (1 - ep / epochs)
        ordre = rng.permutation(len(centres))
        for s in range(0, len(ordre), taille_lot):
            lot = ordre[s:s + taille_lot]
            c, o = centres[lot], contextes[lot]
            ng = rng.choice(n, size=(len(lot), negatifs), p=p_neg)
            vc, vo, vn = W[c], C[o], C[ng]
            g_pos = sigmoid((vc * vo).sum(1)) - 1
            g_neg = sigmoid(np.einsum("bd,bkd->bk", vc, vn))
            grad_c = g_pos[:, None] * vo + np.einsum("bk,bkd->bd", g_neg, vn)
            np.add.at(W, c, -pas * grad_c)
            np.add.at(C, o, -pas * g_pos[:, None] * vc)
            np.add.at(C, ng.ravel(), -pas * (g_neg[:, :, None] * vc[:, None, :]).reshape(-1, dim))
    return W


def keep(cooc, W_n2v, lam, epochs, lr, alpha):
    masque = cooc > 0
    if not masque.any():
        return W_n2v.copy()
    xmax = np.percentile(cooc[masque], 75)
    F = np.where(masque, np.minimum(1.0, (cooc / xmax) ** alpha), 0).astype(np.float32)
    logX = np.log(np.where(masque, cooc, 1)).astype(np.float32)
    W0 = W_n2v.astype(np.float32)
    W, Wt = W0.copy(), W0.copy()
    b, bt = np.zeros(len(W), np.float32), np.zeros(len(W), np.float32)
    hW, hWt, hb, hbt = (np.full_like(a, 1e-8) for a in (W, Wt, b, bt))
    for _ in range(epochs):
        G = F * (W @ Wt.T + b[:, None] + bt[None, :] - logX)
        gW = G @ Wt + lam * (W - W0)
        gWt = G.T @ W
        for p, gp, h in ((W, gW, hW), (Wt, gWt, hWt), (b, G.sum(1), hb), (bt, G.sum(0), hbt)):
            h += gp ** 2
            p -= lr * gp / np.sqrt(h)
    return W.astype(np.float64)


# ---------------------------------------------------------------------------
# Variables par patient
# ---------------------------------------------------------------------------
def multi_hot(sets_par_patient, patients):
    lignes = [(pid, c) for pid in patients for c in sets_par_patient.get(pid, ())]
    if not lignes:
        return pd.DataFrame(index=patients)
    df = pd.DataFrame(lignes, columns=["person_id", "variable"])
    m = pd.crosstab(df["person_id"], df["variable"]).clip(upper=1).reindex(patients, fill_value=0)
    return m.loc[:, m.sum() >= MIN_PATIENTS]


def vecteurs_patients(codes_par_patient, noeuds, W, patients):
    V = np.zeros((len(patients), W.shape[1]))
    for k, pid in enumerate(patients):
        idx = [noeuds[c] for c in codes_par_patient.get(pid, ()) if c in noeuds]
        if idx:
            V[k] = W[idx].mean(0)
    return V


def distances_centroides(V, y, i_train):
    return np.column_stack([np.linalg.norm(V - V[i_train][y[i_train] == c].mean(0), axis=1)
                            for c in (0, 1)])


def cooccurrences(rollup_par_patient, noeuds, patients_train):
    M = np.zeros((len(patients_train), len(noeuds)), dtype=np.float32)
    for k, pid in enumerate(patients_train):
        for c in rollup_par_patient.get(pid, ()):
            M[k, noeuds[c]] = 1
    X = M.T @ M
    np.fill_diagonal(X, 0)
    return X


def modeles():
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                               random_state=SEED, n_jobs=-1),
        "GradBoost": HistGradientBoostingClassifier(random_state=SEED),
    }


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main(dossier_mimic=DOSSIER_MIMIC, dossier_athena=DOSSIER_ATHENA):
    rng = np.random.default_rng(SEED)

    def localiser(d, f):
        """Accepte le dossier exact ou un dossier parent : cherche le fichier dans les sous-dossiers."""
        if (Path(d) / f).exists():
            return Path(d)
        trouves = sorted(Path(d).rglob(f)) if Path(d).exists() else []
        if not trouves:
            raise SystemExit(f"{f} introuvable dans {d} : corrige le chemin en haut du script.")
        print(f"    {f} trouvé dans : {trouves[0].parent}")
        return trouves[0].parent

    dossier_athena = localiser(dossier_athena, "CONCEPT_ANCESTOR.csv")
    dossier_mimic = localiser(dossier_mimic, "condition_occurrence.csv")

    snomed = charger_snomed(dossier_athena)
    racine_id = trouver_concept(snomed, RACINE)
    cible_ids = {}
    for nom in CIBLE_NOMS:
        cid = trouver_concept(snomed, nom, obligatoire=False)
        if cid is None:
            print(f"    (concept cible '{nom}' absent du vocabulaire : ignoré)")
        else:
            cible_ids[cid] = nom
    if not cible_ids:
        raise SystemExit("Aucun concept cible trouvé : vérifie CIBLE_NOMS.")
    print(f"    racine : {RACINE} ({racine_id}) | cible : {list(cible_ids.values())}")

    cond, person = charger_diagnostics(dossier_mimic, snomed, dossier_athena)
    concepts_patients = cond["condition_concept_id"].unique()

    anc, profondeur, liens = lire_ancetres(dossier_athena, concepts_patients, racine_id,
                                           snomed.index.to_numpy())
    noeuds, adj, rep, ancetres = construire_ontologie(concepts_patients, anc, profondeur, liens, snomed)

    # Étape 5 : cible = le concept cible et tous ses descendants
    anc_par_concept = anc.groupby("descendant_concept_id")["ancestor_concept_id"].apply(set).to_dict()
    par_hierarchie = {p for p in concepts_patients
                      if p in cible_ids or anc_par_concept.get(p, set()) & cible_ids.keys()}
    noms_patients = snomed["concept_name"].reindex(concepts_patients).fillna("").str.lower()
    par_mot = {p for p, n in zip(concepts_patients, noms_patients)
               if any(m in n for m in MOTS_CIBLE) and not any(x in n for x in MOTS_EXCLUS)} - par_hierarchie
    concepts_cible = par_hierarchie | par_mot
    cond = cond.assign(est_cible=cond["condition_concept_id"].isin(concepts_cible))
    nom = lambda ids: snomed["concept_name"].reindex(list(ids)).dropna().tolist()
    print(f"[5] Cible : {len(concepts_cible)} concepts présents chez les patients")
    print(f"    par la hiérarchie ({len(par_hierarchie)}) : {nom(par_hierarchie)[:8]}")
    print(f"    par mot-clé seulement ({len(par_mot)}) : {nom(par_mot)[:8]}")
    print(f"    -> {cond.loc[cond['est_cible'], 'person_id'].nunique()} patients concernés")

    patients = np.array(sorted(cond["person_id"].unique()))
    y = cond.groupby("person_id")["est_cible"].any().reindex(patients).astype(int).to_numpy()

    # Anti-fuite par le mapping : OMOP peut mapper UN code source vers PLUSIEURS concepts.
    # Ex. ICD-10 E11.65 "diabète de type 2 avec hyperglycémie" -> "Type 2 diabetes mellitus"
    # (retiré, c'est la cible) + "Hyperglycemia" (resterait dans les variables = la réponse).
    # On retire donc toutes les lignes dont le code source a produit un concept de la cible.
    sources_cible = set(cond.loc[cond["est_cible"], "condition_source_value"])
    co_mappe = ~cond["est_cible"] & cond["condition_source_value"].isin(sources_cible)
    noms_co = snomed["concept_name"].reindex(cond.loc[co_mappe, "condition_concept_id"].unique()).dropna().tolist()
    print(f"    Anti-fuite mapping : {co_mappe.sum()} lignes retirées (même code source qu'un diagnostic "
          f"de la cible), concepts : {noms_co[:8]}")
    feat = cond[~cond["est_cible"] & ~co_mappe]
    if COUPURE_TEMPORELLE:
        date = pd.to_datetime(cond["condition_start_date"])
        premiere = date[cond["est_cible"]].groupby(cond.loc[cond["est_cible"], "person_id"]).min()
        limite = feat["person_id"].map(premiere)
        feat = feat[limite.isna() | (pd.to_datetime(feat["condition_start_date"]) < limite)]
        vides = sum(1 for pid in premiere.index if pid not in set(feat["person_id"]))
        print(f"    Coupure temporelle : {vides}/{len(premiere)} malades n'ont aucun diagnostic antérieur")

    codes = {}           # concepts "feuilles" de chaque patient (dans le graphe tronqué)
    for pid, c in zip(feat["person_id"], feat["condition_concept_id"]):
        codes.setdefault(pid, set()).update(rep.get(c, ()))
    rollup = {pid: set(cs) | set().union(*(ancetres.get(c, set()) for c in cs)) - {racine_id}
              for pid, cs in codes.items()}

    # Démographie
    p = person.set_index("person_id").reindex(patients)
    annee = pd.to_datetime(cond["condition_start_date"]).dt.year.groupby(cond["person_id"]).min()
    demo = pd.DataFrame(index=patients)
    if "year_of_birth" in p:
        demo["age"] = (annee.reindex(patients) - p["year_of_birth"]).to_numpy()
    demo["homme"] = (p["gender_concept_id"] == 8507).astype(int).to_numpy()
    demo = demo.fillna(demo.median())

    A = multi_hot(codes, patients)
    B1 = multi_hot(rollup, patients)
    X_A = np.hstack([A.to_numpy(), demo.to_numpy()])
    X_B1 = np.hstack([B1.to_numpy(), demo.to_numpy()])
    print(f"[6] Patients : {len(patients)} | prévalence : {y.mean():.2f} | "
          f"variables A : {A.shape[1]} | variables B1 : {B1.shape[1]}")

    # Contrôle de fuite : les variables les plus liées à la cible doivent être des
    # comorbidités plausibles (hypertension, insuffisance rénale...), pas la maladie elle-même.
    if A.shape[1]:
        malades, autres = A.to_numpy()[y == 1], A.to_numpy()[y == 0]
        ecart = pd.Series(malades.mean(0) - autres.mean(0), index=A.columns).sort_values(ascending=False)
        print("    Variables les plus associées à la cible (à vérifier : aucune ne doit être la maladie) :")
        for c in ecart.index[:10]:
            j = A.columns.get_loc(c)
            print(f"      {snomed['concept_name'].get(c, c)[:60]:60s} "
                  f"{malades[:, j].mean():4.0%} des malades vs {autres[:, j].mean():4.0%} des autres")
    if y.sum() < CV["n_splits"]:
        raise SystemExit("Trop peu de cas positifs pour une validation croisée.")

    t = time.time()
    print("    Entraînement node2vec...", flush=True)
    W_n2v = node2vec(adj, DIM, rng=rng, **N2V)
    print(f"    node2vec terminé ({time.time() - t:.0f} s)", flush=True)
    W_rand = rng.normal(size=W_n2v.shape)
    V_n2v = vecteurs_patients(codes, noeuds, W_n2v, patients)
    V_rand = vecteurs_patients(codes, noeuds, W_rand, patients)

    print("[7] Évaluation", flush=True)
    cv = RepeatedStratifiedKFold(random_state=SEED, **CV)
    resultats = []
    for split, (itr, ite) in enumerate(cv.split(X_A, y)):
        W_keep = keep(cooccurrences(rollup, noeuds, patients[itr]), W_n2v, **KEEP)
        V_keep = vecteurs_patients(codes, noeuds, W_keep, patients)
        variantes = {
            "A  baseline": X_A,
            "B1 roll-up": X_B1,
            "B2 node2vec EmbedAug": np.hstack([X_A, V_n2v]),
            "B2 node2vec DistAug": np.hstack([X_A, distances_centroides(V_n2v, y, itr)]),
            "B3 KEEP EmbedAug": np.hstack([X_A, V_keep]),
            "B3 KEEP DistAug": np.hstack([X_A, distances_centroides(V_keep, y, itr)]),
            "Ctrl aléatoire EmbedAug": np.hstack([X_A, V_rand]),
        }
        for nom_var, X in variantes.items():
            for nom_mod, m in modeles().items():
                m.fit(X[itr], y[itr])
                proba = m.predict_proba(X[ite])[:, 1]
                resultats.append(dict(variante=nom_var, modele=nom_mod, split=split,
                                      AUPRC=average_precision_score(y[ite], proba),
                                      AUC=roc_auc_score(y[ite], proba)))
        print(f"  split {split + 1}/{CV['n_splits'] * CV['n_repeats']} terminé", flush=True)

    res = pd.DataFrame(resultats)
    res.to_csv("resultats_snomed.csv", index=False)
    print(f"\nAUPRC de référence (classifieur au hasard) = {y.mean():.3f}")
    print(res.groupby(["variante", "modele"])[["AUPRC", "AUC"]].agg(["mean", "std"]).round(3).to_string())
    print()
    comparer(res)
    return res


def comparer(res, metrique="AUPRC", fichier_sortie="comparaisons_snomed.csv"):
    """Différences appariées split par split : Wilcoxon + t de Nadeau-Bengio, Holm par modèle."""
    from scipy import stats

    k = res["split"].nunique()
    ratio = 1 / (CV["n_splits"] - 1)
    if k < 6:
        print(f"Attention : {k} splits seulement, mets n_repeats=5 pour une vraie comparaison.\n")
    references = {"A  baseline": "vs baseline", "Ctrl aléatoire EmbedAug": "vs aléatoire"}
    lignes = []
    for modele, sous in res.groupby("modele"):
        piv = sous.pivot(index="split", columns="variante", values=metrique)
        for ref, nom_ref in references.items():
            for var in piv.columns:
                if var in (ref, "A  baseline") or (ref.startswith("Ctrl") and not var.startswith("B")):
                    continue
                d = (piv[var] - piv[ref]).to_numpy()
                p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                v = d.var(ddof=1)
                p_nb = 2 * stats.t.sf(abs(d.mean() / np.sqrt((1 / k + ratio) * v)), df=k - 1) if v > 0 else 1.0
                lignes.append(dict(modele=modele, comparaison=nom_ref, variante=var,
                                   diff_moy=d.mean(), splits_gagnes=f"{(d > 0).sum()}/{k}",
                                   p_wilcoxon=p_w, p_nadeau_bengio=p_nb))
    tab = pd.DataFrame(lignes)
    tab["p_NB_holm"] = np.nan
    for _, idx in tab.groupby("modele").groups.items():
        p = tab.loc[idx, "p_nadeau_bengio"].to_numpy()
        ordre = np.argsort(p)
        ajust = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[ordre]).clip(max=1)
        tab.loc[np.asarray(idx)[ordre], "p_NB_holm"] = ajust
    tab["signif"] = np.where(tab["p_NB_holm"] < 0.05, "*", "")
    print(f"Comparaisons appariées sur {metrique} ({k} splits). diff_moy > 0 = la variante fait mieux.")
    print(tab.round(3).to_string(index=False))
    tab.to_csv(fichier_sortie, index=False)
    return tab


if __name__ == "__main__":
    main()