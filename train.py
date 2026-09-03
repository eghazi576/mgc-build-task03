"""
Part 3 — baseline model: likelihood that a lead converts.

Data decisions (also summarised in the README):

DROPPED — identifiers / bookkeeping:
  * lead_id, crm_record_hash — identifiers, no signal.
  * created_at — kept out of the baseline for simplicity (month/seasonality
    could be engineered later).

DROPPED — leakage:
  * token_amount_received_pkr — a token is paid when the lead is already
    converting; in the dump every converted lead has token > 0 and almost
    no unconverted lead does. Using it would give a near-perfect score
    that predicts nothing about a NEW lead. This is the column the task
    is really asking about.

FIXED:
  * Duplicate entries (same crm_record_hash, "-B" lead_ids) — dropped one
    of each pair so the same lead is not counted twice.
  * city — normalised case and mapped abbreviations (ISB, Rwp, khi) to the
    full names; the dump has 'Islamabad', 'ISLAMABAD' and 'ISB' as three
    different values.
  * Missing bedrooms (~39%) — mostly plots/commercial where bedrooms make
    no sense; imputed 0 and let the model see property_type alongside.
  * Missing budget / first_response / agent_experience — median-imputed.

METRIC: PR-AUC (average precision).
  Only 6.9% of leads convert. With that imbalance, accuracy is useless
  (predicting "no" for everyone scores 93%). The business use is ranking
  which leads to call first, so the right lens is precision/recall on the
  positive class across thresholds — which is exactly what PR-AUC measures.
  Baseline to beat: 0.069 (a random ranker scores the positive rate).

Run:  python train.py          (writes model.pkl for the web app)
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent

CITY_FIX = {"isb": "Islamabad", "rwp": "Rawalpindi", "khi": "Karachi"}

CATEGORICAL = ["source", "city", "property_type"]
NUMERIC = [
    "budget_pkr_lac", "bedrooms", "first_response_minutes", "calls_made",
    "total_call_seconds", "whatsapp_replies", "site_visits",
    "agent_experience_years", "is_overseas", "referred_by_existing_client",
    "has_financing_approved",
]


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(HERE / "leads.csv")
    df = df.drop_duplicates(subset="crm_record_hash", keep="first")
    df["city"] = (
        df["city"].str.strip().str.lower().replace(CITY_FIX).str.title()
    )
    df["bedrooms"] = df["bedrooms"].fillna(0)  # plots/shops have none
    return df


def build_model() -> Pipeline:
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), NUMERIC),
    ])
    return Pipeline([
        ("prep", prep),
        ("clf", HistGradientBoostingClassifier(random_state=42)),
    ])


def main():
    df = load_clean()
    X, y = df[CATEGORICAL + NUMERIC], df["converted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    print(f"Rows after de-dup: {len(df)}  |  positive rate: {y.mean():.3f}")
    print(f"PR-AUC (average precision): {average_precision_score(y_test, scores):.3f}"
          f"   (random baseline = {y.mean():.3f})")
    print(f"ROC-AUC (for reference):    {roc_auc_score(y_test, scores):.3f}")

    with open(HERE / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved model.pkl")


if __name__ == "__main__":
    main()

