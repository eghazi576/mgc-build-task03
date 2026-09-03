"""
Part 4 — minimal web interface tying Parts 1 and 3 together.

One page, two forms:
  * ask the document assistant a question (Part 1) — needs GEMINI_API_KEY
  * enter a lead's details and get a conversion score (Part 3) —
    needs model.pkl (run `python train.py` once first)

Run:  python app.py   ->  http://localhost:5000
"""

import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, request

from ask import ask, load_documents

HERE = Path(__file__).parent
app = Flask(__name__)

DOCUMENTS = load_documents()
MODEL = None
if (HERE / "model.pkl").exists():
    with open(HERE / "model.pkl", "rb") as f:
        MODEL = pickle.load(f)

PAGE = """<!doctype html>
<title>MGC Sales Assistant</title>
<h1>MGC Sales Assistant</h1>

<h2>1 &middot; Ask the documents</h2>
<form method="post" action="/ask">
  <input name="q" size="70" placeholder="e.g. What's the transfer fee?" value="{q}">
  <button>Ask</button>
</form>
<pre style="white-space:pre-wrap">{answer}</pre>

<h2>2 &middot; Score a lead</h2>
<form method="post" action="/score">
  Source <select name="source">{source_opts}</select>
  City <input name="city" value="Islamabad" size="12">
  Property <select name="property_type">{ptype_opts}</select><br><br>
  Budget (lac) <input name="budget_pkr_lac" value="150" size="6">
  Bedrooms <input name="bedrooms" value="2" size="3">
  First response (min) <input name="first_response_minutes" value="30" size="5">
  Calls made <input name="calls_made" value="1" size="3"><br><br>
  Call seconds <input name="total_call_seconds" value="120" size="6">
  WhatsApp replies <input name="whatsapp_replies" value="1" size="3">
  Site visits <input name="site_visits" value="0" size="3">
  Agent exp (yrs) <input name="agent_experience_years" value="3" size="4"><br><br>
  <label><input type="checkbox" name="is_overseas"> Overseas</label>
  <label><input type="checkbox" name="referred_by_existing_client"> Referred</label>
  <label><input type="checkbox" name="has_financing_approved"> Financing approved</label>
  <button>Score</button>
</form>
<p><b>{score}</b></p>
"""

SOURCES = ["Facebook Ads", "Property Portal", "Google Search", "Instagram",
           "Referral", "Walk-in", "WhatsApp Campaign", "Expo Stall", "Billboard"]
PTYPES = ["Apartment", "Plot", "Villa", "Commercial Shop", "Penthouse", "Farmhouse"]

# The numeric half of the score form. Blank means 0; anything that isn't a
# number is reported back to the salesperson instead of raising a 500.
NUMERIC_FIELDS = {
    "budget_pkr_lac": float,
    "bedrooms": float,
    "first_response_minutes": float,
    "calls_made": int,
    "total_call_seconds": float,
    "whatsapp_replies": int,
    "site_visits": int,
    "agent_experience_years": float,
}


def render(q="", answer="", score=""):
    return PAGE.format(
        q=q, answer=answer, score=score,
        source_opts="".join(f"<option>{s}</option>" for s in SOURCES),
        ptype_opts="".join(f"<option>{p}</option>" for p in PTYPES),
    )


@app.route("/")
def index():
    return render()


@app.route("/ask", methods=["POST"])
def ask_route():
    q = request.form["q"].strip()
    if not q:
        return render()
    if not os.environ.get("GEMINI_API_KEY"):
        return render(q=q, answer="GEMINI_API_KEY is not set — see README.")
    try:
        answer = ask(q, DOCUMENTS)
    except Exception as e:  # keep the page up whatever the API does
        answer = f"Error calling the model: {e}"
    return render(q=q, answer=answer)


@app.route("/score", methods=["POST"])
def score_route():
    if MODEL is None:
        return render(score="model.pkl not found — run `python train.py` first.")
    f = request.form

    numbers = {}
    for field, cast in NUMERIC_FIELDS.items():
        raw = f[field].strip()
        try:
            numbers[field] = cast(raw) if raw else cast(0)
        except ValueError:
            return render(score=f"{field.replace('_', ' ')}: "
                                f"'{raw}' is not a valid number.")

    row = pd.DataFrame([{
        "source": f["source"],
        "city": f["city"].strip().title(),
        "property_type": f["property_type"],
        **numbers,
        "is_overseas": int("is_overseas" in f),
        "referred_by_existing_client": int("referred_by_existing_client" in f),
        "has_financing_approved": int("has_financing_approved" in f),
    }])
    p = MODEL.predict_proba(row)[0, 1]
    return render(score=f"Conversion likelihood: {p:.1%} "
                        f"(dataset average is 7.0%)")


if __name__ == "__main__":
    app.run(debug=False, port=5000)
