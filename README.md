# MGC Build Task — AI Developer & Engineer

Four parts of one problem: a grounded document assistant for the sales team,
a sane home for the leads data, a lead-scoring baseline, and a page that ties
them together.

## How to run

```bash
pip install -r requirements.txt

# Part 1 — document assistant (needs a free key from aistudio.google.com/apikey)
export GEMINI_API_KEY=AIza...
python ask.py "What's the transfer fee?"
python ask.py                      # interactive mode

# Part 2 — schema.sql + queries.sql (any SQL engine; I verified on SQLite/DuckDB)

# Part 3 — train the baseline, writes model.pkl
python train.py

# Part 4 — web page at http://localhost:5000
python app.py
```

Windows: use `set GEMINI_API_KEY=AIza...` instead of `export`.

## Part 1 — design choices

No vector database. Three short documents fit whole in the model's context,
so I inject all of them with source markers and let a strict system prompt do
the work: answer only from the documents, cite the source, surface conflicts
instead of resolving them silently, and refuse when the answer isn't there.
With dozens of documents I'd put chunked embedding retrieval in front of the
same prompt. No SDK either — Gemini is one OpenAI-compatible POST via
`requests`, so there's nothing hidden.

The five test questions from the brief behave as required, including:
transfer fee → both figures (2% price list vs 2.5% booking policy) with both
sources and "confirm before quoting"; rental yield → not in the documents,
refer to marketing manager; anchor tenant → explicitly unconfirmed.

## Part 2 — schema notes

One main `leads` table plus a `lead_sources` lookup (small controlled
vocabulary marketing will want to edit without touching 9,000 rows).
`lead_id` is the natural primary key. The dump contains 160 pairs of the same
lead entered twice under different ids (`MGC-104974` / `MGC-104974-B`) with
identical `crm_record_hash` — so the schema puts a `UNIQUE` constraint on the
hash, and the longer-term fix noted in `queries.sql` is a unique natural key
(phone/CNIC) in the production CRM, since a hash only catches byte-identical
re-entries.

## Part 3 — data decisions and metric

Dropped as leakage: **`token_amount_received_pkr`** — a token is paid when
the lead is already converting (every converted lead has token > 0, almost no
unconverted lead does). Using it scores near-perfect and predicts nothing
about a new lead. Dropped as noise: `lead_id`, `crm_record_hash` (identifiers),
`created_at` (kept out of the baseline; month/seasonality is a possible
later feature) and `area` — 10 values plus 470 blanks, spread near-uniformly
across all 9 cities, with conversion by area running 0.058–0.095 against a
0.070 base rate, so it is noise at this sample size rather than a sub-region
of `city`. Fixed: the 160 duplicate rows (dropped one of each pair),
city case/abbreviation mess (`ISLAMABAD`/`ISB` → `Islamabad`), missing
`bedrooms` (~39%, mostly plots/shops — imputed 0 with `property_type` present
so the model can tell them apart), and median-imputed the other numeric gaps.

Model: `HistGradientBoostingClassifier` in a sklearn pipeline, 80/20
stratified split.

**Metric: PR-AUC = 0.297** (random baseline = 0.070; ROC-AUC 0.787 for
reference). Only 6.9% of leads convert, so accuracy is meaningless — predict
"no" for everyone and score 93%. The business use is ranking which leads to
call first, so precision/recall on the positive class across thresholds is
the honest lens. 0.297 is a modest, believable baseline — roughly 4× better
than random ranking — not a suspicious 0.99.

## Part 4

One Flask page (`app.py`), both features: ask the documents (wired to Part 1)
and score a lead (wired to Part 3's `model.pkl`). No styling, per the brief.
The page stays up if the API key is missing, the model file hasn't been
trained yet, the API call fails, or a numeric box gets something that isn't a
number — each case tells you what to do instead of crashing.

Scoring imports `normalise_city` from `train.py` rather than repeating the
mapping, so a lead typed in as `ISB` is normalised to `Islamabad` exactly as
the training rows were. Duplicating that logic is how train/serve skew starts:
the encoder would silently zero the unseen `Isb` category and quietly return a
worse score with no error.

## Known gaps / what I'd do next

- No automated tests; the five brief questions were verified manually.
- `train.py` has no hyperparameter tuning (deliberate — brief said baseline).
- The assistant answers one question at a time; no conversation memory.
- Next: log every question the assistant can't answer — that list is the
  cheapest possible roadmap for what to add to the documents.
