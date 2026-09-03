"""Automated tests for all four parts.

Deliberately needs no API key, no network and no trained model.pkl — the one
outbound call is stubbed and the classifier is replaced with a stand-in, so
this runs the same on a fresh clone as it does here.

Run:  python test_app.py
      python -m unittest -v
"""

import contextlib
import io
import unittest
from unittest import mock

import requests

import ask as ask_module
import app as web
import train


class FakeResponse:
    def __init__(self, text="answer", status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


class Proba:
    """Stands in for the numpy array sklearn returns; app.py does [0, 1]."""

    def __init__(self, p):
        self.p = p

    def __getitem__(self, key):
        return self.p


class FakeModel:
    """Stands in for the pickled Pipeline, and records what it was handed."""

    def __init__(self, p=0.3):
        self.p = p
        self.seen = None

    def predict_proba(self, row):
        self.seen = row.iloc[0].to_dict()
        return Proba(self.p)


# --------------------------------------------------------------------------
# Part 1 — document assistant
# --------------------------------------------------------------------------
class TestDocumentAssistant(unittest.TestCase):

    def test_loads_every_document_with_a_source_marker(self):
        docs = ask_module.load_documents()
        for name in ("01_mgc_aurora_heights_brochure.md",
                     "02_price_list_payment_plan.md",
                     "03_booking_policy_faq.md"):
            self.assertIn(f"===== DOCUMENT: {name} =====", docs)

    def test_question_and_documents_both_reach_the_model(self):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured.update(json)
            return FakeResponse("answer")

        with mock.patch.object(ask_module.requests, "post", fake_post):
            with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                out = ask_module.ask("What is the transfer fee?", "DOCS HERE")

        self.assertEqual(out, "answer")
        self.assertEqual(captured["temperature"], 0, "must stay deterministic")
        user_turn = captured["messages"][1]["content"]
        self.assertIn("DOCS HERE", user_turn)
        self.assertIn("What is the transfer fee?", user_turn)

    def test_system_prompt_keeps_the_grounding_rules(self):
        prompt = ask_module.SYSTEM_PROMPT
        for rule in ("GROUNDING", "SOURCES", "MISSING INFORMATION",
                     "CONFLICTS", "UNCONFIRMED"):
            self.assertIn(rule, prompt)

    def test_transient_server_errors_are_retried(self):
        replies = [FakeResponse(status_code=503),
                   FakeResponse(status_code=429),
                   FakeResponse("answer after retries")]
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append(1)
            return replies[len(calls) - 1]

        with mock.patch.object(ask_module.requests, "post", fake_post):
            with mock.patch.object(ask_module.time, "sleep") as slept:
                with mock.patch.dict("os.environ",
                                     {"GEMINI_API_KEY": "test-key"}):
                    with contextlib.redirect_stdout(io.StringIO()):
                        out = ask_module.ask("q", "docs")

        self.assertEqual(out, "answer after retries")
        self.assertEqual(len(calls), 3)
        self.assertEqual(slept.call_count, 2, "must back off between tries")

    def test_retries_eventually_give_up_and_raise(self):
        def always_down(url, headers, json, timeout):
            return FakeResponse(status_code=503)

        with mock.patch.object(ask_module.requests, "post", always_down):
            with mock.patch.object(ask_module.time, "sleep"):
                with mock.patch.dict("os.environ",
                                     {"GEMINI_API_KEY": "test-key"}):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(requests.HTTPError):
                            ask_module.ask("q", "docs")

    def test_a_real_error_is_not_retried(self):
        calls = []

        def unauthorised(url, headers, json, timeout):
            calls.append(1)
            return FakeResponse(status_code=401)

        with mock.patch.object(ask_module.requests, "post", unauthorised):
            with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "bad-key"}):
                with self.assertRaises(requests.HTTPError):
                    ask_module.ask("q", "docs")

        self.assertEqual(len(calls), 1, "401 will never fix itself")


# --------------------------------------------------------------------------
# Part 3 — data cleaning
# --------------------------------------------------------------------------
class TestDataCleaning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = train.load_clean()

    def test_city_abbreviations_and_case_are_normalised(self):
        for typed, expected in [("ISB", "Islamabad"), ("isb", "Islamabad"),
                                ("Rwp", "Rawalpindi"), ("khi", "Karachi"),
                                ("  LAHORE  ", "Lahore")]:
            self.assertEqual(train.normalise_city(typed), expected)

    def test_duplicate_crm_records_are_dropped(self):
        self.assertEqual(len(self.df), 9000)
        self.assertFalse(self.df["crm_record_hash"].duplicated().any())

    def test_cities_collapse_to_one_spelling_each(self):
        self.assertEqual(sorted(self.df["city"].unique()), [
            "Abbottabad", "Faisalabad", "Gujranwala", "Islamabad", "Karachi",
            "Lahore", "Multan", "Peshawar", "Rawalpindi"])

    def test_bedrooms_gaps_are_filled(self):
        self.assertFalse(self.df["bedrooms"].isna().any())

    def test_leakage_column_is_not_a_feature(self):
        features = train.CATEGORICAL + train.NUMERIC
        self.assertNotIn("token_amount_received_pkr", features)
        for identifier in ("lead_id", "crm_record_hash", "created_at"):
            self.assertNotIn(identifier, features)

    def test_every_feature_exists_in_the_dump(self):
        for column in train.CATEGORICAL + train.NUMERIC:
            self.assertIn(column, self.df.columns)


# --------------------------------------------------------------------------
# Part 4 — web page
# --------------------------------------------------------------------------
FORM = {"source": "Referral", "city": "Islamabad", "property_type": "Apartment",
        "budget_pkr_lac": "150", "bedrooms": "2",
        "first_response_minutes": "30", "calls_made": "1",
        "total_call_seconds": "120", "whatsapp_replies": "1",
        "site_visits": "0", "agent_experience_years": "3"}


class TestWebApp(unittest.TestCase):

    def setUp(self):
        self.client = web.app.test_client()
        self.original = web.MODEL
        self.model = FakeModel()
        web.MODEL = self.model

    def tearDown(self):
        web.MODEL = self.original

    def post(self, **overrides):
        return self.client.post("/score", data=dict(FORM, **overrides))

    def test_landing_page_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("MGC Sales Assistant", response.get_data(as_text=True))

    def test_untrained_model_explains_itself_instead_of_crashing(self):
        web.MODEL = None
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertIn("model.pkl not found", response.get_data(as_text=True))

    def test_score_is_rendered_as_a_percentage(self):
        response = self.post()
        self.assertIn("Conversion likelihood: 30.0%",
                      response.get_data(as_text=True))

    def test_checkboxes_become_one_and_zero(self):
        self.post(has_financing_approved="on")
        self.assertEqual(self.model.seen["has_financing_approved"], 1)
        self.assertEqual(self.model.seen["is_overseas"], 0)

    def test_every_feature_the_model_expects_is_sent(self):
        self.post()
        self.assertEqual(sorted(self.model.seen),
                         sorted(train.CATEGORICAL + train.NUMERIC))

    def test_blank_numeric_boxes_default_to_zero(self):
        self.post(budget_pkr_lac="", bedrooms="")
        self.assertEqual(self.model.seen["budget_pkr_lac"], 0.0)

    def test_scoring_normalises_city_exactly_like_training(self):
        # An unnormalised 'Isb' would be an unseen category that the encoder
        # silently zeroes, quietly degrading the score with no error.
        for typed in ("ISB", "isb", "islamabad", "  ISLAMABAD  "):
            self.post(city=typed)
            self.assertEqual(self.model.seen["city"], "Islamabad")
            self.assertEqual(self.model.seen["city"],
                             train.normalise_city(typed))

    def test_non_numeric_input_reports_the_field_instead_of_a_500(self):
        for field, junk in [("budget_pkr_lac", "1.5cr"),
                            ("budget_pkr_lac", "150,000"),
                            ("calls_made", "two"),
                            ("site_visits", "3.7")]:
            with self.subTest(field=field, value=junk):
                response = self.post(**{field: junk})
                self.assertEqual(response.status_code, 200)
                body = response.get_data(as_text=True)
                self.assertIn(field.replace("_", " "), body)
                self.assertIn("is not a valid number", body)

    def test_good_input_still_scores_after_a_bad_submission(self):
        self.post(budget_pkr_lac="abc")
        response = self.post()
        self.assertIn("Conversion likelihood: 30.0%",
                      response.get_data(as_text=True))

    def test_missing_api_key_is_reported_not_raised(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            response = self.client.post("/ask", data={"q": "anything"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("GEMINI_API_KEY is not set",
                      response.get_data(as_text=True))

    def test_api_failure_keeps_the_page_up(self):
        def boom(question, documents):
            raise RuntimeError("connection refused")

        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with mock.patch.object(web, "ask", boom):
                response = self.client.post("/ask", data={"q": "anything"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("Error calling the model",
                      response.get_data(as_text=True))

    def test_answer_is_rendered_into_the_page(self):
        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with mock.patch.object(web, "ask", lambda q, d: "The fee is 2%."):
                response = self.client.post("/ask", data={"q": "fee?"})

        self.assertIn("The fee is 2%.", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
