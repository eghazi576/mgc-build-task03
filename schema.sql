-- Part 2 — minimal schema for the leads CRM dump (PostgreSQL dialect).
--
-- Design notes:
--   * One main table. The CSV is one fact per row (a lead) and almost every
--     column is an attribute of that lead, so splitting it apart would be
--     normalisation for its own sake. The one thing I do split out is
--     `source`, because it is a small controlled vocabulary that marketing
--     will want to rename/merge without rewriting 9,000 rows.
--   * lead_id is the natural primary key (already unique in the dump).
--   * crm_record_hash is the CRM's own fingerprint of the record contents.
--     The dump contains the same lead entered twice under different lead_ids
--     ("MGC-104974" and "MGC-104974-B") with identical hashes — so the
--     schema-level fix is a UNIQUE constraint on that hash. See queries.sql.

CREATE TABLE lead_sources (
    source_id   SMALLSERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE          -- 'Facebook Ads', 'Referral', ...
);

CREATE TABLE leads (
    lead_id                     TEXT PRIMARY KEY,          -- 'MGC-104067'
    created_at                  TIMESTAMP NOT NULL,
    source_id                   SMALLINT NOT NULL REFERENCES lead_sources,
    city                        TEXT,                      -- normalise case on ingest
    area                        TEXT,                      -- nullable in the dump
    property_type               TEXT NOT NULL,
    budget_pkr_lac              NUMERIC(8,1),              -- nullable
    bedrooms                    SMALLINT,                  -- nullable (plots/shops)
    first_response_minutes      NUMERIC(8,1),
    calls_made                  SMALLINT NOT NULL DEFAULT 0,
    total_call_seconds          INTEGER  NOT NULL DEFAULT 0,
    whatsapp_replies            SMALLINT NOT NULL DEFAULT 0,
    site_visits                 SMALLINT NOT NULL DEFAULT 0,
    agent_experience_years      NUMERIC(4,1),
    is_overseas                 BOOLEAN NOT NULL DEFAULT FALSE,
    referred_by_existing_client BOOLEAN NOT NULL DEFAULT FALSE,
    has_financing_approved      BOOLEAN NOT NULL DEFAULT FALSE,
    token_amount_received_pkr   NUMERIC(12,0) NOT NULL DEFAULT 0,
    crm_record_hash             BIGINT NOT NULL UNIQUE,    -- <-- blocks double entry
    converted                   BOOLEAN NOT NULL DEFAULT FALSE
);

-- The queries below filter/group on these:
CREATE INDEX idx_leads_source    ON leads (source_id);
CREATE INDEX idx_leads_converted ON leads (converted);
