-- Part 2 — queries. Written against the flat CSV loaded as a table `leads`
-- (i.e. `source` still a text column), so they run directly on the dump
-- with e.g. DuckDB:  duckdb -c "CREATE TABLE leads AS FROM 'leads.csv'; ..."

---------------------------------------------------------------------------
-- Query 1: conversion rate by lead source, best first,
--          only sources with at least 200 leads.
---------------------------------------------------------------------------
SELECT
    source,
    COUNT(*)                                        AS total_leads,
    SUM(converted)                                  AS conversions,
    ROUND(100.0 * SUM(converted) / COUNT(*), 2)     AS conversion_rate_pct
FROM leads
GROUP BY source
HAVING COUNT(*) >= 200
ORDER BY conversion_rate_pct DESC;

---------------------------------------------------------------------------
-- Query 2: duplicate leads — the same lead entered twice by different
--          agents under different lead_ids.
--
-- How they show up in this dump: the second entry reuses the original id
-- with a "-B" suffix (MGC-104974 vs MGC-104974-B), and both rows carry the
-- SAME crm_record_hash — the CRM's fingerprint of the record contents.
-- So the hash is the reliable join key; the id suffix is just the symptom.
---------------------------------------------------------------------------
SELECT
    a.lead_id   AS original_lead_id,
    b.lead_id   AS duplicate_lead_id,
    a.created_at,
    a.source,
    a.city,
    a.crm_record_hash
FROM leads a
JOIN leads b
  ON  a.crm_record_hash = b.crm_record_hash
  AND a.lead_id < b.lead_id            -- each pair once, no self-match
ORDER BY a.crm_record_hash;

-- Preventing this at the schema level (see schema.sql):
--   1. UNIQUE (crm_record_hash) on the leads table — the second INSERT of
--      an identical record fails instead of silently creating a twin.
--   2. Longer term, the real fix is a unique NATURAL key for the person:
--      the dump has no phone/CNIC column, but the production CRM should
--      enforce UNIQUE (phone_number) — the hash only catches entries that
--      are byte-for-byte identical, not re-typed variants.
