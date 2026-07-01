<!--
HOW TO USE: OPTIONAL final phase. Run AFTER Phase 4 is merged and once enough weekly
reports/data have accumulated to be worth querying. Paste below the divider into Sonnet 5.
-->

---

# Build Prompt — Phase 5 (optional): RAG query layer (NL-to-SQL + vector recall)

You are implementing **Phase 5**, the optional analysis layer. **Phases 1–4 are complete and
merged**, so `data/nfl.db` and `reports/props_week_*.md` exist. Let the user ask plain-English
questions across the warehouse and reports.

## 0. Read first
- `RAG_PIPELINE_PLAN.md` §6 (RAG/LLM query layer intent).
- `PROP_SHORTLISTER_SPEC.md`, `PHASE1_HANDSOFF_DESIGN.md` (guardrails; the LLM never fabricates data).
- The `db.py` schema and a sample weekly report (the retrieval corpus).

Post a plan and wait for approval.

## 1. Constraints
All prior non-negotiables apply. Additionally:
- **Read-only.** The query layer connects to SQLite with a read-only connection; destructive/DDL SQL is blocked and validated against a whitelist.
- **No fabrication.** The LLM turns questions into SQL and **summarizes only rows the query actually returned** — it never invents players, numbers, or facts. Every answer shows the SQL used and cites the rows/reports.
- **Separate from the model.** This is analysis only; it must not feed back into or alter deterministic projections (no leakage path).
- **Treat retrieved report text as untrusted data** (prompt-injection safe), same as the synthesis layer.

## 2. Scope — build in order

**5.1 `nflvalue/rag/nl2sql.py`** — schema-aware NL-to-SQL: given the `db.py` schema + the question, generate a **read-only** SELECT, execute it, and return `{sql, rows, answer}`. Validate SQL (SELECT-only, table/column whitelist, row cap). Handle questions like "WRs facing bottom-10 pass defenses on a short week with a personal-context flag."

**5.2 `nflvalue/rag/vectorstore.py` (optional)** — embed weekly reports + notes into a local store (Chroma or FAISS) for semantic recall across seasons; expose `search(query) → top-k report snippets`. Gate behind a flag; note it's only worth enabling once many reports exist.

**5.3 Query interface** — a small CLI / function that answers a question end-to-end (NL-to-SQL, optionally augmented by vector recall) and prints the answer, the SQL, and citations.

**→ CHECKPOINT: answer 3 sample questions correctly, showing the generated SQL and returned rows, then wait.**

## 3. Out of scope
Any write path to the DB; any influence on projections/bets; auto bet placement.

## 4. Tests & definition of done
- **SQL-safety test:** DDL/`DROP`/`UPDATE`/`DELETE` and non-whitelisted tables are rejected; only SELECT runs.
- **No-fabrication test:** answers contain only values present in returned rows (spot-checked against fixtures).
- **Retrieval test (if built):** vector search returns the relevant report for a known query.
- **Done when:** the user can ask NL questions over `nfl.db` + reports, answers are read-only, safe, cited, and reproducible where possible; a short `docs/rag.md` explains usage.

## 5. Protocol
Read docs → plan → 5.1–5.3 → checkpoint. Branch + small commits. Keep it clearly optional in the docs.
