# Kochiyo Order Insights Agent

An agent that cleans a messy restaurant order export, computes operator-facing
KPIs, and answers natural-language questions about the data — including
questions it wasn't specifically built for.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # then add a free Groq API key, or set up Ollama — see below

python run_kpis.py          # prints the KPI report, writes data/cleaned_orders.csv + data/kpi_report.json
python chat.py               # interactive natural-language Q&A
streamlit run app.py         # optional web UI (dashboard + chat)
python tests/test_pipeline.py   # offline test suite — no LLM/network needed
```

No paid services anywhere. See **"LLM backend (free only)"** below.

## Architecture

```
data/kochiyo_orders_export.csv   raw export
        │
src/clean.py                     deterministic cleaning (pandas/regex) → cleaned DataFrame
        │
src/kpis.py                      deterministic KPI aggregation → dashboard numbers
        │
src/tools.py                     same DataFrame exposed as callable "tools" + JSON schemas
        │
src/agent.py                     LLM tool-calling loop + suspicious-result retry logic
        │
src/llm_backend.py               swappable free LLM client (Groq cloud / local Ollama)
```

**Why cleaning and KPIs are NOT done by the LLM:** an operator needs the same
revenue number every time they run the report. Parsing/aggregation is 100%
deterministic pandas code — no hallucination risk, no cost, no latency. The
LLM is reserved for the one part of this task that genuinely needs
flexibility: understanding an arbitrary natural-language question and
mapping it to the right tool call(s). This also means `run_kpis.py` works
with **zero LLM setup at all** — only `chat.py`/`app.py`'s Q&A needs an LLM
key.

**Agent design:** single agent, OpenAI-compatible tool-calling loop (works
against both Groq and Ollama unchanged). Seven tools cover sales summary,
top items, peak hours, per-item stats, status/cancellation breakdown, and
breakdowns by store/payment method — general enough to compose for questions
I didn't anticipate, without needing a giant single "do everything" tool.

## The retry-on-bad-result behavior (per the assignment hint)

This was implemented as **explicit code in `agent.py`**, not left to the LLM's
own judgment — LLMs don't reliably self-correct on their own, and an
interview deliverable shouldn't depend on that being reliable.

Every tool in `tools.py` returns a `matched_rows` field. After each tool
call, the agent loop checks: *did this return 0 rows, and did the call use a
guessed `year`/`month`?* If so, it calls `list_available_period()` (which
reports the years/months that actually exist in the data), corrects the
guessed year, and **retries once automatically** before the result ever
reaches the final answer. Concretely, for "best-selling item in June" with
no year stated, if the model guesses `year=2025` (data is actually all
`2026`), the flow is:

```
get_top_items(year=2025, month=6)  -> matched_rows: 0   (suspicious)
list_available_period()            -> years_present: [2026]
get_top_items(year=2026, month=6)  -> matched_rows: 282, Tonkotsu Ramen on top   (used)
```

Verified in `tests/test_pipeline.py` without needing a live LLM call.

## Data issues found in the CSV (and how they were handled)

The file (1091 raw rows) is messy in several independent ways:

1. **Five different datetime formats**, from what looks like two upstream
   systems: `"Jul 19, 2026 13:56"`, `"07/28/2026 01:25 PM"`,
   `"2026/07/16"`, `"22/06/2026 19:24"`, `"2026-06-27 12:57"`.
   The two slash-separated formats are the trap: one is `MM/DD/YYYY` (always
   carries AM/PM), the other is `DD/MM/YYYY` in 24-hour time (never carries
   AM/PM). They're **only** distinguishable by the presence of AM/PM, not by
   which number is bigger — plenty of rows have day ≤ 12 in both buckets, so
   a naive "guess by magnitude" parser silently mis-parses a chunk of June
   rows into other months. Handled with format-specific regex routing
   (`clean.py`), confirmed correct by checking the out-of-range cases (e.g.
   `11/07/2026` must be 11 July, since there's no 11th month... this dataset
   only spans June–July anyway, confirmed via `list_available_period`).

2. **231 rows are date-only** (`"2026/07/16"`, no clock time). These are
   fine for daily/monthly revenue but would corrupt an hourly breakdown by
   defaulting to midnight — the very first KPI run produced a fake "122
   orders at 00:00" spike. Fixed by tracking a `has_time_component` flag and
   excluding those rows specifically from the peak-hours KPI (kept
   everywhere else).

3. **Yen amounts formatted three ways**: `¥987`, `502.00`, `450`, and
   `¥1,443` (comma thousands separator). Stripped symbol/commas, cast to
   float.

4. **Item names badly inconsistent**: casing, doubled internal spaces
   (`"Beef  Curry  Udon"`), and `"gyoza(6pc)"` / `"Gyoza (6pcs)"` /
   `"Gyoza 6pcs"` all being the same dish. Also `"Kochiyo Chicken Biryani"`
   vs `"Chicken Biryani"` — confirmed these are the *same* SKU (not a
   different, pricier "Kochiyo-brand" item) by comparing average unit price:
   ¥1347.86 vs ¥1347.41 across ~50 rows each — close enough to be the same
   dish with an inconsistent export label, so merged.
   **Assumption kept the other way:** `"Chicken Karaage"` vs
   `"Chicken Karaage Box"` were **kept separate** even though price ranges
   overlap, since "Box" plausibly denotes a different meal-set SKU on the
   menu and merging two differently-named menu items is a bigger risk than
   a slightly finer item breakdown. Noted here as a judgment call, not a
   certainty.

5. **Statuses/payment methods/store names inconsistently cased/spaced**
   (`"DELIVERED"`, `"Delivered "`, `"delivered"`; `"Kochiyo Halal - Edogawa "`
   with a trailing space vs without). Normalized to a canonical form.

6. **~53 rows have a blank `order_id`.** These are still real line items
   (every other field is populated) — dropping them would understate sales,
   so they're kept and given a synthetic `row_id` instead. `order_id` is
   therefore *not* reliable for grouping into parent orders in this
   dataset; each row is treated as an independent line item.

7. **Duplicate `order_id` values are mostly legitimate**, not export bugs —
   they're multi-item orders (one order, several dishes, same `order_id`
   repeated). Verified by checking a duplicated id: same timestamp, same
   store, different items. **Not deduplicated.**

8. **16 groups of fully-duplicated rows** (every field identical, including
   `order_id`) — these *are* export artifacts (the same line item exported
   twice) and were dropped, along with 5 completely blank rows.

9. **~20 rows are missing quantity** or another field needed for a reliable
   dollar figure. Excluded from the cleaned/KPI dataset rather than
   silently treated as 0 or 1 — counted and reported in the cleaning report
   (`report["rows_excluded_missing_critical_fields"]`) so nothing vanishes
   invisibly.

**What counts as "sales":** total revenue = `unit_price × quantity` summed
over rows with `status == "Delivered"` only. Cancelled and Refunded orders
are excluded from revenue (the business didn't keep that money) but are
still surfaced separately — cancellation rate (30.4%) and refund rate
(14.2%) are numbers an operator would want flagged, not buried.

## LLM backend (free only)

Both options are genuinely free — no trial credits, no card on file:

- **Groq** (default) — free cloud API, generous rate limits, fast. Sign up
  at console.groq.com, create a key, put it in `.env` as `GROQ_API_KEY`.
- **Ollama** — fully local, zero signup anywhere. `ollama pull qwen2.5:7b`,
  `ollama serve`, set `LLM_PROVIDER=ollama` in `.env`.

Both speak the same OpenAI-compatible tool-calling API, so `agent.py` and
`tools.py` don't change between them — only `llm_backend.py`'s base URL/key
does.

## Example questions it handles

- "What was our total revenue?" / "...in June?" / "...in July?"
- "What was our best-selling item in June?" (including the wrong-year-guess
  retry case above)
- "What time of day do we get the most orders?"
- "How does Tonkotsu Ramen do?" / "What's our rating on the salmon sushi
  set?"
- "What's our cancellation rate?"
- "Which store makes the most money?"
- "How much revenue comes from PayPay vs cash?"
- A question not specifically designed for: e.g. "Is LINE Pay our least
  popular payment method?" — no dedicated tool for "least popular," but the
  agent composes `get_breakdown_by(dimension="payment_method")` and reads
  off the bottom of the sorted list itself.

## Hosting

`app.py` is a polished Streamlit front end — branded dashboard (KPI cards +
charts) plus a real chat interface (message bubbles, clickable example
questions, expandable tool-call traces) — ready to push live for the bonus:

```bash
streamlit run app.py     # run it locally first at http://localhost:8501
```

**Streamlit Community Cloud (free):**
1. Push this repo to GitHub.
2. Go to share.streamlit.io → "New app" → pick the repo → main file `app.py`.
3. In the app's Settings → Secrets, add `GROQ_API_KEY = "your_key"` (same
   key from your `.env`, added as a secret, never committed).
4. Deploy — you get a public `*.streamlit.app` URL.

**Hugging Face Spaces (free CPU tier)** works the same way: create a Space
with the "Streamlit" SDK, push this repo's contents to it, and add
`GROQ_API_KEY` under Settings → Repository secrets.


## What I'd do with more time

- Fuzzy-match item names with a real similarity metric instead of
  substring/regex rules, for menu items that get renamed more aggressively
  in future exports.
- A confidence check on the ambiguous-date bucket (flag the file for human
  review if a slash-format date ever has day ≤ 12 in *both* buckets in a way
  that isn't resolvable, rather than assuming this file's pattern always
  holds).
- Multi-turn conversation memory in `chat.py` beyond a single question.
