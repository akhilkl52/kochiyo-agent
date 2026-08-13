"""
clean.py
--------
Turns the raw Kochiyo order export into a tidy DataFrame the KPI engine
and agent tools can trust.

Design principle: cleaning is 100% deterministic (pandas/regex), NOT
LLM-based. An LLM is non-deterministic and slow for this; a restaurant
operator needs the same KPI numbers every time they run the tool. The
LLM is reserved for the Q&A layer, where flexibility matters more than
determinism.

Every cleaning decision below was made by actually inspecting the CSV
(see README "Data issues found" section for the full write-up), not
guessed generically.
"""

from __future__ import annotations
import re
import pandas as pd
from datetime import datetime
from pathlib import Path

RAW_COLUMNS = [
    "order_id", "order_datetime", "store", "item_name",
    "quantity", "unit_price", "payment_method", "status", "customer_rating",
]

# ---------------------------------------------------------------------------
# 1. DATETIME PARSING
# ---------------------------------------------------------------------------
# The export contains FIVE distinct datetime formats, coming from what looks
# like two different upstream systems:
#   - "Jul 19, 2026 13:56"            -> month-name, 24h
#   - "07/28/2026 01:25 PM"           -> slash, 12h + AM/PM  => this bucket
#                                         is unambiguously MM/DD/YYYY because
#                                         it always carries AM/PM
#   - "2026/07/16"                    -> ISO-ish, date only, no time
#   - "22/06/2026 19:24"              -> slash, 24h, NO AM/PM => this bucket
#                                         is DAY-FIRST (DD/MM/YYYY). Proven by
#                                         checking rows where day > 12 (e.g.
#                                         "11/07/2026 18:26" must be
#                                         11-July, not the 7th month's
#                                         11th-day-doesn't-exist reading of
#                                         Nov 7th) -- every 24h-no-AM/PM slash
#                                         date in the file is consistent with
#                                         day-first once you check the
#                                         out-of-range cases.
#   - "2026-06-27 12:57"              -> ISO with dashes, 24h
#
# IMPORTANT: the two slash+time formats (12h-with-AM/PM vs 24h-no-AM/PM) are
# only distinguishable by the PRESENCE of "AM"/"PM" -- not by day/month
# magnitude alone, since plenty of rows have day<=12 in both buckets. Get the
# bucketing wrong and June 6th quietly becomes June... 6th, no error thrown,
# just silently wrong for the ambiguous rows. This is the single easiest bug
# to ship in this dataset.
_FORMATS_ORDERED = [
    (re.compile(r"^[A-Za-z]{3} \d{1,2}, \d{4} \d{1,2}:\d{2}$"), "%b %d, %Y %H:%M"),
    (re.compile(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} ?[APap][Mm]$"), "%m/%d/%Y %I:%M %p"),
    (re.compile(r"^\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{2}$"), "%Y-%m-%d %H:%M"),
    (re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$"), "%Y/%m/%d"),
    (re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"), "%Y-%m-%d"),
    (re.compile(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$"), "%d/%m/%Y %H:%M"),  # day-first, see note above
]


_DATE_ONLY_FORMATS = {"%Y/%m/%d", "%Y-%m-%d"}  # formats with no time-of-day info


def parse_datetime(raw: str):
    if not isinstance(raw, str) or not raw.strip():
        return pd.NaT, False
    s = raw.strip()
    for pattern, fmt in _FORMATS_ORDERED:
        if pattern.match(s):
            try:
                ts = pd.Timestamp(datetime.strptime(s, fmt))
                has_time = fmt not in _DATE_ONLY_FORMATS
                return ts, has_time
            except ValueError:
                continue
    return pd.NaT, False  # unparseable -> flagged, not guessed


# ---------------------------------------------------------------------------
# 2. PRICE PARSING
# ---------------------------------------------------------------------------
# Formats seen: "¥987", "502.00", "450", "¥1,443" (yen sign + thousands comma).
# All values are yen; strip symbol/commas and cast to float. No FX conversion
# needed since every row is the same currency once cleaned.
def parse_price(raw) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    s = s.replace("¥", "").replace(",", "").strip()
    try:
        val = float(s)
    except ValueError:
        return None
    return val if val >= 0 else None  # negative prices are bad data, not a discount signal here


# ---------------------------------------------------------------------------
# 3. ITEM NAME CANONICALIZATION
# ---------------------------------------------------------------------------
# Same dish appears under many spellings: extra internal spaces, mixed case,
# "(6pcs)" vs "6pc" vs "(6pcs)" vs "6pcs", and a "Kochiyo " brand prefix on
# some biryani rows. Verified with price comparison (see README) that
# "Kochiyo Chicken Biryani" and "Chicken Biryani" have IDENTICAL average unit
# price (~¥1348) across ~50 rows each -> same SKU, just inconsistent export
# labeling -> merged.
#
# Deliberately KEPT SEPARATE: "Chicken Karaage" vs "Chicken Karaage Box".
# Prices overlap but a "Box" is plausibly a different meal-set SKU on the
# menu, and merging two differently-named menu items is a bigger risk than
# leaving a slightly finer-grained item breakdown. Noted as an assumption.
def normalize_item_name(raw: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.lower().strip()
    s = re.sub(r"[()]", " ", s)          # drop parens
    s = re.sub(r"\bkochiyo\b", " ", s)   # drop stray brand prefix
    s = re.sub(r"(\d)pcs?\b", r"\1pc", s)  # "6pcs"/"6pc" -> "6pc"
    s = re.sub(r"\s+", " ", s).strip()
    # Title-case for display, but keep the pc-count lowercase/tight
    return s.title().replace("Pc", "pc")


# ---------------------------------------------------------------------------
# 4. STATUS NORMALIZATION
# ---------------------------------------------------------------------------
def normalize_status(raw: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip().capitalize()  # "DELIVERED "/"delivered" -> "Delivered"


def normalize_payment(raw: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    mapping = {
        "cash": "Cash", "credit card": "Credit Card",
        "paypay": "PayPay", "line pay": "LINE Pay",
    }
    return mapping.get(s.lower(), s)


def normalize_store(raw: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    return " ".join(raw.split())  # collapse stray whitespace/trailing spaces


def parse_rating(raw):
    if not isinstance(raw, str) or not raw.strip() or raw.strip().upper() == "N/A":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def load_and_clean(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """
    Returns (clean_df, report) where report is a dict of counters describing
    what was found/fixed, for transparency (printed by run_kpis.py / used in
    README).
    """
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    report = {"rows_in_raw_file": len(raw)}

    # --- drop fully-blank rows (every field empty) ---
    is_blank = raw.apply(lambda r: all(str(v).strip() == "" for v in r), axis=1)
    report["fully_blank_rows_dropped"] = int(is_blank.sum())
    df = raw.loc[~is_blank].copy()

    # --- drop exact full-row duplicates (true duplicate export lines) ---
    before = len(df)
    df = df.drop_duplicates(keep="first")
    report["exact_duplicate_rows_dropped"] = before - len(df)

    # --- parse/normalize each column ---
    _parsed = df["order_datetime"].apply(parse_datetime)
    df["order_datetime_parsed"] = _parsed.apply(lambda t: t[0])
    # 231 rows in the raw file are date-only ("2026/07/16", no clock time).
    # They're perfectly fine for daily/monthly revenue, but including them
    # in an hourly breakdown would silently dump them all into hour 00:00
    # and fabricate a fake "midnight rush" -- excluded from peak-hour KPI
    # via this flag, kept everywhere else.
    df["has_time_component"] = _parsed.apply(lambda t: t[1])
    report["unparseable_datetimes"] = int(df["order_datetime_parsed"].isna().sum())
    report["date_only_no_time_rows"] = int((~df["has_time_component"]).sum())

    df["unit_price_clean"] = df["unit_price"].apply(parse_price)
    report["unparseable_prices"] = int(df["unit_price_clean"].isna().sum())

    df["quantity_clean"] = pd.to_numeric(df["quantity"], errors="coerce")
    report["missing_or_bad_quantity"] = int(df["quantity_clean"].isna().sum())

    df["item_name_clean"] = df["item_name"].apply(normalize_item_name)
    report["missing_item_name"] = int(df["item_name_clean"].isna().sum())

    df["status_clean"] = df["status"].apply(normalize_status)
    report["missing_status"] = int(df["status_clean"].isna().sum())

    df["payment_method_clean"] = df["payment_method"].apply(normalize_payment)
    df["store_clean"] = df["store"].apply(normalize_store)
    df["customer_rating_clean"] = df["customer_rating"].apply(parse_rating)

    # order_id: many rows have a blank order_id (53 in the raw file). These
    # are still real line items (the rest of the row is populated) so we do
    # NOT drop them -- we just can't use order_id to group them into a
    # parent "order". We synthesize a row-level id so every line item is
    # still individually addressable.
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    report["blank_order_id_rows_kept"] = int((df["order_id"].str.strip() == "").sum())

    # --- a row is USABLE for KPI/analysis only if the fields that matter
    # (when it happened, what was sold, how much, whether it counts as a
    # sale) are all present. Rows missing any of these are excluded from
    # numeric KPIs but counted/reported, never silently averaged in as 0.
    usable_mask = (
        df["order_datetime_parsed"].notna()
        & df["unit_price_clean"].notna()
        & df["quantity_clean"].notna()
        & df["item_name_clean"].notna()
        & df["status_clean"].notna()
    )
    report["rows_excluded_missing_critical_fields"] = int((~usable_mask).sum())
    clean = df.loc[usable_mask].copy()

    clean["line_total"] = clean["unit_price_clean"] * clean["quantity_clean"]
    clean["order_hour"] = clean["order_datetime_parsed"].dt.hour
    clean["order_date"] = clean["order_datetime_parsed"].dt.date
    clean["order_month"] = clean["order_datetime_parsed"].dt.month
    clean["order_year"] = clean["order_datetime_parsed"].dt.year

    report["rows_in_clean_output"] = len(clean)
    report["status_breakdown_raw"] = clean["status_clean"].value_counts().to_dict()

    keep_cols = [
        "row_id", "order_id", "order_datetime_parsed", "order_date", "order_hour",
        "order_month", "order_year", "has_time_component", "store_clean", "item_name_clean",
        "quantity_clean", "unit_price_clean", "line_total", "payment_method_clean",
        "status_clean", "customer_rating_clean",
    ]
    clean = clean[keep_cols].rename(columns={
        "order_datetime_parsed": "order_datetime",
        "store_clean": "store",
        "item_name_clean": "item_name",
        "quantity_clean": "quantity",
        "unit_price_clean": "unit_price",
        "payment_method_clean": "payment_method",
        "status_clean": "status",
        "customer_rating_clean": "customer_rating",
    })
    return clean, report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    df, rep = load_and_clean(here / "data" / "kochiyo_orders_export.csv")
    df.to_csv(here / "data" / "cleaned_orders.csv", index=False)
    import json
    print(json.dumps(rep, indent=2, default=str))
