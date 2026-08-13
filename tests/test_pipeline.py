"""
Offline, dependency-light test suite. Deliberately does NOT require pytest
or any network/LLM access -- run with:  python3 tests/test_pipeline.py

Covers the parts of "done" that don't need a live LLM call:
  - cleaning produces sane, sanity-checkable counts
  - KPI numbers are internally consistent
  - each tool returns the standardized matched_rows contract
  - the retry-on-suspicious-result logic actually self-corrects
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clean import load_and_clean
from src.kpis import compute_kpis
from src.tools import Tools

DATA = Path(__file__).resolve().parent.parent / "data" / "kochiyo_orders_export.csv"

passed = 0
failed = 0


def check(label, condition):
    global passed, failed
    if condition:
        print(f"  PASS  {label}")
        passed += 1
    else:
        print(f"  FAIL  {label}")
        failed += 1


print("== Cleaning ==")
df, report = load_and_clean(DATA)
check("raw file has 1091 data rows", report["rows_in_raw_file"] == 1091)
check("fully-blank rows dropped", report["fully_blank_rows_dropped"] > 0)
check("exact duplicate rows dropped", report["exact_duplicate_rows_dropped"] > 0)
check("no unparseable datetimes remain", report["unparseable_datetimes"] == 0)
check("date-only rows flagged (no fabricated time)", report["date_only_no_time_rows"] > 0)
check("clean output has no null item_name", df["item_name"].isna().sum() == 0)
check("clean output has no null status", df["status"].isna().sum() == 0)
check("clean output has no negative line_total", (df["line_total"] < 0).sum() == 0)
check("statuses normalized to exactly 3 values", set(df["status"].unique()) == {"Delivered", "Cancelled", "Refunded"})
check("item names normalized ('gyoza 6pc' variants merged)",
      df[df["item_name"].str.lower().str.contains("gyoza")]["item_name"].nunique() <= 2)
check("'Kochiyo' brand-prefix merged into base item name",
      not df["item_name"].str.contains("Kochiyo", case=False, na=False).any())

print("\n== KPIs ==")
kpis = compute_kpis(df)
delivered_rows = df[df["status"] == "Delivered"]
check("total revenue matches manual recompute",
      abs(kpis["sales_summary"]["total_revenue_yen"] - float(delivered_rows["line_total"].sum())) < 0.01)
check("top_items_by_revenue is sorted descending",
      all(kpis["top_items_by_revenue"][i]["revenue_yen"] >= kpis["top_items_by_revenue"][i + 1]["revenue_yen"]
          for i in range(len(kpis["top_items_by_revenue"]) - 1)))
check("peak hour is a plausible meal-time hour, not midnight artifact",
      kpis["peak_order_hours"][0]["hour"] != 0)
check("cancellation + refund + delivered rates are consistent with row counts",
      sum(kpis["order_status_breakdown"]["counts"].values()) == len(df))

print("\n== Tools (standardized matched_rows contract) ==")
tools = Tools(df)
r1 = tools.get_sales_summary()
check("get_sales_summary has matched_rows", "matched_rows" in r1 and r1["matched_rows"] > 0)

r2 = tools.get_top_items(month=6, metric="quantity")
check("get_top_items(month=6) matches June rows only", r2["matched_rows"] == len(delivered_rows[delivered_rows["order_month"] == 6]))
check("June's best seller by quantity is Tonkotsu Ramen", r2["top_items"][0]["item"] == "Tonkotsu Ramen")

r3 = tools.get_item_stats(item_name="ramen")
check("get_item_stats substring match finds Tonkotsu Ramen", "Tonkotsu Ramen" in r3["matched_item_names"])

r4 = tools.get_peak_hours()
check("get_peak_hours excludes date-only (no-time) rows", r4["matched_rows"] < len(delivered_rows))

r5 = tools.list_available_period()
check("list_available_period reports exactly 2026", r5["years_present"] == [2026])
check("list_available_period reports June+July only", r5["months_by_year"][2026] == [6, 7])

print("\n== Retry-on-suspicious-result logic ==")
bad = tools.get_top_items(year=2025, month=6)  # 2025 doesn't exist in the data
check("wrong-year query correctly comes back empty (matched_rows=0)", bad["matched_rows"] == 0)

import os
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("GROQ_API_KEY", "unused-for-this-offline-test")

import types
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=lambda **kw: None))
from src.agent import Agent

agent = Agent(df)
trace = []
result = agent._execute_tool("get_top_items", {"year": 2025, "month": 6, "metric": "quantity"}, trace)
check("agent auto-retries and corrects the year", any(t["retried"] for t in trace))
check("agent's corrected result is no longer empty", result["matched_rows"] > 0)
check("agent's corrected answer matches the direct-tool answer for June", result["top_items"] == r2["top_items"])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
