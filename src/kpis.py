"""
kpis.py
-------
Deterministic KPI computation over the cleaned orders DataFrame.

Sales definition (the key judgment call, see README):
  "Sales" = line_total (unit_price * quantity) for rows with status == "Delivered".
  Cancelled and Refunded orders are EXCLUDED from all revenue figures --
  the customer was not actually charged/retained-charge for those, so
  counting them would overstate real revenue. They ARE still surfaced
  separately (cancellation rate, refund rate) because an operator very
  much wants to know if 30% of orders are being cancelled.
"""

from __future__ import annotations
import pandas as pd


def _delivered(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "Delivered"]


def compute_kpis(df: pd.DataFrame) -> dict:
    delivered = _delivered(df)
    total_rows = len(df)

    kpis = {}

    kpis["date_range"] = {
        "start": str(df["order_datetime"].min()),
        "end": str(df["order_datetime"].max()),
    }

    kpis["sales_summary"] = {
        "total_revenue_yen": round(float(delivered["line_total"].sum()), 2),
        "delivered_line_items": int(len(delivered)),
        "avg_line_value_yen": round(float(delivered["line_total"].mean()), 2) if len(delivered) else 0,
        "total_units_sold": int(delivered["quantity"].sum()),
    }

    status_counts = df["status"].value_counts()
    kpis["order_status_breakdown"] = {
        "counts": status_counts.to_dict(),
        "cancellation_rate_pct": round(100 * status_counts.get("Cancelled", 0) / total_rows, 2),
        "refund_rate_pct": round(100 * status_counts.get("Refunded", 0) / total_rows, 2),
    }

    top_by_revenue = (
        delivered.groupby("item_name")["line_total"].sum().sort_values(ascending=False)
    )
    top_by_qty = (
        delivered.groupby("item_name")["quantity"].sum().sort_values(ascending=False)
    )
    kpis["top_items_by_revenue"] = [
        {"item": k, "revenue_yen": round(float(v), 2)} for k, v in top_by_revenue.head(10).items()
    ]
    kpis["top_items_by_quantity"] = [
        {"item": k, "units_sold": int(v)} for k, v in top_by_qty.head(10).items()
    ]

    # Exclude date-only rows (no real clock time -> defaulted to 00:00,
    # would fake a midnight peak) from the hourly breakdown specifically.
    timed = delivered[delivered["has_time_component"]]
    kpis["rows_excluded_from_hourly_no_timestamp"] = int((~delivered["has_time_component"]).sum())
    hourly = timed.groupby("order_hour")["line_total"].agg(["count", "sum"])
    hourly = hourly.sort_values("count", ascending=False)
    kpis["peak_order_hours"] = [
        {"hour": int(h), "order_count": int(row["count"]), "revenue_yen": round(float(row["sum"]), 2)}
        for h, row in hourly.head(5).iterrows()
    ]

    store_rev = delivered.groupby("store")["line_total"].sum().sort_values(ascending=False)
    kpis["revenue_by_store"] = [
        {"store": k, "revenue_yen": round(float(v), 2)} for k, v in store_rev.items()
    ]

    pay_rev = delivered.groupby("payment_method")["line_total"].sum().sort_values(ascending=False)
    kpis["revenue_by_payment_method"] = [
        {"method": k, "revenue_yen": round(float(v), 2)} for k, v in pay_rev.items()
    ]

    rated = delivered[delivered["customer_rating"].notna()]
    kpis["avg_customer_rating"] = round(float(rated["customer_rating"].mean()), 2) if len(rated) else None
    kpis["rated_line_items"] = int(len(rated))

    monthly = delivered.groupby("order_month")["line_total"].agg(["count", "sum"]).sort_index()
    kpis["revenue_by_month"] = [
        {"month": int(m), "order_count": int(row["count"]), "revenue_yen": round(float(row["sum"]), 2)}
        for m, row in monthly.iterrows()
    ]

    return kpis


def print_kpi_report(kpis: dict) -> None:
    print("=" * 60)
    print("KOCHIYO — KPI REPORT")
    print("=" * 60)
    print(f"Data range: {kpis['date_range']['start']} -> {kpis['date_range']['end']}")
    s = kpis["sales_summary"]
    print(f"\nTotal revenue (delivered only): ¥{s['total_revenue_yen']:,.0f}")
    print(f"Delivered line items: {s['delivered_line_items']}")
    print(f"Total units sold: {s['total_units_sold']}")
    print(f"Avg line value: ¥{s['avg_line_value_yen']:,.0f}")

    ob = kpis["order_status_breakdown"]
    print(f"\nOrder status breakdown: {ob['counts']}")
    print(f"Cancellation rate: {ob['cancellation_rate_pct']}%   Refund rate: {ob['refund_rate_pct']}%")

    print("\nTop items by revenue:")
    for it in kpis["top_items_by_revenue"][:5]:
        print(f"  {it['item']:<25} ¥{it['revenue_yen']:,.0f}")

    print("\nTop items by units sold:")
    for it in kpis["top_items_by_quantity"][:5]:
        print(f"  {it['item']:<25} {it['units_sold']} units")

    print("\nPeak order hours:")
    for h in kpis["peak_order_hours"]:
        print(f"  {h['hour']:02d}:00  -  {h['order_count']} orders, ¥{h['revenue_yen']:,.0f}")

    print("\nRevenue by store:")
    for st in kpis["revenue_by_store"]:
        print(f"  {st['store']:<40} ¥{st['revenue_yen']:,.0f}")

    print("\nRevenue by payment method:")
    for p in kpis["revenue_by_payment_method"]:
        print(f"  {p['method']:<15} ¥{p['revenue_yen']:,.0f}")

    if kpis["avg_customer_rating"] is not None:
        print(f"\nAvg customer rating: {kpis['avg_customer_rating']} (from {kpis['rated_line_items']} rated line items)")

    print("\nRevenue by month:")
    for m in kpis["revenue_by_month"]:
        print(f"  Month {m['month']:02d}: {m['order_count']} orders, ¥{m['revenue_yen']:,.0f}")
    print("=" * 60)
