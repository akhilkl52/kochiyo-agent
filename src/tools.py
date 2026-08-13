"""
tools.py
--------
The functions the LLM agent is allowed to call, plus their JSON-schema
descriptions (OpenAI-compatible "tools" format, which both Groq and Ollama
speak). Every tool returns a small JSON-serializable dict that always
includes "matched_rows" so the agent loop can mechanically detect a
suspicious (empty/zero) result and retry -- see agent.py.

All tools operate only on Delivered rows for revenue/quantity figures
unless the tool is explicitly about status/cancellations.
"""

from __future__ import annotations
import pandas as pd


def _filter_period(df: pd.DataFrame, year: int | None, month: int | None) -> pd.DataFrame:
    out = df
    if year is not None:
        out = out[out["order_year"] == year]
    if month is not None:
        out = out[out["order_month"] == month]
    return out


class Tools:
    """Wraps the cleaned DataFrame and exposes callable tool methods."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ------------------------------------------------------------------
    def list_available_period(self, **_) -> dict:
        """No-argument tool: tells the agent what years/months actually
        exist in the data, so it can self-correct a wrong year/month guess
        instead of asking the user or giving up."""
        df = self.df
        years = sorted(df["order_year"].dropna().unique().tolist())
        months_by_year = {
            int(y): sorted(df[df["order_year"] == y]["order_month"].dropna().unique().tolist())
            for y in years
        }
        return {
            "matched_rows": len(df),
            "years_present": [int(y) for y in years],
            "months_by_year": months_by_year,
            "earliest_date": str(df["order_datetime"].min()),
            "latest_date": str(df["order_datetime"].max()),
        }

    # ------------------------------------------------------------------
    def get_sales_summary(self, year: int | None = None, month: int | None = None, **_) -> dict:
        """Total revenue, order count, avg order value for delivered orders
        in an optional year/month window."""
        d = _filter_period(self.df, year, month)
        delivered = d[d["status"] == "Delivered"]
        return {
            "matched_rows": int(len(delivered)),
            "filters": {"year": year, "month": month},
            "total_revenue_yen": round(float(delivered["line_total"].sum()), 2),
            "total_units_sold": int(delivered["quantity"].sum()),
            "avg_line_value_yen": round(float(delivered["line_total"].mean()), 2) if len(delivered) else 0,
        }

    # ------------------------------------------------------------------
    def get_top_items(self, year: int | None = None, month: int | None = None,
                       metric: str = "revenue", top_n: int = 5, order: str = "top", **_) -> dict:
        """Top- or bottom-selling items by revenue or units sold, for
        delivered orders in an optional year/month window. Set order='bottom'
        for 'least/worst selling item' style questions."""
        d = _filter_period(self.df, year, month)
        delivered = d[d["status"] == "Delivered"]
        if len(delivered) == 0:
            return {"matched_rows": 0, "filters": {"year": year, "month": month}, "top_items": []}
        col = "line_total" if metric == "revenue" else "quantity"
        grouped = delivered.groupby("item_name")[col].sum().sort_values(ascending=(order == "bottom"))
        top = [{"item": k, metric: round(float(v), 2)} for k, v in grouped.head(top_n).items()]
        return {"matched_rows": int(len(delivered)), "filters": {"year": year, "month": month, "metric": metric, "order": order}, "top_items": top}

    # ------------------------------------------------------------------
    def get_peak_hours(self, year: int | None = None, month: int | None = None, top_n: int = 5, **_) -> dict:
        """Busiest hours of day by order count, for delivered orders with a
        real timestamp (date-only rows are excluded to avoid a fake
        midnight spike)."""
        d = _filter_period(self.df, year, month)
        delivered = d[(d["status"] == "Delivered") & (d["has_time_component"])]
        if len(delivered) == 0:
            return {"matched_rows": 0, "filters": {"year": year, "month": month}, "peak_hours": []}
        hourly = delivered.groupby("order_hour")["line_total"].agg(["count", "sum"]).sort_values("count", ascending=False)
        top = [{"hour": int(h), "order_count": int(r["count"]), "revenue_yen": round(float(r["sum"]), 2)}
               for h, r in hourly.head(top_n).iterrows()]
        return {"matched_rows": int(len(delivered)), "filters": {"year": year, "month": month}, "peak_hours": top}

    # ------------------------------------------------------------------
    def get_item_stats(self, item_name: str, year: int | None = None, month: int | None = None, **_) -> dict:
        """Revenue/units/avg-rating for one specific item. Matches
        case-insensitively and by substring so 'ramen' matches 'Tonkotsu
        Ramen'."""
        d = _filter_period(self.df, year, month)
        mask = d["item_name"].str.contains(item_name, case=False, na=False)
        matched = d[mask]
        delivered = matched[matched["status"] == "Delivered"]
        if len(matched) == 0:
            return {"matched_rows": 0, "filters": {"item_name": item_name, "year": year, "month": month}, "note": "no item name matched this substring"}
        rated = delivered[delivered["customer_rating"].notna()]
        return {
            "matched_rows": int(len(matched)),
            "matched_item_names": sorted(matched["item_name"].unique().tolist()),
            "filters": {"item_name": item_name, "year": year, "month": month},
            "delivered_units": int(delivered["quantity"].sum()),
            "delivered_revenue_yen": round(float(delivered["line_total"].sum()), 2),
            "avg_customer_rating": round(float(rated["customer_rating"].mean()), 2) if len(rated) else None,
        }

    # ------------------------------------------------------------------
    def get_status_breakdown(self, year: int | None = None, month: int | None = None, **_) -> dict:
        """Delivered/Cancelled/Refunded counts and rates for an optional
        year/month window."""
        d = _filter_period(self.df, year, month)
        if len(d) == 0:
            return {"matched_rows": 0, "filters": {"year": year, "month": month}}
        counts = d["status"].value_counts().to_dict()
        total = len(d)
        return {
            "matched_rows": int(total),
            "filters": {"year": year, "month": month},
            "counts": counts,
            "cancellation_rate_pct": round(100 * counts.get("Cancelled", 0) / total, 2),
            "refund_rate_pct": round(100 * counts.get("Refunded", 0) / total, 2),
        }

    # ------------------------------------------------------------------
    def get_breakdown_by(self, dimension: str, year: int | None = None, month: int | None = None, **_) -> dict:
        """Revenue broken down by 'store' or 'payment_method', delivered
        orders only, optional year/month window."""
        if dimension not in ("store", "payment_method"):
            return {"matched_rows": 0, "error": f"unsupported dimension '{dimension}', use 'store' or 'payment_method'"}
        d = _filter_period(self.df, year, month)
        delivered = d[d["status"] == "Delivered"]
        if len(delivered) == 0:
            return {"matched_rows": 0, "filters": {"dimension": dimension, "year": year, "month": month}, "breakdown": []}
        grouped = delivered.groupby(dimension)["line_total"].sum().sort_values(ascending=False)
        breakdown = [{dimension: k, "revenue_yen": round(float(v), 2)} for k, v in grouped.items()]
        return {"matched_rows": int(len(delivered)), "filters": {"dimension": dimension, "year": year, "month": month}, "breakdown": breakdown}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_available_period",
            "description": "Get the years/months actually present in the dataset. Call this first if unsure what date range exists, or to self-correct after a year/month filter returned 0 matched_rows.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sales_summary",
            "description": "Total revenue, units sold, and avg order value for delivered orders, optionally filtered to a year and/or month.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": ["integer", "null"], "description": "e.g. 2026. Omit or pass null for all years."},
                    "month": {"type": ["integer", "null"], "description": "1-12. Omit or pass null for all months."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_items",
            "description": "Top- or bottom-selling items by revenue or units sold, for delivered orders, optionally filtered to a year/month. Use order='bottom' for 'least/worst selling item' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": ["integer", "null"]},
                    "month": {"type": ["integer", "null"]},
                    "metric": {"type": "string", "enum": ["revenue", "quantity"], "description": "Rank by revenue (yen) or by units sold. Default revenue."},
                    "top_n": {"type": ["integer", "null"], "description": "How many items to return. Default 5."},
                    "order": {"type": "string", "enum": ["top", "bottom"], "description": "'top' for best-selling (default), 'bottom' for least/worst-selling."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_peak_hours",
            "description": "Busiest hours of day by order count, for delivered orders, optionally filtered to a year/month.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": ["integer", "null"]},
                    "month": {"type": ["integer", "null"]},
                    "top_n": {"type": ["integer", "null"]},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_stats",
            "description": "Revenue, units sold, and avg rating for ONE specific menu item (substring/case-insensitive match), optionally filtered to a year/month.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "e.g. 'ramen', 'chicken karaage', 'tempura'"},
                    "year": {"type": ["integer", "null"]},
                    "month": {"type": ["integer", "null"]},
                },
                "required": ["item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status_breakdown",
            "description": "Counts and rates of Delivered/Cancelled/Refunded orders, optionally filtered to a year/month.",
            "parameters": {
                "type": "object",
                "properties": {"year": {"type": ["integer", "null"]}, "month": {"type": ["integer", "null"]}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_breakdown_by",
            "description": "Revenue broken down by 'store' or by 'payment_method', delivered orders only, optionally filtered to a year/month.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": ["store", "payment_method"]},
                    "year": {"type": ["integer", "null"]},
                    "month": {"type": ["integer", "null"]},
                },
                "required": ["dimension"],
            },
        },
    },
]
