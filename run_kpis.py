"""Run this to produce the sanity-checkable KPI report from the raw CSV."""
from pathlib import Path
from src.clean import load_and_clean
from src.kpis import compute_kpis, print_kpi_report
import json

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    df, report = load_and_clean(HERE / "data" / "kochiyo_orders_export.csv")
    df.to_csv(HERE / "data" / "cleaned_orders.csv", index=False)

    print("Data cleaning report:")
    print(json.dumps(report, indent=2, default=str))
    print()

    kpis = compute_kpis(df)
    print_kpi_report(kpis)

    with open(HERE / "data" / "kpi_report.json", "w") as f:
        json.dump(kpis, f, indent=2, default=str)
