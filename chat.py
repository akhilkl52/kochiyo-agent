"""Interactive Q&A CLI. Run: python chat.py"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from src.clean import load_and_clean
from src.agent import Agent

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    df, _ = load_and_clean(HERE / "data" / "kochiyo_orders_export.csv")
    agent = Agent(df)

    print("Kochiyo order-data assistant. Ask a question (or 'quit').")
    print("e.g. 'What was our best-selling item in June?'")
    print("     'What time of day do we get the most orders?'")
    print()

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit"):
            break
        answer, trace = agent.run(q, verbose=True)
        print(answer)
        print()
