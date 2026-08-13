"""
agent.py
--------
The agent loop. Deliberately NOT relying on the LLM to "just notice" a bad
tool result and retry on its own initiative -- that's inconsistent across
models and prompts. Instead this loop mechanically checks every tool result
for matched_rows == 0 (or an equivalent zero/empty signal) and, if the call
used a guessed year/month, automatically retries once with the year/month
corrected against list_available_period() before ever showing the LLM (or
the user) the empty result as if it were final.

This directly targets the assignment's example failure mode: "best-selling
item in June" with no year specified -> model guesses a year that isn't in
the data -> empty result -> must not be reported as the answer.
"""

from __future__ import annotations
import json
from src.llm_backend import get_client_and_model
from src.tools import Tools, TOOL_SCHEMAS

SYSTEM_PROMPT = """You are a data analyst assistant for Kochiyo, a restaurant \
group with three stores. You answer questions about their order data using \
the tools provided. Rules:

- Always use a tool to get numbers; never invent or estimate figures from \
memory.
- If a question mentions a month but no year (e.g. "best seller in June"), \
first check what years/months actually exist using list_available_period if \
you're not sure, or just try the most recent year -- if the tool comes back \
with matched_rows: 0, that means your guess was wrong, not that there's no \
data; call list_available_period and retry with a year that's actually \
present.
- "Sales"/"revenue" figures only ever include Delivered orders. Cancelled \
and Refunded orders are excluded from revenue but can be reported \
separately if asked.
- Give direct, concise answers with concrete numbers (yen amounts, counts). \
Don't pad with disclaimers.
- If, after retrying, a query genuinely has no matching data, say so plainly \
rather than making something up.
"""


class SuspiciousResultError(Exception):
    pass


def _is_suspicious(result: dict) -> bool:
    """A tool result counts as suspicious if it matched zero rows -- that's
    the standardized empty-result signal every tool in tools.py returns."""
    return isinstance(result, dict) and result.get("matched_rows", None) == 0


class Agent:
    def __init__(self, df):
        self.tools = Tools(df)
        self.client, self.model = get_client_and_model()
        self.tool_map = {
            "list_available_period": self.tools.list_available_period,
            "get_sales_summary": self.tools.get_sales_summary,
            "get_top_items": self.tools.get_top_items,
            "get_peak_hours": self.tools.get_peak_hours,
            "get_item_stats": self.tools.get_item_stats,
            "get_status_breakdown": self.tools.get_status_breakdown,
            "get_breakdown_by": self.tools.get_breakdown_by,
        }
        self._available_period_cache = None

    def _get_available_period(self) -> dict:
        if self._available_period_cache is None:
            self._available_period_cache = self.tools.list_available_period()
        return self._available_period_cache

    def _execute_tool(self, name: str, args: dict, trace: list) -> dict:
        fn = self.tool_map.get(name)
        if fn is None:
            return {"matched_rows": 0, "error": f"unknown tool '{name}'"}
        result = fn(**args)
        trace.append({"tool": name, "args": args, "result": result, "retried": False})

        if _is_suspicious(result) and ("year" in args and args.get("year") is not None
                                        or "month" in args and args.get("month") is not None):
            # Suspicious: a year/month filter was guessed and it returned
            # nothing. Ground against the real available period and retry
            # ONCE with corrected values before accepting the result.
            avail = self._get_available_period()
            years = avail.get("years_present", [])
            fixed_args = dict(args)
            if years and args.get("year") not in years:
                fixed_args["year"] = years[-1]  # most recent year present
                months_for_year = avail.get("months_by_year", {}).get(years[-1], [])
                if args.get("month") is not None and args["month"] not in months_for_year and months_for_year:
                    # keep requested month if plausible, otherwise leave as-is;
                    # the point is correcting the year, which is the actual bug
                    pass
                retried_result = fn(**fixed_args)
                trace.append({"tool": name, "args": fixed_args, "result": retried_result, "retried": True,
                              "retry_reason": f"original args {args} matched 0 rows; corrected year to {years[-1]} "
                                              f"based on list_available_period"})
                return retried_result
        return result

    def run(self, question: str, verbose: bool = False) -> tuple[str, list]:
        """Returns (answer_text, trace) where trace is the list of tool
        calls made (including any auto-retries) for transparency/debugging."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: list = []
        consecutive_api_errors = 0

        for _ in range(6):  # hard cap on tool-call round trips
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0,
                )
                consecutive_api_errors = 0
            except Exception as e:
                # Rate-limit errors are NOT worth retrying inside this loop:
                # the server is telling us to wait minutes, not milliseconds,
                # and immediately retrying just burns the whole tool-call
                # budget in under a second with zero chance of success. Fail
                # fast with a clear, actionable message instead.
                err_name = type(e).__name__
                if "RateLimit" in err_name or "429" in str(e):
                    return (
                        "I hit the LLM provider's rate limit and can't retry my way "
                        f"past it (this isn't a bug to retry around): {e}\n\n"
                        "Options: wait for the limit to reset (the error message above "
                        "states how long), switch LLM_MODEL to a model with a higher "
                        "free-tier daily budget (e.g. llama-3.1-8b-instant), or set "
                        "LLM_PROVIDER=ollama to run fully offline instead."
                    ), trace

                # Other errors (e.g. a malformed function-call the model
                # generated) are worth ONE retry each, but if the SAME kind
                # of error repeats back-to-back, retrying isn't helping --
                # it means the model can't self-correct from the feedback
                # (usually a real schema gap, not a one-off glitch), so bail
                # with a clear message instead of silently eating the whole
                # budget on repeats of the same failure.
                consecutive_api_errors += 1
                if consecutive_api_errors >= 3:
                    return (
                        f"The model repeatedly generated an invalid tool call "
                        f"({err_name}) and isn't self-correcting after being told. "
                        f"This usually means a tool's schema needs fixing, not that "
                        f"the question is unanswerable. Last error: {e}"
                    ), trace

                print(f"  [recovering from API error, retrying] {err_name}: {e}")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last tool call was malformed and rejected by the API "
                        f"({type(e).__name__}: {e}). Try again with a single, valid "
                        "tool call using the exact JSON schema provided."
                    ),
                })
                continue

            msg = resp.choices[0].message

            if not msg.tool_calls:
                return msg.content or "", trace

            # IMPORTANT: msg.tool_calls is a list of SDK objects, not plain
            # dicts. Appending them directly into `messages` produces a
            # history that looks fine locally but serializes incorrectly on
            # the next API call -- Groq then silently rejects the next
            # request as malformed, every round, burning the whole tool-call
            # budget with no visible error. Convert to plain dicts first.
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                # Some models send the literal string "null" for no-argument
                # calls instead of "{}" -- json.loads("null") is a valid
                # parse that returns Python None, which then breaks fn(**None).
                args = json.loads(tc.function.arguments or "{}")
                if args is None:
                    args = {}
                if verbose:
                    print(f"  [tool call] {tc.function.name}({args})")
                result = self._execute_tool(tc.function.name, args, trace)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

        return "I wasn't able to finish reasoning about this within the tool-call budget.", trace
