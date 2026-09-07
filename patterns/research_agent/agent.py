"""
Startup research agent: finds real UAE/GCC startup companies and their websites.

Design:
- The model NEVER answers from its own memory. It must call `web_search` to find
  real results, and every company in the final answer must be grounded in a
  search result URL (evidence_url).
- The agent loop runs tool calls until the model is ready, then the final
  message is forced into strict JSON matching CompanyList and validated with
  Pydantic. If validation fails, we retry once with the error fed back to the
  model (self-correction), then give up loudly rather than return bad data.
"""

import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from schema import CompanyList

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MODEL = "gpt-4o"
MAX_TOOL_ROUNDS = 6
MAX_SELF_CORRECTIONS = 1

SYSTEM_PROMPT = """You are a research analyst that finds real, currently operating startup \
companies headquartered in the UAE or GCC (Saudi Arabia, Qatar, Bahrain, Kuwait, Oman).

Rules:
- Use the web_search tool to find companies. Never invent a company, website, or country \
from memory alone - every company you report must come from a search result.
- The "website" field MUST be that specific company's own domain (e.g. https://company.com), \
never a news article, directory, market-report, or aggregator page that merely lists or \
mentions the company. If a search result is a list/report of multiple companies (e.g. \
"Top 10 fintech startups in UAE"), that URL may only be used as "evidence_url", never as \
"website" - and it may be reused as evidence_url for more than one company, but each \
company's "website" must be unique to that company.
- If you cannot find a company's own official website, do not include that company.
- Do not include the same company twice.
- When you have enough verified results, respond with ONLY a JSON object matching this shape \
(no markdown, no prose):
{"query": "<original user query>", "companies": [
  {"name": str, "website": str, "country": str, "evidence_url": str, "summary": str}, ...
]}
"""


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Tavily and return simplified results."""
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set - add it to your .env file")

    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
        for r in data.get("results", [])
    ]


tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date information. Use this to find real startup companies, never rely on memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Number of results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]

available_functions = {"web_search": web_search}


def _run_tool_loop(messages: list) -> str:
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            print(f"[tool call] {fn_name}({fn_args})")
            try:
                result = available_functions[fn_name](**fn_args)
            except Exception as exc:  # tool failure shouldn't crash the agent
                result = {"error": str(exc)}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    raise RuntimeError(f"Agent did not finish within {MAX_TOOL_ROUNDS} tool rounds")


def research_companies(query: str, verify_grounding: bool = True) -> CompanyList:
    """Run the agent end-to-end and return a validated, grounded CompanyList, or raise.

    Pipeline: LLM output -> schema validation (structure) -> grounding check (truth).
    Companies that pass schema but fail grounding (dead website, evidence page
    doesn't actually mention them) are dropped rather than shipped.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    last_error = None
    result: CompanyList | None = None
    for attempt in range(MAX_SELF_CORRECTIONS + 1):
        raw = _run_tool_loop(messages)
        try:
            data = json.loads(raw)
            result = CompanyList.model_validate(data)
            break
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your last response failed validation with this error:\n"
                        f"{exc}\n"
                        "Respond again with ONLY corrected JSON matching the required schema."
                    ),
                }
            )

    if result is None:
        raise ValueError(f"Agent failed to produce valid output after retries: {last_error}")

    if verify_grounding:
        from grounding import filter_grounded

        grounded_result, checks = filter_grounded(result)
        for check in checks:
            if not check.passed:
                print(f"[grounding] dropped '{check.company.name}': {check.issues}")
        return grounded_result

    return result


if __name__ == "__main__":
    result = research_companies("Find fintech startups based in the UAE")
    print(result.model_dump_json(indent=2))
