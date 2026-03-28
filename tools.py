"""
SoyLM - Tool calling module (preserved for future use)
Nemotron tool definitions + execution + agent loop

Usage:
    from tools import AVAILABLE_TOOLS, execute_tool_call, nemotron_agent_loop
"""

import asyncio
import json

from search import ddg_search, fetch_url_text


# ─── Tool Definitions (OpenAI function calling format) ────────
TOOL_DDG_SEARCH = {
    "type": "function",
    "function": {
        "name": "ddg_search",
        "description": "Web search (DuckDuckGo). Search for current info not in sources.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (default: 5, max: 10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

TOOL_CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate math expressions for calculations.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression (e.g. '1234 * 5.6', '100 / 3', '2**10')"
                }
            },
            "required": ["expression"]
        }
    }
}

TOOL_DATETIME = {
    "type": "function",
    "function": {
        "name": "datetime_info",
        "description": "Get current date or calculate date differences.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["now", "days_between", "add_days"],
                    "description": "now=current datetime, days_between=days between two dates, add_days=add N days to date"
                },
                "date1": {
                    "type": "string",
                    "description": "Start date YYYY-MM-DD (for days_between, add_days)"
                },
                "date2": {
                    "type": "string",
                    "description": "End date YYYY-MM-DD (for days_between)"
                },
                "days": {
                    "type": "integer",
                    "description": "Days to add (for add_days)"
                }
            },
            "required": ["action"]
        }
    }
}

TOOL_STOCK = {
    "type": "function",
    "function": {
        "name": "stock_info",
        "description": "Get stock price, market cap, and news by ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker symbol (e.g. 'NVDA', 'TSLA', '7203.T')"
                }
            },
            "required": ["ticker"]
        }
    }
}

AVAILABLE_TOOLS = [TOOL_DDG_SEARCH, TOOL_CALCULATOR, TOOL_DATETIME, TOOL_STOCK]


# ─── Tool Execution ──────────────────────────────────────────────
async def execute_tool_call(name: str, arguments: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "ddg_search":
        query = arguments.get("query", "")
        max_results = min(arguments.get("max_results", 5), 10)
        results = await ddg_search(query, max_results=max_results)
        if not results:
            return json.dumps({"results": [], "message": "No results found"}, ensure_ascii=False)
        urls = [r["url"] for r in results[:3] if r.get("url")]
        if urls:
            async def _safe_fetch(u):
                try:
                    return await asyncio.wait_for(fetch_url_text(u), timeout=15)
                except Exception:
                    return ""
            pages = await asyncio.gather(*[_safe_fetch(u) for u in urls])
            for i, page in enumerate(pages):
                if page:
                    results[i]["content"] = page[:3000]
        return json.dumps({"results": results}, ensure_ascii=False)

    elif name == "calculator":
        expr = arguments.get("expression", "")
        try:
            allowed = set("0123456789+-*/.() %e")
            if not all(c in allowed for c in expr.replace("**", "").replace("//", "")):
                return json.dumps({"error": "Invalid expression"})
            result = eval(expr)  # noqa: S307
            return json.dumps({"expression": expr, "result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif name == "datetime_info":
        from datetime import datetime as dt, timedelta
        action = arguments.get("action", "now")
        try:
            if action == "now":
                return json.dumps({"now": dt.now().strftime("%Y-%m-%d %H:%M:%S %A")})
            elif action == "days_between":
                d1 = dt.strptime(arguments.get("date1", ""), "%Y-%m-%d")
                d2 = dt.strptime(arguments.get("date2", ""), "%Y-%m-%d")
                return json.dumps({"from": arguments["date1"], "to": arguments["date2"], "days": (d2 - d1).days})
            elif action == "add_days":
                d = dt.strptime(arguments.get("date1", ""), "%Y-%m-%d")
                n = int(arguments.get("days", 0))
                result = d + timedelta(days=n)
                return json.dumps({"from": arguments["date1"], "days": n, "result": result.strftime("%Y-%m-%d %A")})
            else:
                return json.dumps({"now": dt.now().strftime("%Y-%m-%d %H:%M:%S %A")})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif name == "stock_info":
        ticker = arguments.get("ticker", "")
        try:
            import yfinance as yf
            loop = asyncio.get_running_loop()
            def _fetch():
                t = yf.Ticker(ticker)
                info = t.info
                news_raw = t.news[:3] if t.news else []
                return info, news_raw
            info, news_raw = await loop.run_in_executor(None, _fetch)
            news = [{"title": n.get("title", ""), "link": n.get("link", "")} for n in news_raw if n.get("title")]
            if not news:
                company = info.get("shortName", ticker)
                ddg_results = await ddg_search(f"{company} {ticker} stock news", max_results=3)
                news = [{"title": r["title"], "link": r["url"]} for r in ddg_results]
            result = {
                "ticker": ticker,
                "name": info.get("shortName", ""),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "currency": info.get("currency", ""),
                "market_cap": info.get("marketCap"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "news": news,
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Stock lookup failed: {e}"})

    return json.dumps({"error": f"Unknown tool: {name}"})


# ─── Agent Loop ──────────────────────────────────────────────────
async def nemotron_agent_loop(nemotron_generate_fn, messages: list[dict],
                               tools: list[dict], max_rounds: int = 3,
                               max_tokens: int = 2048) -> list[dict]:
    """Run agent loop: call LLM with tools, execute tool calls, repeat.

    Args:
        nemotron_generate_fn: async function(prompt, messages_override, tools, max_tokens) -> dict
        messages: conversation messages
        tools: tool definitions
        max_rounds: max tool-calling rounds
        max_tokens: max tokens per agent call
    Returns:
        Updated messages list with all tool interactions appended.
    """
    for _ in range(max_rounds):
        response = await nemotron_generate_fn(
            prompt="", messages_override=messages, tools=tools, max_tokens=max_tokens
        )
        tool_calls = response.get("tool_calls")
        if not tool_calls:
            break

        messages.append(response)

        for tc in tool_calls:
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
            except json.JSONDecodeError:
                args = {}
            result = await execute_tool_call(fn["name"], args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    return messages
