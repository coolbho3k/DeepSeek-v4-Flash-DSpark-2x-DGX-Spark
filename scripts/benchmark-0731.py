import argparse
import asyncio
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url, body):
    for attempt in range(4):
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                return json.load(response)
        except urllib.error.URLError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def tokenize_url(base_url):
    return base_url.removesuffix("/v1") + "/tokenize"


def build_prompt(base_url, model, target, nonce):
    unit = "benchmark context datum "
    text = f"unique request {nonce} " + unit * max(1, target // 3)
    while True:
        count = request_json(tokenize_url(base_url), {"model": model, "prompt": text})["count"]
        if count >= target:
            return text
        text += unit * max(1, (target - count) // 3)


def stream_one(base_url, model, prompt, max_tokens, temperature, ignore_eos):
    requested_words = max_tokens if max_tokens > 0 else 128
    instruction = (
        f"\nReturn exactly {requested_words} numbered lowercase English words, then stop."
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + instruction}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "top_p": 0.95,
        "chat_template_kwargs": {"thinking": False},
    }
    if max_tokens > 0:
        body["max_tokens"] = max_tokens
    if ignore_eos is not None:
        body["ignore_eos"] = ignore_eos
    request = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    usage = None
    output = []
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                break
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if first is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                first = time.perf_counter()
            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
            content = delta.get("content") or ""
            output.extend((reasoning, content))
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    measured = None if usage else request_json(tokenize_url(base_url), {"model": model, "prompt": "".join(output)})["count"]
    output_tokens = (usage or {}).get("completion_tokens", measured or 0)
    ttft = (first or finished) - started
    prompt_tokens = (usage or {}).get("prompt_tokens", 0)
    return {"ttft_s": ttft, "elapsed_s": finished - started, "prompt_tokens": prompt_tokens, "prefill_tok_s": prompt_tokens / max(0.001, ttft), "output_tokens": output_tokens, "output_tok_s": output_tokens / max(0.001, finished - (first or finished))}


async def run_case(
    base_url,
    model,
    target_prompt_tokens,
    concurrency,
    max_tokens,
    temperature,
    ignore_eos,
    per_request_max_tokens=None,
    nonce_prefix="",
):
    prompts = await asyncio.gather(*[
        asyncio.to_thread(
            build_prompt,
            base_url,
            model,
            target_prompt_tokens,
            f"{nonce_prefix}p{target_prompt_tokens}-c{concurrency}-r{index}",
        )
        for index in range(concurrency)
    ])
    started = time.perf_counter()
    request_max_tokens = per_request_max_tokens or [max_tokens] * concurrency
    if len(request_max_tokens) != concurrency:
        raise ValueError(
            "per-request max-token count must match concurrency: "
            f"{len(request_max_tokens)} != {concurrency}"
        )
    results = await asyncio.gather(*[
        asyncio.to_thread(
            stream_one,
            base_url,
            model,
            prompt,
            request_max_tokens[index],
            temperature,
            ignore_eos,
        )
        for index, prompt in enumerate(prompts)
    ])
    elapsed = time.perf_counter() - started
    total = sum(item["output_tokens"] for item in results)
    return {"concurrency": concurrency, "elapsed_s": elapsed, "aggregate_tok_s": total / max(0.001, elapsed), "median_ttft_s": statistics.median(item["ttft_s"] for item in results), "median_prefill_tok_s": statistics.median(item["prefill_tok_s"] for item in results), "median_output_tok_s": statistics.median(item["output_tok_s"] for item in results), "requests": results}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--prompt-lengths", default="256,2048,8192,32768,131072")
    parser.add_argument("--concurrency", default="1,2,4,6")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--per-request-max-tokens",
        default="",
        help="comma-separated output limits; length must match each concurrency",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--nonce-prefix",
        default="",
        help="unique prompt prefix for repeated runs with prefix caching enabled",
    )
    parser.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "base_url": args.base_url, "cases": []}
    per_request_max_tokens = (
        [int(value) for value in args.per_request_max_tokens.split(",")]
        if args.per_request_max_tokens
        else None
    )
    for prompt_length in [int(value) for value in args.prompt_lengths.split(",")]:
        for concurrency in [int(value) for value in args.concurrency.split(",")]:
            case = await run_case(
                args.base_url,
                args.model,
                prompt_length,
                concurrency,
                args.max_tokens,
                args.temperature,
                args.ignore_eos,
                per_request_max_tokens,
                args.nonce_prefix,
            )
            case["target_prompt_tokens"] = prompt_length
            report["cases"].append(case)
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps(case, sort_keys=True), flush=True)


asyncio.run(main())
