#!/usr/bin/env python3
"""LLM-routed template agent for MOPS tasks.

Workflow:
1) Route user request to one task family: remapping / streamline / pathline.
2) Generate a NEW Python job script from an abstracted template.
3) Execute the generated job script (unless --dry-run).

This avoids directly running tutorial scripts and instead uses tutorial APIs
as reusable building blocks.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime
import urllib.error
import urllib.request
from urllib.parse import urlencode
from pathlib import Path
from typing import Any

from task_templates import (
    DEFAULT_REMAPPING_CONFIG,
    DEFAULT_STREAMLINE_CONFIG,
    DEFAULT_PATHLINE_CONFIG,
    render_job_script,
    generate_remapping_yaml_config,
)


SUPPORTED_TASKS = ("remapping", "streamline", "pathline")

TASK_ALIASES = {
    "remap": "remapping",
    "regrid": "remapping",
    "re_mapping": "remapping",
    "stream_line": "streamline",
    "path_line": "pathline",
}

SUPPORTED_PROVIDERS = ("auto", "openai", "foundry")


class RouteResult:
    def __init__(self, task, confidence, reason):
        self.task = task
        self.confidence = confidence
        self.reason = reason


def _normalize_task(value):
    if not value:
        return None
    key = value.strip().lower()
    key = TASK_ALIASES.get(key, key)
    if key in SUPPORTED_TASKS:
        return key
    return None


def _extract_first_json_object(text):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _pick_first_non_empty(*values):
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return ""


def resolve_provider_and_credentials(args):
    provider = (args.provider or "auto").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        provider = "auto"

    base_url = _pick_first_non_empty(
        args.base_url,
        os.environ.get("FOUNDRY_BASE_URL"),
        os.environ.get("AZURE_INFERENCE_ENDPOINT"),
        os.environ.get("AZURE_OPENAI_ENDPOINT"),
        os.environ.get("OPENAI_BASE_URL"),
    )
    api_key = _pick_first_non_empty(
        args.api_key,
        os.environ.get("FOUNDRY_API_KEY"),
        os.environ.get("AZURE_INFERENCE_API_KEY"),
        os.environ.get("AZURE_OPENAI_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
    )

    if provider == "auto":
        if any(tag in base_url.lower() for tag in ["azure.com", "ai.azure.com", "services.ai.azure.com"]):
            provider = "foundry"
        elif os.environ.get("FOUNDRY_API_KEY") or os.environ.get("AZURE_INFERENCE_API_KEY"):
            provider = "foundry"
        else:
            provider = "openai"

    return provider, base_url, api_key


def build_chat_endpoint(base_url, provider, api_version, model=None):
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        endpoint = normalized
    else:
        # Azure OpenAI requires /openai/deployments/{deployment}/chat/completions
        if provider == "foundry" and model:
            endpoint = normalized + f"/openai/deployments/{model}/chat/completions"
        else:
            endpoint = normalized + "/chat/completions"

    if provider == "foundry" and api_version:
        separator = "&" if "?" in endpoint else "?"
        endpoint = endpoint + separator + urlencode({"api-version": api_version})
    return endpoint


def call_chat_completion(messages, model, temperature, provider, base_url, api_key, timeout_sec, api_version):
    endpoint = build_chat_endpoint(base_url, provider, api_version, model)
    payload = {
        "model": model,
        "temperature": float(temperature),
        "messages": messages,
    }

    headers = {"Content-Type": "application/json"}
    if provider == "foundry":
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = "Bearer {0}".format(api_key)

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError("LLM API HTTP {0}: {1}".format(e.code, body)) from e
    except urllib.error.URLError as e:
        raise RuntimeError("LLM API connection failed: {0}".format(e)) from e

    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError("Unexpected LLM response format: {0}".format(raw)) from e
    return content


def extract_remapping_config_with_llm(
    user_request,
    model,
    api_key,
    base_url,
    timeout_sec,
    provider,
    api_version,
    defaults,
):
    system_prompt = textwrap.dedent(
        """
        You are an AI planner for MOPS remapping.
        Read natural-language request and return ONLY one JSON object for remapping config.
        Must support vague geo descriptions, e.g. "near Gulf of Mexico" by inferring lat/lon ranges.
        Return ONLY one JSON object with these exact keys:
        {
          "yaml_path": string,
          "data_folder": string,
          "device": "gpu|cpu",
          "time_stamp": "YYYY-MM-DD",
          "time_step": integer,
          "width": integer,
          "height": integer,
          "lat_range": [float, float],
          "lon_range": [float, float],
          "fixed_depth": float,
          "add_temperature": boolean,
          "add_salinity": boolean,
          "channels": [integer,...],
          "cmap_name": string,
          "save_colorbar": boolean,
          "output_subdir": string
        }
                Rules:
                - If user gives explicit YAML path, set "yaml_path" to it.
                - If user gives data folder path instead, set "data_folder" to it and "yaml_path" to null.
                - If user gives place name but no coordinates, infer reasonable lat_range/lon_range.
                - If user omits a field, set it to null.
                - Keep all numbers in proper numeric type.
        Do not output markdown.
        """
    ).strip()

    content = call_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "User request:\n{0}\n\nDefault config:\n{1}".format(
                    user_request,
                    json.dumps(defaults, ensure_ascii=False),
                ),
            },
        ],
        model=model,
        temperature=0.0,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
        api_version=api_version,
    )
    parsed = _extract_first_json_object(content)
    if parsed is None:
        raise RuntimeError("Config extractor did not return valid JSON")

    cleaned = {}
    for k, v in parsed.items():
        if v is not None:
            cleaned[k] = v
    return cleaned


def extract_streamline_config_with_llm(
    user_request,
    model,
    api_key,
    base_url,
    timeout_sec,
    provider,
    api_version,
    defaults,
):
    system_prompt = textwrap.dedent(
        """
        You are an AI planner for MOPS streamline analysis.
        Read natural-language request and return ONLY one JSON object for streamline config.
        Must support vague geo descriptions, e.g. "near Gulf of Mexico" by inferring lat/lon ranges.
        Return ONLY one JSON object with these exact keys:
        {
          "yaml_path": string,
          "device": "gpu|cpu",
          "start_date": "YYYY-MM-DD",
          "duration_days": integer,
          "fixed_depth": float,
          "lat_range": [float, float],
          "lon_range": [float, float],
          "grid": [integer, integer],
          "method": "rk4|euler",
          "delta_minutes": integer,
          "record_every_minutes": integer,
          "color_by": "speed|velocity",
          "output_subdir": string
        }
        Rules:
        - If user gives place name but no coordinates, infer reasonable lat_range/lon_range.
        - If user omits a field, set it to null.
        - Keep all numbers in proper numeric type.
        - grid should be [rows, cols] for seed points.
        Do not output markdown.
        """
    ).strip()

    content = call_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "User request:\n{0}\n\nDefault config:\n{1}".format(
                    user_request,
                    json.dumps(defaults, ensure_ascii=False),
                ),
            },
        ],
        model=model,
        temperature=0.0,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
        api_version=api_version,
    )
    parsed = _extract_first_json_object(content)
    if parsed is None:
        raise RuntimeError("Config extractor did not return valid JSON")

    cleaned = {}
    for k, v in parsed.items():
        if v is not None:
            cleaned[k] = v
    return cleaned


def extract_pathline_config_with_llm(
    user_request,
    model,
    api_key,
    base_url,
    timeout_sec,
    provider,
    api_version,
    defaults,
):
    system_prompt = textwrap.dedent(
        """
        You are an AI planner for MOPS pathline analysis.
        Read natural-language request and return ONLY one JSON object for pathline config.
        Must support vague geo descriptions, e.g. "near Gulf of Mexico" by inferring lat/lon ranges.
        Return ONLY one JSON object with these exact keys:
        {
          "yaml_path": string,
          "device": "gpu|cpu",
          "start_year": integer,
          "start_month": integer (1-12),
          "end_year": integer,
          "end_month": integer (1-12),
          "direction": "forward|backward",
          "fixed_depth": float,
          "lat_range": [float, float],
          "lon_range": [float, float],
          "grid": [integer, integer],
          "method": "rk4|euler",
          "delta_minutes": integer,
          "record_every_minutes": integer,
          "color_by": "velocity|speed",
          "output_subdir": string
        }
        Rules:
        - If user gives place name but no coordinates, infer reasonable lat_range/lon_range.
        - TIME DURATION RULES (IMPORTANT):
          * Default: start_year=1, start_month=1, end_year=1, end_month=12 (1 year)
          * "2 years" → start_year=1, start_month=1, end_year=2, end_month=12
          * "3 months" → start_year=1, start_month=1, end_year=1, end_month=3
          * "24 months" → start_year=1, start_month=1, end_year=2, end_month=12
          * "18 months" → start_year=1, start_month=1, end_year=1, end_month=18 (which wraps to year 2, month 6)
          * If user says "X years", set end_year = start_year + X - 1, end_month = 12
          * If user says "X months", calculate: total_months = X, end_year = 1 + (total_months - 1) // 12, end_month = ((total_months - 1) % 12) + 1
        - For multi-year simulations, use coarser sampling:
          * delta_minutes: 60 (not 1)
          * record_every_minutes: 360 (not 6)
        - If user omits a field, set it to null.
        - Keep all numbers in proper numeric type.
        - grid should be [rows, cols] for seed points.
        Do not output markdown.
        """
    ).strip()

    content = call_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "User request:\n{0}\n\nDefault config:\n{1}".format(
                    user_request,
                    json.dumps(defaults, ensure_ascii=False),
                ),
            },
        ],
        model=model,
        temperature=0.0,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
        api_version=api_version,
    )
    parsed = _extract_first_json_object(content)
    if parsed is None:
        raise RuntimeError("Config extractor did not return valid JSON")

    cleaned = {}
    for k, v in parsed.items():
        if v is not None:
            cleaned[k] = v
    return cleaned


def build_task_config(
    task,
    user_request,
    model,
    api_key,
    base_url,
    timeout_sec,
    provider="openai",
    api_version="",
    strict_llm=False,
    output_dir=None,
):
    if task == "remapping":
        cfg = dict(DEFAULT_REMAPPING_CONFIG)
        extractor = extract_remapping_config_with_llm
    elif task == "streamline":
        cfg = dict(DEFAULT_STREAMLINE_CONFIG)
        extractor = extract_streamline_config_with_llm
    elif task == "pathline":
        cfg = dict(DEFAULT_PATHLINE_CONFIG)
        extractor = extract_pathline_config_with_llm
    else:
        return {}

    if api_key and base_url:
        try:
            llm_cfg = extractor(
                user_request=user_request,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_sec=timeout_sec,
                provider=provider,
                api_version=api_version,
                defaults=cfg,
            )

            # Auto-generate YAML if data_folder provided but no yaml_path
            if task == "remapping" and "data_folder" in llm_cfg and not llm_cfg.get("yaml_path"):
                data_folder = llm_cfg.pop("data_folder")
                print(f"[Agent] Generating YAML from data folder: {data_folder}")
                try:
                    import yaml
                    yaml_config = generate_remapping_yaml_config(data_folder)
                    if yaml_config and output_dir:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        yaml_path = output_dir / "auto_generated.yaml"
                        with open(str(yaml_path), 'w') as f:
                            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
                        llm_cfg["yaml_path"] = str(yaml_path)
                        print(f"[Agent] Generated YAML config: {yaml_path}")
                    else:
                        print(f"[Agent] Warning: Could not generate YAML from {data_folder}")
                except Exception as e:
                    print(f"[Agent] Warning: YAML generation failed: {e}")

            cfg = _deep_merge(cfg, llm_cfg)
        except Exception as e:
            if strict_llm:
                raise
            print(f"[Agent] {task} JSON extraction fallback: {e}")
    elif strict_llm:
        raise RuntimeError("AI-only mode enabled but LLM endpoint/key not configured")

    return cfg


def route_task_with_llm(
    user_request,
    model,
    api_key,
    base_url,
    temperature,
    timeout_sec,
    provider,
    api_version,
):
    system_prompt = textwrap.dedent(
        """
        You are a strict task router for a scientific toolkit.
        Map user request into exactly one task:
        - remapping
        - streamline
        - pathline

        Return ONLY JSON with keys:
        {
          "task": "remapping|streamline|pathline",
          "confidence": 0.0_to_1.0,
          "reason": "short reason"
        }

        Rules:
        - If intent is about remap, regrid, projection to image, use remapping.
        - If intent is about flow lines in one period or one snapshot, use streamline.
        - If intent is about time-evolution trajectory across months, use pathline.
        - Always pick one.
        """
    ).strip()

    content = call_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request},
        ],
        model=model,
        temperature=float(temperature),
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
        api_version=api_version,
    )

    parsed = _extract_first_json_object(content)
    if parsed is None:
        raise RuntimeError(f"LLM did not return parseable JSON: {content}")

    task = _normalize_task(str(parsed.get("task", "")))
    if task is None:
        raise RuntimeError(f"LLM returned unknown task: {parsed.get('task')}")

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(parsed.get("reason", "")).strip() or "No reason provided"
    return RouteResult(task=task, confidence=confidence, reason=reason)


def write_generated_job(output_dir, task, script_text):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = output_dir / "job_{0}_{1}.py".format(task, ts)
    with open(str(path), "w") as f:
        f.write(script_text)
    return path


def write_task_config(output_dir, task, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = output_dir / "config_{0}_{1}.json".format(task, ts)
    with open(str(path), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path


def fallback_route(user_request):
    req = user_request.lower()
    if any(k in req for k in ["remap", "regrid", "grid", "mapping", "投影", "重映射"]):
        return RouteResult("remapping", 0.35, "keyword fallback: remap/regrid")
    if any(k in req for k in ["stream", "流线", "snapshot"]):
        return RouteResult("streamline", 0.35, "keyword fallback: streamline")
    if any(k in req for k in ["pathline", "trajectory", "轨迹", "月份", "time-evolution"]):
        return RouteResult("pathline", 0.35, "keyword fallback: pathline")
    return RouteResult("pathline", 0.2, "default fallback")


def run_script(script_path, python_exe, dry_run=False):
    cmd = [python_exe, str(script_path)]
    print(f"[Agent] command: {' '.join(cmd)}")

    if dry_run:
        print("[Agent] dry-run enabled; script not executed.")
        return 0

    completed = subprocess.run(cmd, cwd=str(script_path.parent), check=False)
    return int(completed.returncode)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-based task agent for MOPS tutorial examples"
    )
    parser.add_argument(
        "--model",
        default="",
        help="LLM model/deployment name. If omitted, auto-read from env FOUNDRY_MODEL/AZURE_OPENAI_DEPLOYMENT/OPENAI_MODEL.",
    )
    parser.add_argument("--request", required=True, help="User request in natural language")
    parser.add_argument(
        "--base-url",
        default="",
        help="LLM endpoint base URL. Supports OpenAI-compatible and Foundry endpoints.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key. Can also be set via env vars OPENAI_API_KEY/FOUNDRY_API_KEY/AZURE_OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(SUPPORTED_PROVIDERS),
        default="auto",
        help="LLM provider selection: auto/openai/foundry",
    )
    parser.add_argument(
        "--api-version",
        default=os.environ.get("FOUNDRY_API_VERSION", "2024-05-01-preview"),
        help="API version for Foundry/Azure endpoints",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for routing",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=60,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--task",
        choices=sorted(SUPPORTED_TASKS),
        default=None,
        help="Optional manual override. If set, skip LLM routing.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for generated job scripts (default: Agent/generated)",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable for running tutorial task scripts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only route and print selected command; do not execute",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow keyword fallback when LLM is unavailable (default is AI-only strict mode)",
    )
    parser.add_argument(
        "--data-folder",
        default="",
        help="Data folder path (for auto-YAML generation if no yaml-path provided)",
    )
    parser.add_argument(
        "--yaml-path",
        default="",
        help="Explicit YAML config file path (overrides LLM extraction)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strict_llm = not args.allow_fallback

    resolved_model = _pick_first_non_empty(
        args.model,
        os.environ.get("FOUNDRY_MODEL"),
        os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        os.environ.get("OPENAI_MODEL"),
        "test-model",
    )

    provider, resolved_base_url, resolved_api_key = resolve_provider_and_credentials(args)

    if not resolved_base_url:
        if provider == "foundry":
            print("[Agent] missing base URL. Set --base-url or FOUNDRY_BASE_URL/AZURE_INFERENCE_ENDPOINT.")
        else:
            print("[Agent] missing base URL. Set --base-url or OPENAI_BASE_URL.")
    if not resolved_api_key:
        if provider == "foundry":
            print("[Agent] missing API key. Set --api-key or FOUNDRY_API_KEY/AZURE_OPENAI_API_KEY.")
        else:
            print("[Agent] missing API key. Set --api-key or OPENAI_API_KEY.")

    print("[Agent] provider: {0}".format(provider))
    if resolved_base_url:
        print("[Agent] endpoint: {0}".format(build_chat_endpoint(resolved_base_url, provider, args.api_version, resolved_model)))

    repo_root = Path(__file__).resolve().parent.parent
    default_output_dir = repo_root / "Agent" / "generated"
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir

    if strict_llm and (not resolved_base_url or not resolved_api_key):
        print("[Agent] AI-only mode: missing endpoint or key, aborting.")
        return 2

    if args.task:
        route = RouteResult(task=args.task, confidence=1.0, reason="manual override")
    else:
        if not resolved_api_key or not resolved_base_url:
            print("[Agent] LLM credentials unavailable, using keyword fallback router.")
            route = fallback_route(args.request)
        else:
            try:
                route = route_task_with_llm(
                    user_request=args.request,
                    model=resolved_model,
                    api_key=resolved_api_key,
                    base_url=resolved_base_url,
                    temperature=args.temperature,
                    timeout_sec=args.timeout_sec,
                    provider=provider,
                    api_version=args.api_version,
                )
            except Exception as e:
                if strict_llm:
                    print("[Agent] AI-only mode: routing failed, aborting.")
                    raise
                print(f"[Agent] LLM routing failed: {e}")
                print("[Agent] Falling back to keyword router.")
                route = fallback_route(args.request)

    print(f"[Agent] selected task: {route.task}")
    print(f"[Agent] confidence: {route.confidence:.2f}")
    print(f"[Agent] reason: {route.reason}")

    task_config = build_task_config(
        task=route.task,
        user_request=args.request,
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        timeout_sec=args.timeout_sec,
        provider=provider,
        api_version=args.api_version,
        strict_llm=strict_llm,
        output_dir=output_dir,
    )

    # Override with explicit CLI parameters if provided
    if args.data_folder and route.task == "remapping":
        print(f"[Agent] Using data folder from CLI: {args.data_folder}")
        # Generate YAML from data folder
        try:
            import yaml
            yaml_config = generate_remapping_yaml_config(args.data_folder)
            if yaml_config:
                output_dir.mkdir(parents=True, exist_ok=True)
                yaml_path = output_dir / "auto_generated_cli.yaml"
                with open(str(yaml_path), 'w') as f:
                    yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
                task_config["yaml_path"] = str(yaml_path)
                print(f"[Agent] Generated YAML from CLI data folder: {yaml_path}")
        except Exception as e:
            print(f"[Agent] Warning: YAML generation from CLI data folder failed: {e}")

    if args.yaml_path:
        print(f"[Agent] Using YAML path from CLI: {args.yaml_path}")
        task_config["yaml_path"] = args.yaml_path

    config_path = ""
    if task_config:
        cfg_file = write_task_config(output_dir, route.task, task_config)
        config_path = str(cfg_file)
        print("[Agent] generated config: {0}".format(config_path))

    job_text = render_job_script(route.task, args.request, config_path=config_path)
    job_path = write_generated_job(output_dir, route.task, job_text)
    print(f"[Agent] generated job: {job_path}")

    rc = run_script(job_path, python_exe=args.python_exe, dry_run=args.dry_run)
    if rc == 0:
        print("[Agent] done")
    else:
        print(f"[Agent] task failed with code {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
