#!/usr/bin/env python3
"""
Example: Natural Language to Task Generation

This script demonstrates how to use the LLM agent to convert natural language
requests into executable MOPS tasks.

The agent can:
1. Determine task type (remapping/streamline/pathline) from natural language
2. Extract all required parameters (including geographic location inference)
3. Generate and execute the task code
"""

import subprocess
import sys
from pathlib import Path


def run_example(description, request, task_override=None, dry_run=True):
    """Run an example natural language request."""
    print(f"\n{'='*80}")
    print(f"Example: {description}")
    print(f"User request: '{request}'")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable,
        "Agent/llm_task_agent.py",
        "--request", request,
    ]

    if task_override:
        cmd.extend(["--task", task_override])

    if dry_run:
        cmd.append("--dry-run")

    # Add fallback mode for demonstration without LLM credentials
    cmd.append("--allow-fallback")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def main():
    examples = [
        {
            "description": "Remapping with geographic location and depth",
            "request": "I want to visualize the results for a depth of 20 meters near the Gulf of Mexico",
            "task": None,  # Let LLM determine
        },
        {
            "description": "Remapping with specific resolution and region",
            "request": "Create a remapping visualization with width=1201, height=601, depth 15m, covering latitude 20-40 and longitude -100 to -70",
            "task": None,
        },
        {
            "description": "Streamline analysis with location",
            "request": "Generate streamlines near the North Atlantic at 50 meters depth for 7 days",
            "task": None,
        },
        {
            "description": "Pathline trajectory analysis",
            "request": "Analyze particle trajectories from January 2015 to December 2015 in the Pacific Ocean",
            "task": None,
        },
        {
            "description": "Simple remapping request",
            "request": "Show me ocean velocity remapping at 10 meters depth",
            "task": "remapping",  # Manual override
        },
    ]

    print("\nNatural Language Task Generation Examples")
    print("=" * 80)
    print("\nThese examples demonstrate how the AI agent converts natural language")
    print("into structured MOPS tasks with full parameter extraction.\n")

    for i, example in enumerate(examples, 1):
        run_example(
            description=f"{i}. {example['description']}",
            request=example["request"],
            task_override=example.get("task"),
            dry_run=True
        )

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("\nTo run with actual LLM (requires API credentials):")
    print("  export OPENAI_API_KEY=your-key")
    print("  export OPENAI_BASE_URL=https://api.openai.com/v1")
    print("  python3 Agent/llm_task_agent.py --request 'your request'")
    print("\nTo execute (not just dry-run), remove --dry-run flag")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
