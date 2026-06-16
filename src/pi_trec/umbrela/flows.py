from __future__ import annotations

import asyncio
from typing import Any

from pi_trec.config import UmbrelaJudgeConfig
from pi_trec.jsonl import append_jsonl, read_jsonl
from pi_trec.runner import run_prompt
from pi_trec.umbrela.prompts import parse_umbrela_judgment
from pi_trec.umbrela.stages import iter_prompt_tasks


async def judge(config: UmbrelaJudgeConfig) -> None:
    if config.overwrite and config.output_file.exists():
        config.output_file.unlink()
    if config.overwrite and config.failed_output and config.failed_output.exists():
        config.failed_output.unlink()
    tasks = iter_prompt_tasks(list(read_jsonl(config.input_file)), prompt_type=config.prompt_type)
    if config.limit is not None:
        tasks = tasks[: config.limit]
    agent_config = config.local_agent_config()
    raw_events_dir = config.raw_events_dir or config.output_file.parent / "raw-events" / config.output_file.stem
    semaphore = asyncio.Semaphore(max(1, config.max_concurrency))

    async def one(task: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            result = await run_prompt(
                task_id=task["task_id"],
                evaluator="umbrela",
                instruction=task["instruction"],
                raw_events_dir=raw_events_dir,
                config=agent_config,
                metadata=task["metadata"],
            )
            metadata = task["metadata"]
            judgment = parse_umbrela_judgment(result["output_text"]) if result["status"] == "completed" else None
            row = {
                "query": metadata["query"],
                "passage": metadata["passage"],
                "judgment": -1 if judgment is None else judgment,
            }
            if metadata.get("qid") is not None:
                row["qid"] = metadata["qid"]
            if metadata.get("docid") is not None:
                row["docid"] = metadata["docid"]
            if config.include_trace:
                row.update(
                    {
                        "task_id": task["task_id"],
                        "prompt": None if config.redact_prompts else task["instruction"],
                        "prediction": result["output_text"],
                        "result_status": result["status"],
                        "error": result["error"],
                    }
                )
            return row if result["status"] == "completed" and judgment is not None else {**row, "result_status": "failed", "error": result["error"]}

    for future in asyncio.as_completed([asyncio.create_task(one(task)) for task in tasks]):
        row = await future
        append_jsonl(config.output_file, row)
    print(f"processed={len(tasks)} output={config.output_file} raw_events_dir={raw_events_dir}")
