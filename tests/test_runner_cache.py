from pathlib import Path

from pi_trec.config import LocalAgentConfig
from pi_trec.runner import (
    CACHE_TTL_SECONDS,
    build_agent_args,
    cache_key,
    extract_usage,
    read_cache,
    write_cache,
)


def test_extract_usage_anthropic_and_openai_shapes() -> None:
    anthropic = [{"type": "message_end", "message": {"usage": {"input_tokens": 10, "output_tokens": 5}}}]
    assert extract_usage(anthropic) == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    openai = [{"usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}}]
    assert extract_usage(openai) == {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10}
    assert extract_usage([{"type": "message_start"}]) == {}


def test_extract_usage_pi_shape_with_cost() -> None:
    # Pi/openai-codex emits bare input/output keys, totalTokens, and a nested USD cost.
    pi = [{"type": "assistant", "message": {"usage": {
        "input": 552, "output": 180, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 732,
        "cost": {"input": 0.000414, "output": 0.00081, "total": 0.001224},
    }}}]
    assert extract_usage(pi) == {
        "input_tokens": 552, "output_tokens": 180, "total_tokens": 732, "cost_usd": 0.001224,
    }


def test_cache_key_depends_on_inputs() -> None:
    base = LocalAgentConfig(model="m", thinking="minimal")
    key = cache_key(base, "prompt")
    assert key == cache_key(base, "prompt")  # stable
    assert key != cache_key(base, "other prompt")
    assert key != cache_key(LocalAgentConfig(model="m2", thinking="minimal"), "prompt")


def test_cache_ttl_fresh_hit_and_stale_prune(tmp_path: Path) -> None:
    cache_path = tmp_path / "entry.json"
    result = {"status": "completed", "output_text": "ok"}
    write_cache(cache_path, result, now=1000.0)

    fresh = read_cache(cache_path, now=1000.0 + 60)
    assert fresh is not None and fresh["output_text"] == "ok"

    stale = read_cache(cache_path, now=1000.0 + CACHE_TTL_SECONDS + 1)
    assert stale is None
    assert not cache_path.exists()  # expired entry is pruned


def test_temperature_emitted_only_when_set() -> None:
    assert "--temperature" not in build_agent_args(model="m", thinking="minimal")
    args = build_agent_args(model="m", thinking="minimal", temperature=0.0)
    assert args[args.index("--temperature") + 1] == "0.0"
