"""Typed configuration objects for every pi-trec subcommand.

Single root of the package import graph: depends only on the standard library
(plus PyYAML, imported lazily). Configs merge dataclass defaults, an optional
YAML file, and explicit CLI flags (later sources win).
"""

from __future__ import annotations

import dataclasses
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Union, get_args, get_origin, get_type_hints

DEFAULT_MODEL = "openai-codex/gpt-5.5"
DEFAULT_THINKING = "medium"
DEFAULT_PROVIDER = "pi"
DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_SYSTEM_PROMPT = ""
DEFAULT_SEED = 13

DEFAULT_WINDOW_SIZE = 10
DEFAULT_MAX_TRIALS = 4


@dataclass(frozen=True)
class LocalAgentConfig:
    """Execution settings for a single Pi local-agent invocation."""

    agent_binary: str = "pi"
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    thinking: str = DEFAULT_THINKING
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    agent_state_dir: Path | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    extension_path: Path | None = None
    extension_cwd: Path | None = None
    extension_env: dict[str, str] | None = None


@dataclass(frozen=True)
class PyseriniWrapperConfig:
    """Settings for the Pyserini HTTP wrapper request handlers."""

    pyserini_base_url: str
    pyserini_index: str
    backend_id: str = "pyserini-http"
    default_limit: int = 10
    max_page_size: int = 100
    read_limit: int = 200
    search_word_limit: int = 512
    read_word_limit: int = 4096
    token_env: str = "PYSERINI_API_TOKEN"


def _unwrap_optional(hint: Any) -> Any:
    """Return ``T`` for ``Optional[T]`` / ``T | None``; otherwise ``hint``."""
    if get_origin(hint) in (Union, types.UnionType):
        non_none = [arg for arg in get_args(hint) if arg is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return hint


def _coerce(hint: Any, value: Any, *, field_name: str) -> Any:
    """Coerce a YAML/CLI value into the field's declared type."""
    if value is None:
        return None
    if field_name == "extension_env":
        return dict(value) if isinstance(value, list) else value
    base = _unwrap_optional(hint)
    if base is Path:
        return value if isinstance(value, Path) else Path(value)
    if get_origin(base) is list:
        args = get_args(base)
        if args and args[0] is Path:
            return [item if isinstance(item, Path) else Path(item) for item in value]
        return list(value)
    return value


class BaseConfig:
    """Shared construction and validation behavior for config dataclasses."""

    _required: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaseConfig":
        """Build a config from a mapping, coercing types and ignoring unknown keys."""
        hints = get_type_hints(cls)
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {
            key: _coerce(hints[key], value, field_name=key)
            for key, value in data.items()
            if key in field_names
        }
        return cls(**kwargs)

    @classmethod
    def from_sources(
        cls,
        *,
        file_data: Mapping[str, Any] | None = None,
        cli_overrides: Mapping[str, Any],
    ) -> "BaseConfig":
        """Merge defaults < YAML file < explicit CLI flags into a config."""
        merged: dict[str, Any] = {}
        if file_data:
            merged.update(file_data)
        merged.update(cli_overrides)
        return cls.from_mapping(merged)

    def validate(self) -> None:
        """Raise ``SystemExit`` if any required field is missing."""
        missing = [name for name in self._required if getattr(self, name) is None]
        if missing:
            flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            raise SystemExit(f"missing required configuration: {flags} (set via flag or --config)")


@dataclass
class FileIOConfig(BaseConfig):
    input_file: Path | None = None
    output_file: Path | None = None

    _required: ClassVar[tuple[str, ...]] = ("input_file", "output_file")


@dataclass
class MaterializeUmbrelaConfig(FileIOConfig):
    prompt_type: str = "bing"


@dataclass
class MaterializeNuggetCreateConfig(FileIOConfig):
    max_nuggets: int = 30


@dataclass
class MaterializeNuggetAgenticCreateConfig(FileIOConfig):
    max_nuggets: int = 30


@dataclass
class MaterializeNuggetScoreConfig(FileIOConfig):
    pass


@dataclass
class MaterializeNuggetAssignConfig(FileIOConfig):
    input_json: str | None = None
    assign_mode: str = "support-grade-3"

    _required: ClassVar[tuple[str, ...]] = ("output_file",)

    def validate(self) -> None:
        super().validate()
        if bool(self.input_file) == bool(self.input_json):
            raise SystemExit("nugget assign requires exactly one of --input-file or --input-json")


@dataclass
class MaterializeSupportConfig(FileIOConfig):
    pass


@dataclass
class RunConfig(BaseConfig):
    """Base config for commands that run prompts through the Pi local agent."""

    input_file: Path | None = None
    output_file: Path | None = None
    failed_output: Path | None = None
    raw_events_dir: Path | None = None
    agent_binary: str = "pi"
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    thinking: str = DEFAULT_THINKING
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    agent_state_dir: Path | None = None
    extension_path: Path | None = None
    extension_cwd: Path | None = None
    extension_env: dict[str, str] | None = None
    resume: bool = False
    overwrite: bool = False
    limit: int | None = None
    shuffle: bool = False
    seed: int = DEFAULT_SEED

    _required: ClassVar[tuple[str, ...]] = ("input_file", "output_file")

    def local_agent_config(self) -> LocalAgentConfig:
        """Project the shared agent fields onto a ``LocalAgentConfig``."""
        return LocalAgentConfig(
            agent_binary=self.agent_binary,
            provider=self.provider,
            model=self.model,
            thinking=self.thinking,
            timeout_seconds=self.timeout_seconds,
            agent_state_dir=self.agent_state_dir,
            system_prompt=self.system_prompt,
            extension_path=self.extension_path,
            extension_cwd=self.extension_cwd,
            extension_env=dict(self.extension_env) if self.extension_env else None,
        )


@dataclass
class LocalAgentRunConfig(RunConfig):
    """`run local-agent`: run a materialized task JSONL through Pi."""


@dataclass
class UmbrelaJudgeConfig(RunConfig):
    prompt_type: str = "bing"
    include_trace: bool = False
    redact_prompts: bool = False


@dataclass
class SupportJudgeConfig(RunConfig):
    include_prompt: bool = False


@dataclass
class _NuggetRunConfig(RunConfig):
    window_size: int = DEFAULT_WINDOW_SIZE
    max_trials: int = DEFAULT_MAX_TRIALS
    include_trace: bool = False


@dataclass
class NuggetCreateConfig(_NuggetRunConfig):
    max_nuggets: int = 30


@dataclass
class NuggetAgenticCreateConfig(_NuggetRunConfig):
    max_nuggets: int = 30


@dataclass
class NuggetAssignConfig(_NuggetRunConfig):
    input_json: str | None = None
    assign_mode: str = "support-grade-3"

    _required: ClassVar[tuple[str, ...]] = ("output_file",)

    def validate(self) -> None:
        super().validate()
        if bool(self.input_file) == bool(self.input_json):
            raise SystemExit("nugget assign requires exactly one of --input-file or --input-json")


@dataclass
class PyseriniServeConfig(BaseConfig):
    pyserini_base_url: str | None = None
    pyserini_index: str | None = None
    host: str = "127.0.0.1"
    port: int = 8091
    backend_id: str = "pyserini-http"
    default_limit: int = 10
    max_page_size: int = 100
    read_limit: int = 200
    search_word_limit: int = 512
    read_word_limit: int = 4096
    token_env: str = "PYSERINI_API_TOKEN"
    print_config: bool = False

    _required: ClassVar[tuple[str, ...]] = ("pyserini_base_url", "pyserini_index")

    def wrapper_config(self) -> PyseriniWrapperConfig:
        """Project serve settings onto a ``PyseriniWrapperConfig``."""
        return PyseriniWrapperConfig(
            pyserini_base_url=self.pyserini_base_url,
            pyserini_index=self.pyserini_index,
            backend_id=self.backend_id,
            default_limit=self.default_limit,
            max_page_size=self.max_page_size,
            read_limit=self.read_limit,
            search_word_limit=self.search_word_limit,
            read_word_limit=self.read_word_limit,
            token_env=self.token_env,
        )


def load_config_file(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a flat field-name/value mapping."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: config file must contain a top-level mapping")
    return data


def load_env_file(path: Path = Path(".env")) -> None:
    """Load ``KEY=VALUE`` pairs from a dotenv file, never overriding existing vars."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
