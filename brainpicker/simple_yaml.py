from __future__ import annotations

from pathlib import Path
from typing import Any


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the small YAML subset used by BrainPicker config files.

    This is intentionally not a general YAML parser. It supports nested maps,
    lists of scalars, and lists of maps using two-space indentation.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    parsed, index = _parse_block(lines, 0, 0)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected top-level mapping in {path}")
    return parsed


def _parse_block(lines: list[str], index: int, indent: int) -> tuple[Any, int]:
    container: dict[str, Any] | list[Any] | None = None

    while index < len(lines):
        raw = lines[index]
        if not raw.strip() or raw.lstrip().startswith("#"):
            index += 1
            continue

        current_indent = len(raw) - len(raw.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation on line {index + 1}: {raw}")

        text = raw.strip()
        if text.startswith("- "):
            if container is None:
                container = []
            if not isinstance(container, list):
                raise ValueError(f"Mixed list and mapping on line {index + 1}")
            item_text = text[2:].strip()
            if not item_text:
                child, index = _parse_block(lines, index + 1, indent + 2)
                container.append(child)
                continue
            if ":" in item_text and not _looks_like_scalar(item_text):
                key, value_text = _split_key_value(item_text)
                item: dict[str, Any] = {}
                if value_text:
                    item[key] = _parse_scalar(value_text)
                    index += 1
                else:
                    child, index = _parse_block(lines, index + 1, indent + 2)
                    item[key] = child
                while index < len(lines):
                    next_raw = lines[index]
                    if not next_raw.strip() or next_raw.lstrip().startswith("#"):
                        index += 1
                        continue
                    next_indent = len(next_raw) - len(next_raw.lstrip(" "))
                    if next_indent < indent + 2:
                        break
                    if next_indent != indent + 2 or next_raw.strip().startswith("- "):
                        break
                    key, value_text = _split_key_value(next_raw.strip())
                    if value_text:
                        item[key] = _parse_scalar(value_text)
                        index += 1
                    else:
                        child, index = _parse_block(lines, index + 1, indent + 4)
                        item[key] = child
                container.append(item)
                continue
            container.append(_parse_scalar(item_text))
            index += 1
            continue

        if container is None:
            container = {}
        if not isinstance(container, dict):
            raise ValueError(f"Mixed mapping and list on line {index + 1}")
        key, value_text = _split_key_value(text)
        if value_text:
            container[key] = _parse_scalar(value_text)
            index += 1
        else:
            child, index = _parse_block(lines, index + 1, indent + 2)
            container[key] = child

    return ({} if container is None else container), index


def _split_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"Expected key/value pair: {text}")
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _looks_like_scalar(text: str) -> bool:
    return text.startswith(("http://", "https://"))


def _parse_scalar(value: str) -> Any:
    if value == "[]":
        return []
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")

