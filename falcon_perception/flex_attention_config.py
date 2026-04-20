import logging
import threading

import torch


_LOG = logging.getLogger(__name__)

_AUTO_SAFE_SHARED_MEM_LIMIT = 65536
_KERNEL_TUNING_KEYS = ("BLOCK_M", "BLOCK_N", "num_stages")

SAFE_KERNEL_OPTIONS_64 = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}
SAFE_KERNEL_OPTIONS_32 = {"BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1}

_STATE_LOCK = threading.Lock()
_FORCED_SAFE_BY_DEVICE: dict[str, dict] = {}
_AUTO_WARNED_DEVICES: set[str] = set()
_FALLBACK_WARNED_DEVICES: set[str] = set()


def _normalize_device(device: str | torch.device | None) -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device)


def _device_key(device: str | torch.device | None) -> str:
    dev = _normalize_device(device)
    if dev is None:
        return "cpu"
    if dev.index is None:
        return f"cuda:{torch.cuda.current_device()}"
    return f"cuda:{dev.index}"


def _get_device_props(device: str | torch.device | None):
    dev = _normalize_device(device)
    if dev is None:
        return None
    return torch.cuda.get_device_properties(dev)


def _extract_tuning(kernel_options: dict | None) -> dict:
    if not kernel_options:
        return {}
    return {k: kernel_options[k] for k in _KERNEL_TUNING_KEYS if k in kernel_options}


def _tuning_profile_name(tuning: dict) -> str:
    if not tuning:
        return "default"
    return (
        f"BLOCK_M={tuning.get('BLOCK_M', '?')},"
        f"BLOCK_N={tuning.get('BLOCK_N', '?')},"
        f"num_stages={tuning.get('num_stages', '?')}"
    )


def _device_shared_memory_limit(device: str | torch.device | None) -> int | None:
    props = _get_device_props(device)
    if props is None:
        return None
    per_block = int(getattr(props, "shared_memory_per_block", 0) or 0)
    per_block_optin = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
    return max(per_block, per_block_optin)


def is_auto_safe_device(device: str | torch.device | None = None) -> bool:
    shared_mem = _device_shared_memory_limit(device)
    return shared_mem is not None and shared_mem <= _AUTO_SAFE_SHARED_MEM_LIMIT


def _auto_safe_tuning(device: str | torch.device | None = None) -> dict:
    if not is_auto_safe_device(device):
        return {}

    dev_key = _device_key(device)
    with _STATE_LOCK:
        if dev_key not in _AUTO_WARNED_DEVICES:
            props = _get_device_props(device)
            shared_mem = _device_shared_memory_limit(device)
            name = props.name if props is not None else dev_key
            _LOG.warning(
                "Auto-enabling flex-attn safe kernel options on %s "
                "(shared_memory_per_block_limit=%s).",
                name,
                shared_mem,
            )
            _AUTO_WARNED_DEVICES.add(dev_key)

    return dict(SAFE_KERNEL_OPTIONS_64)


def get_forced_safe_tuning(device: str | torch.device | None = None) -> dict:
    dev_key = _device_key(device)
    with _STATE_LOCK:
        return dict(_FORCED_SAFE_BY_DEVICE.get(dev_key, {}))


def resolve_flex_kernel_options(
    *,
    device: str | torch.device | None = None,
    user_kernel_options: dict | None = None,
    force_safe: bool | None = None,
) -> dict | None:
    resolved = dict(user_kernel_options or {})

    if force_safe is True:
        resolved.update(SAFE_KERNEL_OPTIONS_64)
    elif force_safe is None:
        resolved.update(_auto_safe_tuning(device))

    forced_tuning = get_forced_safe_tuning(device)
    if forced_tuning:
        resolved.update(forced_tuning)

    return resolved or None


def mark_runtime_fallback(
    *,
    device: str | torch.device | None,
    applied_kernel_options: dict | None,
) -> None:
    tuning = _extract_tuning(applied_kernel_options)
    if not tuning:
        return

    dev_key = _device_key(device)
    with _STATE_LOCK:
        _FORCED_SAFE_BY_DEVICE[dev_key] = dict(tuning)
        if dev_key not in _FALLBACK_WARNED_DEVICES:
            props = _get_device_props(device)
            name = props.name if props is not None else dev_key
            _LOG.warning(
                "FlexAttention runtime fallback activated on %s with profile [%s].",
                name,
                _tuning_profile_name(tuning),
            )
            _FALLBACK_WARNED_DEVICES.add(dev_key)


def candidate_retry_kernel_options(current_kernel_options: dict | None) -> list[dict]:
    current = dict(current_kernel_options or {})
    current_tuning = _extract_tuning(current)
    candidates: list[dict] = []
    for tuning in (SAFE_KERNEL_OPTIONS_64, SAFE_KERNEL_OPTIONS_32):
        if current_tuning == tuning:
            continue
        merged = dict(current)
        merged.update(tuning)
        candidates.append(merged)
    return candidates


def is_triton_resource_error(exc: BaseException) -> bool:
    resource_terms = (
        "no valid triton configs",
        "out of resource",
        "hardware limit",
        "required:",
    )
    backend_terms = (
        "triton",
        "inductor",
    )

    seen: set[int] = set()
    stack = [exc]
    while stack:
        cur = stack.pop()
        obj_id = id(cur)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        text = f"{type(cur).__name__}: {cur}".lower()
        if any(term in text for term in backend_terms) and any(term in text for term in resource_terms):
            return True

        cause = getattr(cur, "__cause__", None)
        context = getattr(cur, "__context__", None)
        if cause is not None:
            stack.append(cause)
        if context is not None:
            stack.append(context)

    return False