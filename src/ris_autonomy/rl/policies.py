"""Policy network architecture selection.

``build_policy_net`` returns the SB3 ``policy_kwargs`` dict for the configured
network size.  The actor (``pi``) and value (``vf``) heads are separate MLPs.
"""
from __future__ import annotations

_SIZES = {
    "default": dict(pi=[256, 128], vf=[256, 128]),
    "small": dict(pi=[128, 64], vf=[128, 64]),
    "large": dict(pi=[512, 256], vf=[512, 256]),
}


def build_policy_net(config) -> dict:
    """Return the ``policy_kwargs`` for the configured network architecture.

    Parameters
    ----------
    config:
        Validated configuration (``rl.net_arch`` selects the size).

    Returns
    -------
    dict
        ``{"net_arch": {"pi": [...], "vf": [...]}}``.

    Raises
    ------
    ValueError
        If ``rl.net_arch`` is not a known size.
    """
    name = config.rl.net_arch
    if name not in _SIZES:
        raise ValueError(f"unknown net_arch {name!r}; choose from {sorted(_SIZES)}")
    return {"net_arch": _SIZES[name]}