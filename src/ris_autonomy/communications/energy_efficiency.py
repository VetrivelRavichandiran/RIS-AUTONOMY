"""Power and energy-efficiency computations.

Energy efficiency is defined as ``EE = sum_rate / total_power`` in bps/W,
where total power is transmit power plus the RIS element power and the
base-station circuit power.
"""
from __future__ import annotations


def ris_power_w(n_elements: int, power_per_element_w: float) -> float:
    """Total RIS power (W) for ``n_elements`` elements.

    Parameters
    ----------
    n_elements:
        Number of RIS elements.
    power_per_element_w:
        Power consumed by a single element (W).

    Returns
    -------
    float
        Total RIS power in watts.
    """
    return n_elements * power_per_element_w


def total_power_w(
    p_tx_w: float,
    n_elements: int,
    power_per_element_w: float,
    p_circuit_w: float,
) -> float:
    """Total system power (W): transmit + RIS + circuit.

    Parameters
    ----------
    p_tx_w:
        Transmit power (W).
    n_elements:
        Number of RIS elements.
    power_per_element_w:
        Power per RIS element (W).
    p_circuit_w:
        Base-station circuit power (W).

    Returns
    -------
    float
        Total power in watts.
    """
    return p_tx_w + ris_power_w(n_elements, power_per_element_w) + p_circuit_w


def energy_efficiency(sum_rate_bps: float, power_w: float) -> float:
    """Energy efficiency in bps/W.

    Parameters
    ----------
    sum_rate_bps:
        Sum achievable rate (bps).
    power_w:
        Total system power (W).

    Returns
    -------
    float
        ``sum_rate_bps / power_w``.
    """
    return sum_rate_bps / power_w