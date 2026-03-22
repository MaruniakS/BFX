from __future__ import annotations

from typing import Dict


def add_phase_coefficients(row: Dict[str, object], epsilon: float) -> Dict[str, object]:
    m_pre = float(row["M_pre"])
    m_during = float(row["M_during"])
    m_post = float(row["M_post"])
    background = (m_pre + m_post) / 2.0

    contrast = (m_during - background) / (background + epsilon)
    recovery = abs(m_post - m_pre) / (abs(m_during - m_pre) + epsilon)

    out = dict(row)
    out["contrast_coefficient"] = float(contrast)
    out["recovery_coefficient"] = float(recovery)
    return out
