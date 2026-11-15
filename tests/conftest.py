"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand():
    """Das Handbeispiel der Demo: L = 2, α = 6/5, β = 11/20 (das Preset „Von Hand“), Bedarfsstoß e_0 = 1.

    Von Hand (Brüche): i_1 = −1, u_0 = 0; u_1 = −α i_1 = 6/5; i_2 = i_1 + u_0 = −1; u_2 = −α (i_2 + β u_1) = 51/125; i_3 = i_2 + u_1 = 1/5;
    u_3 = −α (i_3 + β u_2) = −3183/6250; i_4 = i_3 + u_2 = 76/125. Spur 1 − αβ = 17/50, Determinante α (1 − β) = 27/50."""
    return {"L": 2, "alpha": F(6, 5), "beta": F(11, 20), "i": [F(-1), F(-1), F(1, 5), F(76, 125)], "u": [F(0), F(6, 5), F(51, 125), F(-3183, 6250)]}
