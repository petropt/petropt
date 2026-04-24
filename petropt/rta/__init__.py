"""Rate Transient Analysis (RTA).

Production-based reservoir characterization without shut-in pressure tests.
Transforms variable-rate, variable-pressure production history into
type-curve coordinates, then extracts reservoir properties via matching
or straight-line analysis.

Methods:
    transforms    — rate-normalized drawdown, material balance time
    type_curves   — Blasingame, Agarwal-Gardner, NPI variables
    fmb           — Flowing Material Balance (OOIP/OGIP from slope)
    linear_flow   — sqrt(t) linear-flow analysis and k*xf extraction

References:
    Blasingame, T.A., McCray, T.L., and Lee, W.J., "Decline Curve Analysis
        for Variable Pressure Drop / Variable Flowrate Systems," SPE 21513,
        1991.
    Agarwal, R.G., Gardner, D.C., Kleinsteiber, S.W., and Fussell, D.D.,
        "Analyzing Well Production Data Using Combined Type Curve and
        Decline Curve Analysis Concepts," SPE 57916, 1999.
    Mattar, L. and Anderson, D.M., "Dynamic Material Balance —
        Oil or Gas-in-Place Without Shut-Ins," Petroleum Society, CIPC
        2005-113.
    Wattenbarger, R.A., El-Banbi, A.H., Villegas, M.E., and Maggard, J.B.,
        "Production Analysis of Linear Flow Into Fractured Tight Gas
        Wells," SPE 39931, 1998.
"""

from petropt.rta.fmb import flowing_material_balance
from petropt.rta.linear_flow import (
    permeability_from_linear_flow,
    sqrt_time_analysis,
)
from petropt.rta.transforms import (
    material_balance_time,
    pressure_normalized_rate,
)
from petropt.rta.type_curves import (
    agarwal_gardner_variables,
    blasingame_variables,
    npi_variables,
)

__all__ = [
    "agarwal_gardner_variables",
    "blasingame_variables",
    "flowing_material_balance",
    "material_balance_time",
    "npi_variables",
    "permeability_from_linear_flow",
    "pressure_normalized_rate",
    "sqrt_time_analysis",
]
