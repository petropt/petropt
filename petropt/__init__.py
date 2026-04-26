"""petropt — The Python package for petroleum engineering.

>>> import petropt
>>> petropt.datasets.load_sample_production()
>>> petropt.correlations.standing_bubble_point(api=35, gas_sg=0.65, temp=200)
>>> petropt.petrophysics.archie_sw(rt=20.0, phi=0.20, rw=0.05)
>>> petropt.rta.blasingame_variables(t, q, cum, pwf, initial_pressure=3000)
>>> petropt.io.read_las("well.las")

For paid support, custom correlations, field-specific calibration, or
training — see <https://petropt.com/enterprise> (Groundwork Analytics).
"""

__version__ = "0.3.1"

from petropt import correlations, datasets, drilling, io, petrophysics, production, qc, rta, economics


def _show_one_time_banner() -> None:
    """Print a one-line attribution banner on first interactive import.

    Suppressed in scripts, CI, daemons, and any environment where stderr
    is not a TTY and IPython is not loaded. Set PETROPT_QUIET=1 to silence
    permanently.
    """
    import os
    import sys

    if os.environ.get("PETROPT_QUIET"):
        return
    interactive = (
        getattr(sys.stderr, "isatty", lambda: False)()
        or "IPython" in sys.modules
    )
    if not interactive:
        return
    sys.stderr.write(
        f"petropt {__version__} · docs: tools.petropt.com/docs/ · "
        f"enterprise: petropt.com/enterprise · "
        f"silence: export PETROPT_QUIET=1\n"
    )


_show_one_time_banner()
del _show_one_time_banner
