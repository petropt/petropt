"""petropt — The Python package for petroleum engineering.

>>> import petropt
>>> petropt.datasets.load_sample_production()
>>> petropt.correlations.standing_bubble_point(api=35, gas_sg=0.65, temp=200)
>>> petropt.petrophysics.archie_sw(rt=20.0, phi=0.20, rw=0.05)
>>> petropt.rta.blasingame_variables(t, q, cum, pwf, initial_pressure=3000)
>>> petropt.io.read_las("well.las")
"""

__version__ = "0.3.0"

from petropt import correlations, datasets, drilling, io, petrophysics, production, qc, rta, economics
