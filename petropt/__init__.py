"""petropt — The Python package for petroleum engineering.

>>> import petropt
>>> petropt.datasets.load_sample_production()
>>> petropt.correlations.standing_bubble_point(api=35, gas_sg=0.65, temp=200)
>>> petropt.io.read_las("well.las")
"""

__version__ = "0.1.0"

from petropt import correlations, datasets, io, qc, economics
