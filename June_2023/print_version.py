import sys
python_version = sys.version.split(" ")[0]
print("Python version: {}".format(python_version))
import pandas as pd
print("Pandas version: {}".format(pd.__version__))
import numpy as np
print("NumPy version: {}".format(np.__version__))
import scipy as sp
print("SciPy version: {}".format(sp.__version__))
import sklearn
print("Scikit-learn version: {}".format(sklearn.__version__))
import matplotlib
print("Matplotlib version: {}".format(matplotlib.__version__))