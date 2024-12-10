
import json
import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_model.predict.predict import *
from data_engineering.data import *

# Load the cleaned data
df = GetCleanedData()
print(df.columns)