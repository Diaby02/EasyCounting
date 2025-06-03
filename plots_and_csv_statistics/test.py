import os
from matplotlib import pyplot as plt
import yaml
import subprocess
from pathlib import Path
import pandas as pd
import duckdb as db
import numpy as np
import argparse

from os.path import dirname, abspath
rootDirectory = dirname(abspath(__file__))

path_of_interest = os.path.join(rootDirectory,"hm_adapted_loca_64.csv")
df = pd.read_csv(path_of_interest)

df = db.sql("SELECT * FROM df WHERE method = 'otm'").to_df()
df.to_csv(os.path.join(rootDirectory,"new.csv"),index=False)