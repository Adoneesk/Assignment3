import pandas as pd
import numpy as np

file_dir = "crime.csv"
crime_df = pd.read_csv(file_dir)
violent_df = crime_df[["ViolentCrimesPerPop"]]
mean = violent_df.mean()
median = violent_df.median()
sd = violent_df.std()
maximum = violent_df.max()
minimum = violent_df.min()

print(f"mean: {mean}\nmedian: {median}\nsd: {sd}\nmaximum: {maximum}\nminimum: {minimum}")

"""
DISTRIBUTION:
- Since mean is 0.44 and median is 0.39,
and 0.44>0.39, then we can conclude that
the distribution is right skewed

OUTLIERS:
- Unless the sample or population size is
notably small, then the median won't be 
directly affected (it is simply the middle
value). On the other hand, the mean is
calculated by using every value in the
dataset. Outliers and overall extreme values
will shift the mean significantly.

"""