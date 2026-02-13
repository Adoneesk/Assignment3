import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

crime_csv = pd.read_csv("crime.csv")
violent_df = crime_csv[["ViolentCrimesPerPop"]]

plt.hist(violent_df,bins = round(np.sqrt(len(violent_df))))
plt.title("Distribution of Violent Crimes")
plt.xlabel("Violent Crimes")
plt.ylabel("Count")
plt.show()

plt.boxplot(violent_df)
plt.title("Distribution of Violent Crimes")
plt.xlabel("Violent Crimes")
plt.ylabel("Count")
plt.show()

"""

1)
- The histogram suggests that this data is right skewed.

2)
- By observing the median on the "Distribution of Violent
Crimes", we can see that it is closer to the first quartile
than the third. Once again, we conclude that the data is
right skewed.

3)
- When observing the boxplot, it is safe to assume that
there are no outliers. When generating the plot, the matplot 
library automatically places outliers. Here we see none. 

"""