import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import critical_temp.csv as a pandas 
data = pd.read_csv('critical_temp.csv')

# convert pandas to numpy array
df = data.to_numpy()

df_filtered = df[(df['ml'] == 38) & (df['order'] == 1)]

# Plot the data
plt.scatter(df_filtered['mu'], df_filtered['Tc'])
plt.xlabel('mu')
plt.ylabel('Tc')
plt.title('Critical temperature vs. Chemical potential for ml=38, order=1')
plt.show()