import numpy as np
import pandas as pd

t = 1359

num_id = pd.read_csv('tensor_id.csv')

allocation_id = []
for i in range(t):
    allocation_id.append(0)

print(len(allocation_id))
for i in range(len(num_id)):
    allocation_id[num_id.iat[i,0]] = 1

print(allocation_id)
