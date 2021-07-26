import pandas as pd
import matplotlib.pyplot as plt
import os


dataset = pd.DataFrame(columns=['ClassID', 'Frequency'])

paths = os.listdir("gtsrb-german-traffic-sign/Train")

count = 0
for path in paths:
	dataset.loc[count] = [int(path), len(os.listdir("gtsrb-german-traffic-sign/Train"+"/"+path))] 
	count+=1


dat = dataset.sort_values(by=['ClassID'])

dat.plot(x='ClassID', y='Frequency', figsize=(12, 6),  kind='bar', legend=False)
plt.xticks(rotation=0)
plt.xlabel("ClassID")
plt.ylabel("Frequency")
plt.show()