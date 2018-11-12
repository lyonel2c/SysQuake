import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans



plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

data = pd.read_csv('locations_and_depth.csv',usecols=[0,1])
print(data.shape)
print(data.head())

f1 = data['lat'].values
f2 = data['lon'].values

X = np.array(list(zip(f1,f2)))
plt.scatter(f1,f2,c='black',s=7)

plt.show()

Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k,random_state=1)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)
    print ("k:",k, " cost:", km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


## Acording to Elbow Method the optimal k sholud be 3.
## Note: I'm not sure if these values have special deal because are coordinates


kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
lables = kmeans.predict(X)
centroids = kmeans.cluster_centers_

print("The final Centroids with k=3 are:")
print(centroids)

