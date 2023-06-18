import numpy as np
import pygalmesh
import meshio
import torch
from mesh import Mesh
from sklearn.cluster import KMeans
from model import Model
from utils import pc_normalize


INPUT_PARTS_COUNT = 3

mesh = Mesh()
mesh.surftovol('teddy.obj')
coms = mesh.getCOM()
print(coms.shape)

kmeans = KMeans(n_clusters=512, max_iter=1000).fit(coms)

model = Model()
model.load_state_dict(torch.load('./params/model_66.pt', map_location=torch.device('cpu')))

x, _, _ = pc_normalize(torch.tensor(kmeans.cluster_centers_).float().unsqueeze(0))
sim, result = model(x, torch.tensor([INPUT_PARTS_COUNT]))

sim = sim[0]
result = result[0]

# make adjacency matrix "hard"
sim[sim < 0.9] = 0
sim[sim > 0] = 1
sim = sim.long()

# assign class number for arbitrarily order
pred = np.ones(512, dtype='long') * -1
for i in range(len(pred)):
    if pred[i] == -1:
        label = pred.max() + 1
        for j in range(len(pred)):
            if i == j or sim[i, j] == 1:
                pred[j] = label

result = result.argmax(dim=1).numpy()

print(pred)
print(len(set(pred)))

mesh.mergemesh(pred[kmeans.labels_])