import numpy as np
import pygalmesh
import meshio
import torch
from mesh import Mesh
from sklearn.cluster import KMeans
from model import Model
from utils import normalize


INPUT_PARTS_COUNT = 12

mesh = Mesh()
mesh.surftovol('teddy.obj')
coms = mesh.getCOM()
print(coms.shape)

kmeans = KMeans(n_clusters=512, max_iter=1000).fit(coms)

model = Model()
model.load_state_dict(torch.load('./model_param_for_test.pt', map_location=torch.device('cpu')))

x = normalize(torch.tensor(kmeans.cluster_centers_).float().unsqueeze(0))
sim, result = model(x, torch.tensor([INPUT_PARTS_COUNT]))

sim = sim[0]
result = result[0]

result = result.argmax(dim=1).numpy()

print(sim)
print(result)
print(len(set(result)))

mesh.mergemesh(result[kmeans.labels_])