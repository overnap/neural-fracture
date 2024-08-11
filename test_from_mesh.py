import numpy as np
import pygalmesh
import meshio
import torch
from mesh import Mesh
from sklearn.cluster import KMeans
from model import Model
from utils import normalize


INPUT_PARTS_COUNT = 5

mesh = Mesh()
mesh.surftovol("teddy.obj")
coms = mesh.getCOM()
print(coms.shape)

kmeans = KMeans(n_clusters=512, max_iter=1000).fit(coms)

model = Model().cuda()
model.load_state_dict(
    torch.load(
        "./model_param_for_test.pt",
        # map_location=torch.device("cpu"),
    )
)

x = normalize(torch.tensor(kmeans.cluster_centers_).float().unsqueeze(0)).cuda()
y = model(x, torch.tensor([INPUT_PARTS_COUNT]).cuda())

result = y[0].argmax(1).cpu().numpy()
print(result)
mesh.mergemesh(result[kmeans.labels_])
