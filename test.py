import numpy as np
import pygalmesh
import meshio
import torch
from mesh import Mesh
from sklearn.cluster import KMeans
from model import Model
from utils import normalize


INPUT_PARTS_COUNT = 8
POINT_COUNT = 1024

fs = open("./vertices_1024.obj")
coms = []
while True:
    line = fs.readline()
    if not line:
        break
    coms.append([float(x) for x in line[2:-1].split(" ")])
coms = torch.tensor(coms)
print(coms.shape)

model = Model(point_count=1024).cuda()
model.load_state_dict(
    torch.load(
        "./model_1024.pt",
        # map_location=torch.device("cpu"),
    )
)
model.eval()

x = normalize(coms.unsqueeze(0).cuda())
y = model(x, torch.tensor([INPUT_PARTS_COUNT]).cuda())

y = y[0].argmax(1).cpu().numpy()
imap = {}
result = "["
for x in y:
    if x not in imap.keys():
        imap[x] = len(imap)
    result += str(imap[x]) + ", "
result = result[:-2] + "]"

print(result)
