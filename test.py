import numpy as np
import pygalmesh
import meshio
import torch
from mesh import Mesh
from sklearn.cluster import KMeans
from model import Model
from utils import normalize, BatchTransform
from glob import glob


INPUT_PARTS_COUNT = float(torch.normal(10.5, 0.5, (1,)))
POINT_COUNT = 256

bcoms = []
fmap = {}
for fname in glob("./tcom/*"):
    if "result" in fname or "out" in fname:
        continue
    fs = open(fname)
    coms = []
    while True:
        line = fs.readline()
        if not line:
            break
        coms.append([float(x) for x in line[:-1].split(" ")])
    if len(coms) == POINT_COUNT:
        fmap[len(bcoms)] = fname.split("/")[2].split(".")[0]
        bcoms.append(coms)
    fs.close()
bcoms = torch.tensor(bcoms)

model = Model(point_count=POINT_COUNT).cuda()
model.load_state_dict(
    torch.load(
        f"./model_{POINT_COUNT}.pt",
        # map_location=torch.device("cpu"),
    )
)
model.eval()

transforms = BatchTransform.Compose(
    [
        normalize,
        BatchTransform.RandomJitter(0.005, 0.02),
        BatchTransform.RandomRotate([0.25, 0.25, 0.25]),
        BatchTransform.RandomScale([0.95, 1.05]),
    ]
)

x = transforms(bcoms.cuda())
y = model(x, torch.tensor([INPUT_PARTS_COUNT]).cuda())

for i in range(y.shape[0]):
    t = y[i].argmax(1).cpu().numpy()
    imap = {}
    result = "["
    for x in t:
        if x not in imap.keys():
            imap[x] = len(imap)
        result += str(imap[x]) + ", "
    result = result[:-2] + "]"

    print(result)
    fs = open("./tcom/result/" + fmap[i] + "_out.txt", mode="w")
    fs.write(result)
    fs.close()
