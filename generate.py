import os
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
from glob import glob
from utils import mesh_to_pcd


SAMPLE_GRID_SIZE = 0.025
DEBUG = False

def main():
    objects = glob('./artifact/*') + glob('./everyday/*/*')
    frequency = np.zeros(101)
    
    for obj in tqdm(objects):
        # set the name of the object
        name = obj.replace('\\', '/').split('/')
        name = name[2] if name[1] == 'artifact' else name[2] + '-' + name[3]

        if not os.path.exists('./dataset/' + name):
            os.mkdir('./dataset/' + name)

        # load the mesh and points from it
        points = mesh_to_pcd(obj + '/mode_0/piece_0.obj', SAMPLE_GRID_SIZE)

        # visualize for debug
        if DEBUG:
            print(points.shape)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([pcd])

            pcd.points = o3d.utility.Vector3dVector(points[np.random.randint(points.shape[0], size=256)])
            o3d.visualization.draw_geometries([pcd])

        # save input points (point cloud within the mesh)
        torch.save(torch.from_numpy(points).float(), './dataset/' + name + '/input.pt')

        # save output results (the piece index of each point)
        for case_number, fracturing_case in enumerate(glob(obj + '/fractured_*')):
            indices = np.ones(points.shape[0], dtype='long') * -1

            # record the segment count
            frequency[len(glob(fracturing_case + '/*'))] += 1

            for idx, piece in enumerate(glob(fracturing_case + '/*')):
                mesh = trimesh.load(piece)
                indices[mesh.contains(points)] = idx

            # show points that don't match any index
            if DEBUG and (indices == -1).sum() > 15:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                # pay attention to the white points
                test = indices.astype(float)
                test[test == -1] = test.max() * 2
                test /= test.max()

                pcd.colors = o3d.utility.Vector3dVector(test)
                o3d.visualization.draw_geometries([pcd])
            
            torch.save(torch.from_numpy(indices).long(), './dataset/' + name + '/output_' + str(case_number) + '.pt')
    
    # show the percentile and histogram of segment counts
    if DEBUG:
        percentile = frequency.cumsum()
        percentile = 100*percentile/percentile[-1]
        print(percentile[16], percentile[32], percentile[64])

        plt.bar(np.arange(101), frequency)
        plt.show()


if __name__ == "__main__":
    main()