import pygalmesh
import meshio
import numpy as np

class Mesh:
    def __init__(self):
        self.vtxs = []
        self.cells = []
        self.tetra_idx = []

    def surftovol(self, obj_file_name):
        '''surface to volume -> get COM
        generate_volume_mesh_from_surface_mesh에서 
            max_radius_surface_delaunay_ball: 표면에서 Delaunay ball의 최대 반지름을 지정
            max_facet_distance: 생성된 면의 최대 거리 제어
        조정해서 쪼개는 정도 설정하기
        '''
        mesh = pygalmesh.generate_volume_mesh_from_surface_mesh(
            obj_file_name,
            max_radius_surface_delaunay_ball=1.0,
            max_facet_distance=0.3,
            max_circumradius_edge_ratio=3.0,
            verbose=False,
            reorient=True
        )

        self.vtxs = mesh.points
        self.cells = mesh.get_cells_type("tetra")
    
    def getCOM(self):
        '''get center of mass for every tetrahedra
        '''
        COM = np.zeros((len(self.cells), 3))
        for i, cell in enumerate(self.cells):
            vertices = self.vtxs[cell]
            com = np.mean(vertices, axis=0)
            COM[i] = com

        return COM
    
    def maketetra_idx(self):
        '''make tetra
        cell로 구해진 사면체의 정점 정보를 이용해 combination으로 face 정보 구성
        index version) face 정보를 작성할 때 정점말고 index로 작성한 버전
                       (이후 merge에서 face 비교를 간단히 하기 위해)
        '''
        tetra_idx = []
        combinations = [(2, 1, 0), (1, 3, 0), (3, 2, 0), (2, 3, 1)]

        for cell in self.cells:
            tetra = []
            for comb in combinations:
                face_idx = [cell[idx] for idx in comb]
                tetra.append(face_idx)
            tetra_idx.append(tetra)

        return tetra_idx

    def maketetra(self):
        '''
        cell로 구해진 사면체의 정점 정보를 이용해 combination으로 face 정보 구성
        필요한 경우에만 쓰도록 분리, index version에서 같이 구해도 됨
        '''
        tetrahedra = []
        combinations = [(2, 1, 0), (1, 3, 0), (3, 2, 0), (2, 3, 1)]

        for cell in self.cells:
            tmp_poly = [self.vtxs[cell[0]], self.vtxs[cell[1]], self.vtxs[cell[2]], self.vtxs[cell[3]]]
            faces = []
            for comb in combinations:
                face = [tmp_poly[idx] for idx in comb]
                faces.append(face)
            tetrahedra.append(faces)

        return tetrahedra
    
    def mergemesh(self, group_id=None):
        '''
        같은 group(class)인지 체크해서 remove list 작성
        cells(face) 정보에서 값을 지워줘야 할 것 (result에 저장)
        '''
        self.tetra_idx = self.maketetra_idx()

        if group_id is None:         # group(class) 정보는 받아올 예정 (없으면 임시로 같은걸로 설정)
            group_id = [0] * len(self.tetra_idx)
        
        group_id = np.array(group_id)
        num_group = group_id.max() + 1
        
        for num in range(num_group):
            piece = np.where(group_id == num)[0]
            if piece.shape[0] == 0:
                continue
            
            tmp_list = [self.tetra_idx[i] for i in piece]
            rows = len(piece)
            cols = len(tmp_list[0])
            rm_list = [[False for _ in range(cols)] for _ in range(rows)]

            # merge...
            for i in range(len(piece)):
                tetra_A = self.tetra_idx[piece[i]]
                for j in range(i+1, len(piece)):
                    tetra_B = tmp_list[j]
                    for n in range(len(tetra_A)):
                        for m in range(len(tetra_B)):
                            face_A = tetra_A[n]
                            face_B = tetra_B[m]
                            shared_vertices = set(face_A) & set(face_B)
                            if len(shared_vertices) >= 3:
                                rm_list[i][n] = True
                                rm_list[j][m] = True
            result = []
            for tetra, check_rm in zip(tmp_list, rm_list):
                for face, check in zip(tetra, check_rm):
                    if not check:
                        result.append(face)
                        t = [self.vtxs[idx] for idx in face]

            faces = result
            self.exportmesh(num, result)


    def exportmesh(self, idx, faces):
        '''
        조각들을 obj로 나눠서 export (only surface)
        '''
        output = meshio.Mesh(points=self.vtxs, cells=[("triangle", faces)])
        meshio.write(f"result{idx}.obj", output, file_format="obj")      #성공
    