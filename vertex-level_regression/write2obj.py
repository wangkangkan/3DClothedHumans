import numpy as np
import os

def read_obj(filename):

    faces = []
    vertices = []
    fid = open(filename, "r")
    node_counter = 0
    while True:

        line = fid.readline()
        if line == "":
            break
        while line.endswith("\\"):
            # Remove backslash and concatenate with next line
            line = line[:-1] + fid.readline()
        if line.startswith("v"):
            coord = line.split()
            coord.pop(0)
            node_counter += 1
            vertices.append(np.array([float(c) for c in coord]))

        elif line.startswith("f "):
            fields = line.split()
            fields.pop(0)

            # in some obj faces are defined as -70//-70 -69//-69 -62//-62
            cleaned_fields = []
            for f in fields:
                f = int(f.split("/")[0]) - 1
                if f < 0:
                    f = node_counter + f
                cleaned_fields.append(f)
            faces.append(np.array(cleaned_fields))
    fid.close()
    faces_np = np.row_stack(faces)
    vertices_np = np.row_stack(vertices)

    return vertices_np, faces_np


def write_to_obj(filename, vertices, faces=None):
    if not filename.endswith('obj'):
        filename += '.obj'
    name = filename.split('/')[-1]
    path = filename.strip(name)
    if path == '':
        path = './'
    if not os.path.exists(path):
        os.makedirs(path)
    num = vertices.shape[0]
    if faces is None:
        faces = np.loadtxt('./{:d}face.txt'.format(num), dtype=np.int)
    num_face = faces.shape[0]
    with open(filename, 'w') as f:
        f.write(('v {:f} {:f} {:f}\n'*num).format(*vertices.reshape(-1).tolist()))
        f.write(('f {:d} {:d} {:d}\n'*num_face).format(*faces.reshape(-1).tolist()))

def cal_rotation_matrix(rotation_angle=0, axis='x'):
    cos_value = np.cos(rotation_angle)
    sin_value = np.sin(rotation_angle)
    if axis == 'x':
        rotation_matrix = np.array(
            [
                [1., 0., 0.],
                [0., cos_value, -1*sin_value],
                [0., 1*sin_value, cos_value]
            ]
        )
    elif axis == 'y':
        rotation_matrix = np.array(
            [
                [cos_value, 0., sin_value],
                [0., 1., 0.],
                [-1*sin_value, 0., cos_value]
            ]
    )
    elif axis == 'z':
        rotation_matrix = np.array(
            [
                [cos_value, -1*sin_value, 0],
                [1*sin_value, cos_value, 0.],
                [0., 0., 1.]
            ]
        )

    else:
        print('axis input should in [\'x\', \'y\', \'z\']')

    return rotation_matrix

def mesh_3x(vert, faces):
    faces_list = faces.tolist()
    vert_list = vert.tolist()
    new_vert = []
    new_face = list()
    new_vert.extend(vert_list)
    start = len(vert_list)
    for face in faces_list:
        new_point = (vert[face[0]] + vert[face[1]] + vert[face[2]]) / 3
        new_vert.append([new_point[0], new_point[1], new_point[2]])
        new_face.append([face[0], face[1], start])
        new_face.append([face[0], start, face[2]])
        new_face.append([start, face[1], face[2]])
        start += 1
    new_verts = np.array(new_vert)
    new_faces = np.array(new_face, dtype=int)
    
    return new_verts, new_faces

def cal_distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))


if __name__ == "__main__":
    pass