import os
import pickle
import argparse
import numpy as np

from stl import mesh
from scipy.io import loadmat


__DATASETS = [
    'ICL',
    'Fischer',
    'SSM-Tibia'
]


def find_connected_components(triangles):
    """Find connected components in the mesh."""
    graph = {}
    for tri in triangles:
        for i in range(3):
            v1, v2 = tuple(tri[i]), tuple(tri[(i+1)%3])
            if v1 not in graph:
                graph[v1] = []
            if v2 not in graph:
                graph[v2] = []
            graph[v1].append(v2)
            graph[v2].append(v1)

    visited = set()
    components = []

    def dfs(v, component):
        stack = [v]
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            component.append(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    stack.append(neighbor)

    for vertex in graph.keys():
        if vertex not in visited:
            component = []
            dfs(vertex, component)
            components.append(component)

    return components


def prepare_ICL(dir, save_dir):
    femur_folder = 'Femur/'
    tibia_folder = 'Tibia/'
    for femur in os.listdir(os.path.join(dir, femur_folder)):
        if 'R' in femur:
            save_folder = os.path.join(save_dir, 'Femur_R')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        elif 'L' in femur:
            save_folder = os.path.join(save_dir, 'Femur_L')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        else:
            raise ValueError("Bone name does not contain L or R")
        femur_mesh = mesh.Mesh.from_file(os.path.join(dir, femur_folder, femur))
        femur_pc = femur_mesh.vectors.reshape(3 * len(femur_mesh.vectors), 3)
        femur_pc = np.unique(femur_pc, axis=0)
        with open(os.path.join(save_folder, femur.replace('.stl', '.pkl')), 'wb') as f:
            pickle.dump(femur_pc, f)
    for tibia in os.listdir(os.path.join(dir, tibia_folder)):
        if 'R' in tibia:
            save_folder_tibia = os.path.join(save_dir, 'Tibia_R')
            save_folder_fibula = os.path.join(save_dir, 'Fibula_R')
            if not os.path.exists(save_folder_tibia):
                os.makedirs(save_folder_tibia)
            if not os.path.exists(save_folder_fibula):
                os.makedirs(save_folder_fibula)
        elif 'L' in tibia:
            save_folder_tibia = os.path.join(save_dir, 'Tibia_L')
            save_folder_fibula = os.path.join(save_dir, 'Fibula_L')
            if not os.path.exists(save_folder_tibia):
                os.makedirs(save_folder_tibia)
            if not os.path.exists(save_folder_fibula):
                os.makedirs(save_folder_fibula)
        else:
            raise ValueError("Bone name does not contain L or R")
        tibia_mesh = mesh.Mesh.from_file(os.path.join(dir, tibia_folder, tibia))
        components = find_connected_components(tibia_mesh.vectors)
        if len(components) != 2:
            print("More than 2 connected components found in the mesh. Skipping it.")
        else:
            comp_1 = components[0]
            comp_2 = components[1]
            if len(comp_1) > len(comp_2):
                tibia_pc = comp_1
                fibula_pc = comp_2
            else:
                tibia_pc = comp_2
                fibula_pc = comp_1
            tibia_pc = np.unique(tibia_pc, axis=0)
            fibula_pc = np.unique(fibula_pc, axis=0)
            with open(os.path.join(save_folder_tibia, tibia.replace('.stl', '.pkl')), 'wb') as f:
                pickle.dump(tibia_pc, f)
            with open(os.path.join(save_folder_fibula, tibia.replace('.stl', '.pkl')), 'wb') as f:
                pickle.dump(fibula_pc, f)


def prepare_SSM_Tibia(dir, save_dir):
    data_path = os.path.join(dir, 'Segmentation')
    if not os.path.exists(os.path.join(save_dir, 'Tibia_R')):
        os.makedirs(os.path.join(save_dir, 'Tibia_R'))
    if not os.path.exists(os.path.join(save_dir, 'Fibula_R')):
        os.makedirs(os.path.join(save_dir, 'Fibula_R'))
    for case in os.listdir(data_path):
        if 'case' in case:
            files = os.listdir(os.path.join(data_path, case))
            for file in files:
                if 'cortical.stl' in file:
                    tibia_cortical_mesh = mesh.Mesh.from_file(
                        os.path.join(data_path, case, file)
                    )
                    pc = tibia_cortical_mesh.vectors.reshape(3 * len(tibia_cortical_mesh.vectors), 3)
                    pc = np.unique(pc, axis=0)

                    with open(os.path.join(save_dir, 'Tibia_R', file.replace('.stl', '.pkl')), 'wb') as f:
                        pickle.dump(pc, f)
                elif 'fibula' in file:
                    fibula_mesh = mesh.Mesh.from_file(
                        os.path.join(data_path, case, file)
                    )
                    pc = fibula_mesh.vectors.reshape(3 * len(fibula_mesh.vectors), 3)
                    pc = np.unique(pc, axis=0)
                    with open(os.path.join(save_dir, 'Fibula_R', file.replace('.stl', '.pkl')), 'wb') as f:
                        pickle.dump(pc, f)


def prepare_Fischer(dir, save_dir):
    bones_path = os.path.join(dir, 'Bones')
    patients = os.listdir(bones_path)
    for patient in patients:
        if patient.endswith('.mat'):
            mat = loadmat(os.path.join(bones_path, patient))
            for bone in mat['B'].T:
                if not bone[0][0][0] == 'Sacrum':
                    if not os.path.exists(os.path.join(save_dir, bone[0][0][0])):
                        os.makedirs(os.path.join(save_dir, bone[0][0][0]))
                    if len(bone[0][2]) > 0:
                        bone_pc = bone[0][2][0][0][0]
                        with open(os.path.join(save_dir, bone[0][0][0], patient.replace('.mat', '.pkl')), 'wb') as f:
                            pickle.dump(bone_pc, f)
                    else:
                        print("Missing", bone[0][0][0], "for patient", patient.replace('.mat', ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='ICL',
                        help='Dataset name (ICL, Fischer, SSM-Tibia)')
    parser.add_argument('-f', '--dir', type=str, default='data/ICL',
                        help='Path to dataset')
    parser.add_argument('-s', '--save_dir', type=str, default='data/BSE_Human/',
                        help='Path to dataset')
    args = parser.parse_args()

    dataset = args.dataset
    if dataset not in __DATASETS:
        raise ValueError(f"'dataset' must be one of {__DATASETS}, received {repr(dataset)}.")

    if dataset == 'ICL':
        prepare_ICL(args.dir, args.save_dir)
    elif dataset == 'Fischer':
        prepare_Fischer(args.dir, args.save_dir)
    elif dataset == 'SSM-Tibia':
        prepare_SSM_Tibia(args.dir, args.save_dir)
