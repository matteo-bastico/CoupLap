import os
import pickle
import random
import argparse
import numpy as np

from model.bse import BoneSideEstimation

_SEED = 42
random.seed = _SEED
np.random.seed = _SEED

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--cross', type=float, default=0.2,
                        help='Fraction of source nodes used for coupled graph building')
    parser.add_argument('-v', '--voxel_size', type=float, default=2,
                        help='Voxel size for point cloud down-sampling')
    parser.add_argument('-c', '--maps', type=int, default=20,
                        help='Number of coupled maps')
    parser.add_argument('-th', '--threshold', type=int, default=13000,
                        help='Max number of points')
    parser.add_argument('-dl', '--dir_left', type=str, default='data/BSE_Human/Femur_L',
                        help='Folder containing samples')
    parser.add_argument('-dr', '--dir_right', type=str, default='data/BSE_Human/Femur_R',
                        help='Folder containing samples')
    parser.add_argument('-m', '--method', type=str, default='spectral',
                        help='Matching method')
    parser.add_argument('-o', '--output_dir', type=str, default='results/Femur',
                        help='Output directory')
    args = parser.parse_args()

    # Check all the valid files
    dir_left = args.dir_left
    dir_right = args.dir_right
    files_left = [(os.path.join(dir_left, file), 'L') for file in sorted(os.listdir(dir_left))]
    files_right = [(os.path.join(dir_right, file), 'R') for file in sorted(os.listdir(dir_right))]
    files = files_right + files_left

    result = {}

    for (source_path, source_side) in files:
        result[source_path] = {}
        print("SOURCE:", source_path)
        with open(source_path, 'rb') as f:
            source = pickle.load(f)
            # Source pre-processing
            if source.shape[0] > args.threshold:
                random_points = np.random.choice(range(len(source)), args.threshold, replace=False)
                source = source[random_points]
        bse = BoneSideEstimation(
            source=source,
            method=args.method,
            voxel_size=args.voxel_size,
            cross=args.cross,
            maps=args.maps
        )
        for (target_path, target_side) in sorted(list(set(files) - {(source_path, source_side)})):
            print("TARGET:", target_path)
            with open(target_path, 'rb') as f:
                target = pickle.load(f)
                # target pre-processing
                if target.shape[0] > args.threshold:
                    random_points = np.random.choice(range(len(target)), args.threshold, replace=False)
                    target = target[random_points]
            res = bse.predict(
                target
            )
            # Same side
            gt = source_side == target_side
            if gt == res:
                print("Correct prediction")
                result[source_path][target_path] = 1
            else:
                print("Wrong prediction")
                result[source_path][target_path] = 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, args.method + '.pkl'), 'wb') as f:
        pickle.dump(result, f)
        print("Saved result file in", os.path.join(args.output_dir, args.method + '.pkl'))
