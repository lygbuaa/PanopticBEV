#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np

#defined in metadata_ortho.bin
g_semantic_names = ['flat.driveable_surface', 'flat.sidewalk', 'static.manmade', 'static.vegetation', 'flat.terrain', 'occlusion', 'human.pedestrian.adult', 'vehicle.car', 'vehicle.truck', 'vehicle.motorcycle']
g_num_stuff = 6
g_num_thing = 4
# (6*stuff + 4*thing)
g_semantic_colours = np.array([
    [255, 255, 255],    # 0, white, driveable_surface
    [96, 64, 64],       # 1, red-gray, sidewalk
    [64, 64, 96],       # 2, blue-gray, manmade
    [64, 96, 64],       # 3, green-gray, vegetation
    [96, 96, 64],       # 4, yellow-gray, terrain
    [128, 128, 128],    # 5, gray, occlusion

    [255, 0, 0],        # 6, red, pedestrian
    [0, 0, 255],        # 7, blue, car
    [0, 255, 0],        # 8, green, truck
    [255, 255, 0],      # 9, yellow, bicycle

    [0, 0, 0],          # 10, black, ignore
], dtype=np.uint8)

if __name__ == '__main__':
    print(g_semantic_colours)
    pred = np.full((10, 10), 5, dtype=np.int32)
    plot = g_semantic_colours[pred]
    print(plot)

