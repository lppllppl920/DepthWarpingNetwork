# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 12:08:09 2017

@author: DELL1
"""

import numpy as np
from numpy import matrix
import cv2

def visualize_depth_map(depth_map_test, title):
    
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(depth_map_test)
    depth_map_visualize = (depth_map_test - min_value) / (max_value - min_value) * 255
    depth_map_visualize = np.asarray(depth_map_visualize, dtype = 'uint8')
    cv2.imshow(title, depth_map_visualize)
    cv2.waitKey(100)
   
cv2.destroyAllWindows()

prefix = 'G:/Johns Hopkins University/Projects/Depth estimation/'
depth_data = np.load(prefix + "original_depth_data_affine.npy")
synthesis_depth_data = np.load(prefix + "synthesis_depth_data_affine.npy")
translation_data = np.load(prefix + "affine_data_t.npy")
rotation_data = np.load(prefix + "affine_data_r.npy")

##projection matrix
P = np.zeros((3, 3), dtype = 'float64')

fx_rgb = 5.1885790117450188e+02
fy_rgb = 5.1946961112127485e+02
cx_rgb = 3.2558244941119034e+02
cy_rgb = 2.5373616633400465e+02

P[0,0] = fx_rgb
P[0,2] = cx_rgb
P[1,1] = fy_rgb
P[1,2] = cy_rgb
P[2,2] = 1.0

index = 1

translation_vector = np.reshape(translation_data[index], (3, 1))

translation_vec = matrix(translation_vector)
rotation_mat = matrix(rotation_data[index])
intrinsic_mat = matrix(P)

W = np.array(-intrinsic_mat * rotation_mat.I * translation_vec)
M = np.array(intrinsic_mat * rotation_mat.I * intrinsic_mat.I)

x = np.array(np.arange(depth_data[index].shape[1]), dtype = "float32")
y = np.array(np.arange(depth_data[index].shape[0]), dtype = "float32")
u, v = np.meshgrid(x, y, sparse=False, indexing='xy')


W_2 = np.array(intrinsic_mat * translation_vec)
M_2 = np.array(intrinsic_mat * rotation_mat * intrinsic_mat.I)


z_2 = depth_data[index]
z_1 = synthesis_depth_data[index]


z_2_calculate = W[2, 0] + z_1 * (M[2, 0] * u + M[2, 1] * v + M[2, 2])
u_2 = (z_1 * (M[0, 0] * u + M[0, 1] * v + M[0, 2]) + W[0, 0]) / z_2_calculate
v_2 = (z_1 * (M[1, 0] * u + M[1, 1] * v + M[1, 2]) + W[1, 0]) / z_2_calculate

z_1_calculate = W_2[2, 0] + z_2 * (M_2[2, 0] * u + M_2[2, 1] * v + M_2[2, 2])

#u_2 = (c[0] + z_1 * u) / z_2
#v_2 = (c[1] + z_1 * v) / z_2
#z_1_calculate = z_2 - c[2]


z_2_warped = np.zeros_like(z_1_calculate)

for h in range(z_2_warped.shape[0]):
    for w in range(z_2_warped.shape[1]):
        
        y = v_2[h, w]
        x = u_2[h, w]
        
        x0 = np.floor(u_2[h, w])
        x1 = x0 + 1
        y0 = np.floor(v_2[h, w])
        y1 = y0 + 1
        
        x0 = np.clip(x0, 0, z_2_warped.shape[1] - 1)
        x1 = np.clip(x1, 0, z_2_warped.shape[1] - 1)
        y0 = np.clip(y0, 0, z_2_warped.shape[0] - 1)
        y1 = np.clip(y1 , 0, z_2_warped.shape[0] - 1)
        
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)
        
        
        z_2_warped[h, w] = wa * z_1_calculate[y0, x0] + wb * z_1_calculate[y1, x0] + \
            wc * z_1_calculate[y0, x1] + wd * z_1_calculate[y1, x1]
visualize_depth_map(z_1_calculate, "calculate depth 1") 
visualize_depth_map(z_2_calculate, "calculate depth 2")
visualize_depth_map(z_2_warped, "warped depth")
visualize_depth_map(synthesis_depth_data[index], "synthesis depth")
visualize_depth_map(np.abs(synthesis_depth_data[index] - z_2_warped), "subtract depth")        
