# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import h5py
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
import scipy.signal
#import randrot
import math

def visualize_depth_map(depth_map_test, title):
    
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(depth_map_test)
    depth_map_visualize = (depth_map_test - min_value) / (max_value - min_value) * 255
    depth_map_visualize = np.asarray(depth_map_visualize, dtype = 'uint8')
    cv2.imshow(title, depth_map_visualize)
    cv2.waitKey(100)
    
    
def rotation_matrix_generating(angle):
    
    roll = angle[0]
    pitch = angle[1]
    yaw = angle[2]

    print "roll = ", roll
    print "pitch = ", pitch
    print "yaw = ", yaw
    print ""
    
    yawMatrix = np.matrix([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw), math.cos(yaw), 0],
    [0, 0, 1]
    ])
    
    pitchMatrix = np.matrix([
    [math.cos(pitch), 0, math.sin(pitch)],
    [0, 1, 0],
    [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])
    
    R = yawMatrix * pitchMatrix * rollMatrix
    return R
    
cv2.destroyAllWindows()


filepath = 'G:/Johns Hopkins University/Projects/Depth estimation/nyu_depth_v2_labeled.mat'
arrays = {}
f = h5py.File(filepath)



for k, v in f.items():
    print(k)
    if(k == 'depths'):
        depth_data = np.array(v)
#    



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

translation_data = []
rotation_data = []
synthesis_depth_data = []
for j in range(depth_data.shape[0]):
    
    if(j >= 50):
        break
    print(j)
    depth_map_test = depth_data[j]
    
    point_clouds = []
    
    for w in range(depth_map_test.shape[1]):
        for h in range(depth_map_test.shape[0]):
            x = (w - cx_rgb) / fx_rgb * depth_map_test[h, w]
            y = (h - cy_rgb) / fy_rgb * depth_map_test[h, w]
            z = depth_map_test[h, w]
            point_clouds.append([x, y, z])
            
    point_clouds = np.array(point_clouds)
    ## We first will only translate the camera coordinate to generate new view depth image# 
    depth_map_synthesis = np.zeros_like(depth_map_test)
    
    translation = 0.5 * np.random.rand(1, 3) - 0.25
    angle = (1.0 / 9 * 3.14) * np.random.rand(1, 3) - (1.0 / 18 * 3.14)
    rotation = rotation_matrix_generating(angle[0])
    
    for i in range(point_clouds.shape[0]):
        point = point_clouds[i, :]
        point = np.reshape(point, (3, 1))
        translation = np.reshape(translation, (3, 1))
        point = np.matmul(rotation, point) + translation
        
        projected_point = np.matmul(P, point)
        projected_point = projected_point / projected_point[2]
        
        round_u = np.round(projected_point[0])
        round_v = np.round(projected_point[1])
        
        if(round_u >= 0 and round_u < depth_map_test.shape[1] and round_v >= 0 and round_v < depth_map_test.shape[0]):
            if(depth_map_synthesis[int(round_v), int(round_u)] <= 0.0 or point[2] < depth_map_synthesis[int(round_v), int(round_u)]):
                depth_map_synthesis[int(round_v), int(round_u)] = point[2]
    
    search_radius = 4                           
    depth_map_synthesis_filled = np.copy(depth_map_synthesis)                               
    ## Find nearest non-zero depth value as interpolated depth value
    
    
    for w in range(depth_map_synthesis.shape[1]):
        for h in range(depth_map_synthesis.shape[0]):
            ## Not filled in before
            non_zero_found = False
            if(depth_map_synthesis[h, w] <= 0.0):
                ## Searching
                for total in range(1, search_radius + 1):
                    for u in range(0, total + 1):
                        v = total - u
                        
                        for sign_u in [-1, 1]:
                            for sign_v in [-1, 1]:
                                
                                signed_u = sign_u * u
                                signed_v = sign_v * v
                                ## Boundary check
                                if(signed_u + w >= 0 and signed_u + w < depth_map_synthesis.shape[1] and
                                   signed_v + h >= 0 and signed_v + h < depth_map_synthesis.shape[0]):                          
                                    if(depth_map_synthesis[signed_v + h, signed_u + w] > 0.0):
                                        depth_map_synthesis_filled[h, w] = depth_map_synthesis[signed_v + h, signed_u + w]
                                        non_zero_found = True
                                        break
                            if (non_zero_found):
                                break
                        if(non_zero_found):
                            break
                    if(non_zero_found):
                        break
                                    
    visualize_depth_map(depth_map_synthesis_filled, "synthesis nearest filled")
    visualize_depth_map(depth_map_synthesis, "synthesis")
    visualize_depth_map(depth_map_test, "original")
    
    synthesis_depth_data.append(depth_map_synthesis_filled)
    translation_data.append(translation)
    rotation_data.append(rotation)
#    cv2.waitKey()
    #vertex = np.array(point_clouds, dtype = [('x', 'f4'), ('y', 'f4'),('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    #el = PlyElement.describe(vertex, 'vertex')
    #PlyData([el]).write('G:/Johns Hopkins University/Projects/Depth estimation/visualize.ply')

#rotation_data = np.array(rotation_data)
#translation_data = np.array(translation_data)
#synthesis_depth_data = np.array(synthesis_depth_data)
np.save("G:/Johns Hopkins University/Projects/Depth estimation/original_depth_data_affine", depth_data)
np.save("G:/Johns Hopkins University/Projects/Depth estimation/synthesis_depth_data_affine", synthesis_depth_data)
np.save("G:/Johns Hopkins University/Projects/Depth estimation/affine_data_t", translation_data)
np.save("G:/Johns Hopkins University/Projects/Depth estimation/affine_data_r", rotation_data)
