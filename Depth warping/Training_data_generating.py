# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 10:11:56 2017

@author: DELL1
"""

import cv2
import yaml
import transformations
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from xml.dom import minidom
from lxml import etree
import random

def noisy(image):
    if(len(image.shape) > 2):
        row,col,channel = image.shape
        mean = 0
        gauss = np.random.normal(mean, 0.02,(row,col,channel))
        gauss = gauss.reshape(row,col,channel)
    else:
        row,col = image.shape
        mean = 0
        gauss = np.random.normal(mean, 0.02,(row,col))
        gauss = gauss.reshape(row,col)        
    noisy = image + gauss
    return noisy

cv2.destroyAllWindows()
    
prefix = 'G:/Johns Hopkins University/Projects/Sinus Navigation/Data/'
prefix_seq = prefix + 'seq01/'

# downsampling the image size by 4 x 4
downsampling = 4

## Load endoscope intrinsic matrix     
img_mask = cv2.imread(prefix + 'mask.png', cv2.IMREAD_GRAYSCALE)
img_mask = cv2.resize(img_mask, (img_mask.shape[1] / downsampling, img_mask.shape[0] / downsampling), interpolation = cv2.INTER_CUBIC)

indexes =np.where(img_mask > 130)
start_h = indexes[0].min()
end_h = indexes[0].max()
start_w = indexes[1].min()
end_w = indexes[1].max()

img_mask_shrinked = img_mask[start_h:end_h+2, start_w-2:end_w+2]
np.save(prefix + "mask", img_mask_shrinked)

indexes = [1, 4, 5, 6, 9, 11, 13]
count = 0

training_rotation_matrices = []
training_rotation_matrices_I = []
training_translation_vectors = []
training_translation_vectors_I = []
training_input_color_image_1 = []
training_input_color_image_2 = []
training_mask_imgs_1 =[]
training_mask_imgs_2 = []
training_masked_depth_imgs_1 = []
training_masked_depth_imgs_2 = []
## Load endoscope intrinsic matrix     
## We need to change the intrinsic matrix to allow for downsampling naturally  
stream = open(prefix + "endoscope.yaml", 'r')
doc_endoscope = yaml.load(stream)
intrinsic_data = doc_endoscope['camera_matrix']['data']
intrinsic_matrix = np.zeros((3,4))
for j in range(3):
    for i in range(3):
        intrinsic_matrix[j][i] = intrinsic_data[j * 3 + i] / downsampling

#intrinsic_matrix[0][2] = intrinsic_matrix[0][2] - start_w + 2
#intrinsic_matrix[1][2] = intrinsic_matrix[1][2] - start_h
intrinsic_matrix[2][2] = 1.0

# sequence number
#for m in indexes:
#    print(m)
#    prefix_seq = prefix + 'seq' + ('%02d')%(m) + '/'
#    
#    lists_3D_points = []
#    plydata = PlyData.read(prefix_seq + 'structure.ply')
#    for n in range(plydata['vertex'].count):
#        temp = list(plydata['vertex'][n])
#        temp = temp[:3]
#        temp.append(1.0)
#        lists_3D_points.append(temp)
#        
#        
#    stream = open(prefix_seq + "motion.yml", 'r')
#    doc = yaml.load(stream)
#    keys, values = doc.items()
#    poses = values[1]
#
#    # Read the scale parameters obtained from SfM - CT registration
#    doc = etree.parse(prefix_seq + 'icp-init.xml')
#    ele = doc.find('scale')
#    scale = float(ele.text)
#
##    print("scale: ", scale)
#
#    training_mask_imgs = []
#    training_masked_depth_imgs = []
#    rotation_matrices = []
#    translation_vectors = []
#    sv_images = []
#
#    projection_matrices = []
#    extrinsic_matrices = []
#    projection_matrix = np.zeros((3, 4))
#    for n in range(len(poses)):
#        rigid_transform = transformations.quaternion_matrix([poses[n]['orientation']['w'], poses[n]['orientation']['x'], 
#                                                             poses[n]['orientation']['y'], poses[n]['orientation']['z']])
#        rigid_transform[0][3] = poses[n]['position']['x']
#        rigid_transform[1][3] = poses[n]['position']['y']
#        rigid_transform[2][3] = poses[n]['position']['z']
#
##        rotation_matrix = np.zeros((3, 3), dtype = 'float32')
##        translation_vector = np.zeros((3, 3), dtype = 'float32')
#        
##        rotation_matrix = rigid_transform[:3, :3]
##        translation_vector = rigid_transform[:3, 3]
##        
##        rotation_matrices.append(rotation_matrix)
##        translation_vectors.append(translation_vector)
#        
#        img = cv2.imread(prefix_seq + ('frame%04d')%(n) + '.png')
#        img = cv2.resize(img, (img.shape[1] / downsampling, img.shape[0] / downsampling), interpolation = cv2.INTER_CUBIC)
#        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
#        
#        img_sv = img_hsv[:, :, 1:]
#        img_sv = img_sv[start_h:end_h+2, start_w-2:end_w+2, :]
#        sv_images.append(img_sv)
#        
#        transform = np.asmatrix(rigid_transform)
#        extrinsic_matrices.append(transform)
#    
#        projection_matrix = np.dot(intrinsic_matrix, transform)
#        projection_matrices.append(projection_matrix)
#
#    for i in range(len(projection_matrices)):
#        img = cv2.imread(prefix_seq + ('frame%04d')%(i) + '.png')
#        img = cv2.resize(img, (img.shape[1] / downsampling, img.shape[0] / downsampling), interpolation = cv2.INTER_CUBIC)
#        height, width = img.shape[:2]
#        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
#        
#        projection_matrix = projection_matrices[i]
#        extrinsic_matrix = extrinsic_matrices[i]
#    
#        masked_depth_img = np.zeros((height, width))
#        mask_img = np.zeros((height, width))
#        
#        img_ratio = img_hsv[:, :, 2] / (1.0 + img_hsv[:, :, 1])
#        img_ratio = np.asarray(img_ratio, dtype = np.uint8)
#        
#        threshold, binary_image = cv2.threshold(img_ratio, 2, 255, cv2.THRESH_BINARY)
#        
#        ret, markers = cv2.connectedComponents(binary_image)
#        cv2.imshow("specularity candidates", binary_image)
#        sum_list = []
#        label_list = []
#        for label in range(ret):
#          if (label > 0):
#              sum_temp = (markers == label).sum()
#              label_list.append(label)
#              sum_list.append(sum_temp)
#              
#        for j in range(len(label_list)):
#          if(sum_list[j] > height * width * 0.01):
#              large_region_image = np.asarray(((markers==label_list[j]) * 255), dtype = np.uint8)
#              binary_image = binary_image - large_region_image
#              
#        cv2.imshow("specularity", binary_image)
#        cv2.waitKey(10)
#        
#        ## Get 2D positions of points on this image
#        for j in range(len(lists_3D_points)):
#            point_3D_position = np.asarray(lists_3D_points[j])
#            point_3D_position_camera = np.asarray(extrinsic_matrix).dot(point_3D_position)
#            point_3D_position_camera = point_3D_position_camera / point_3D_position_camera[3]
#            point_3D_position_camera[0:3] = point_3D_position_camera[0:3] * scale
#
#            point_projected_undistorted = np.asarray(projection_matrix).dot(point_3D_position)
#            point_projected_undistorted[0] = point_projected_undistorted[0] / point_projected_undistorted[2]
#            point_projected_undistorted[1] = point_projected_undistorted[1] / point_projected_undistorted[2]
#            round_u = int(round(point_projected_undistorted[0]))
#            round_v = int(round(point_projected_undistorted[1]))
#            # We will treat this point as valid if it is projected onto the mask region
#            if(round_u < width and round_v < height and round_u >= 0 and round_v >= 0):
#                if((img_mask[round_v, round_u] > 220 and binary_image[round_v, round_u] < 130) and (point_3D_position_camera[2] < 200 and point_3D_position_camera[2] > 0.25)):
#                    mask_img[round_v][round_u] = 1.0
#                    masked_depth_img[round_v][round_u] = np.log(4.0 * (point_3D_position_camera[2] - 0.25) + 4.0) / 6.0 #point_3D_position_camera[2]
##                    masked_depth_img[round_v][round_u] = np.log(4.0 * (point_3D_position_camera[2] - 0.25) + 4.0) / 6.0
##                    cv2.circle(img, (round_u, round_v), 1, (0, 255, 0))
#     
#        mask_img = mask_img[start_h:end_h+2, start_w-2:end_w+2]
#        masked_depth_img = masked_depth_img[start_h:end_h+2, start_w-2:end_w+2]
#        img = img[start_h:end_h+2, start_w-2:end_w+2, :]
#
#        training_mask_imgs.append(mask_img)
#        training_masked_depth_imgs.append(masked_depth_img)
##        cv2.imshow("sparse points image", img)
##        cv2.waitKey()
for m in indexes:
    
    training_mask_imgs = []
    training_masked_depth_imgs = []
    rotation_matrices = []
    translation_vectors = []
    training_sv_imgs = []

    print(m)
    prefix_seq = prefix + 'seq' + ('%02d')%(m) + '/'
    lists_3D_points = []
    plydata = PlyData.read(prefix_seq + 'structure.ply')
    for n in range(plydata['vertex'].count):
        temp = list(plydata['vertex'][n])
        temp = temp[:3]
        temp.append(1.0)
        lists_3D_points.append(temp)
        
        
    stream = open(prefix_seq + "motion.yml", 'r')
    doc = yaml.load(stream)
    keys, values = doc.items()
    poses = values[1]

    # Read the scale parameters obtained from SfM - CT registration
    doc = etree.parse(prefix_seq + 'icp-init.xml')
    ele = doc.find('scale')
    scale = float(ele.text)

    print("scale: ", scale)
    projection_matrices = []
    extrinsic_matrices = []
    projection_matrix = np.zeros((3, 4))
    for n in range(len(poses)):
        rigid_transform = transformations.quaternion_matrix([poses[n]['orientation']['w'], poses[n]['orientation']['x'], 
                                                             poses[n]['orientation']['y'], poses[n]['orientation']['z']])
        rigid_transform[0][3] = poses[n]['position']['x']
        rigid_transform[1][3] = poses[n]['position']['y']
        rigid_transform[2][3] = poses[n]['position']['z']
        
        transform = np.asmatrix(rigid_transform)
        extrinsic_matrices.append(transform)
    
        projection_matrix = np.dot(intrinsic_matrix, transform)
        projection_matrices.append(projection_matrix)
    
    point_cloud_contamination_accumulator = np.zeros(len(lists_3D_points))
    # frame number
    for i in range(len(projection_matrices)):
        img = cv2.imread(prefix_seq + ('frame%04d')%(i) + '.png')
        img = cv2.resize(img, (img.shape[1] / downsampling, img.shape[0] / downsampling), interpolation = cv2.INTER_CUBIC)
#        cv2.imshow("original", img)
#        img_v = img_hsv[:, :, 2]
        height, width = img.shape[:2]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        img_ratio = img_hsv[:, :, 2] / (1.0 + img_hsv[:, :, 1])
        img_ratio = np.asarray(img_ratio, dtype = np.uint8)
        
        threshold, binary_image = cv2.threshold(img_ratio, 2, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        binary_image = cv2.dilate(binary_image, kernel)     
        
        ret, markers = cv2.connectedComponents(binary_image)
        cv2.imshow("specularity candidates", binary_image)
        sum_list = []
        label_list = []
        for label in range(ret):
          if (label > 0):
              sum_temp = (markers == label).sum()
              label_list.append(label)
              sum_list.append(sum_temp)
              
        for j in range(len(label_list)):
          if(sum_list[j] > height * width * 0.01):
              large_region_image = np.asarray(((markers==label_list[j]) * 255), dtype = np.uint8)
              binary_image = binary_image - large_region_image
              
        kernel = np.ones((7, 7), np.uint8)
        binary_image = cv2.dilate(binary_image, kernel)
        cv2.imshow("specularity", binary_image)
        cv2.waitKey(10)

        
        print(i)
        projection_matrix = projection_matrices[i]
        extrinsic_matrix = extrinsic_matrices[i]

        img = cv2.bilateralFilter(img, 9, 25, 25)
#        cv2.imshow("filtered", img)
#        cv2.waitKey(1)
        height, width = img.shape[:2]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        sanity_array = []
        ## Get 2D positions of points on this image
        
        ## TODO: One way to get rid of the contaminated sparse points is to track d^2 * V channel to see if it exceeds the setting threshold for a lot of times
        ## If so, we need to remove these points and reproject to get the clearer training data again.
        for j in range(len(lists_3D_points)):
            point_3D_position = np.asarray(lists_3D_points[j])
            point_3D_position_camera = np.asarray(extrinsic_matrix).dot(point_3D_position)
            point_3D_position_camera = point_3D_position_camera / point_3D_position_camera[3]
            point_3D_position_camera[0:3] = point_3D_position_camera[0:3] * scale

            point_projected_undistorted = np.asarray(projection_matrix).dot(point_3D_position)
            point_projected_undistorted[0] = point_projected_undistorted[0] / point_projected_undistorted[2]
            point_projected_undistorted[1] = point_projected_undistorted[1] / point_projected_undistorted[2]
            round_u = int(round(point_projected_undistorted[0]))
            round_v = int(round(point_projected_undistorted[1]))
            # We will treat this point as valid if it is projected onto the mask region
            if(round_u < width and round_v < height and round_u >= 0 and round_v >= 0):
                if(img_mask[round_v, round_u] > 220 and point_3D_position_camera[2] > 0.0):
                    sanity_array.append(point_3D_position_camera[2] * point_3D_position_camera[2] * img_hsv[round_v, round_u, 2])

        ## TODO: Use histogram to rule out all potential contaminated points with large product value
#        plt.hist(sanity_array, np.arange(1000) * 400)
#        plt.show(10)
#        plot_url = py.plot_mpl(numpy_hist, filename='numpy-bins')
        
        hist, bin_edges = np.histogram(sanity_array, bins = np.arange(1000) * 400, density=True)
        histogram_percentage = hist * np.diff(bin_edges)
        histogram_sum = 0
        percentage = 0.05
        ## Let's assume there are 80% points in each frame that are not contaminated 
        for j in range(histogram_percentage.shape[0]):
            histogram_sum = histogram_sum + histogram_percentage[j]
            if(histogram_sum >= percentage):
                sanity_threshold_min = bin_edges[j + 1]
                break
            
        histogram_sum = 0
        for j in range(histogram_percentage.shape[0]):
            histogram_sum = histogram_sum + histogram_percentage[histogram_percentage.shape[0] - 1 - j]
            if(histogram_sum >= percentage):
                sanity_threshold_max = bin_edges[histogram_percentage.shape[0] - 1 - j]
                break
        print(sanity_threshold_min, sanity_threshold_max)
        
        for j in range(len(lists_3D_points)):
            point_3D_position = np.asarray(lists_3D_points[j])
            point_3D_position_camera = np.asarray(extrinsic_matrix).dot(point_3D_position)
            point_3D_position_camera = point_3D_position_camera / point_3D_position_camera[3]
            point_3D_position_camera[0:3] = point_3D_position_camera[0:3] * scale

            point_projected_undistorted = np.asarray(projection_matrix).dot(point_3D_position)
            point_projected_undistorted[0] = point_projected_undistorted[0] / point_projected_undistorted[2]
            point_projected_undistorted[1] = point_projected_undistorted[1] / point_projected_undistorted[2]
            round_u = int(round(point_projected_undistorted[0]))
            round_v = int(round(point_projected_undistorted[1]))
            # We will treat this point as valid if it is projected onto the mask region
            if(round_u < width and round_v < height and round_u >= 0 and round_v >= 0):
                if(img_mask[round_v, round_u] > 220 and point_3D_position_camera[2] > 0.0):
                    if(point_3D_position_camera[2] * point_3D_position_camera[2] * img_hsv[round_v, round_u, 2] <= sanity_threshold_min or \
                       point_3D_position_camera[2] * point_3D_position_camera[2] * img_hsv[round_v, round_u, 2] >= sanity_threshold_max or \
                        binary_image[round_v, round_u] > 130):
                        point_cloud_contamination_accumulator[j] = point_cloud_contamination_accumulator[j] + 1
                        
    contaminated_point_cloud_indexes = []
    for i in range(point_cloud_contamination_accumulator.shape[0]):
        if(point_cloud_contamination_accumulator[i] >= 5):
            contaminated_point_cloud_indexes.append(i)
    print("number of points", len(lists_3D_points))
    print("number of contaminated points", len(contaminated_point_cloud_indexes))

    for i in range(len(projection_matrices)):
        img = cv2.imread(prefix_seq + ('frame%04d')%(i) + '.png')
        img = cv2.resize(img, (img.shape[1] / downsampling, img.shape[0] / downsampling), interpolation = cv2.INTER_CUBIC)
        height, width = img.shape[:2]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        
        print(i)
        projection_matrix = projection_matrices[i]
        extrinsic_matrix = extrinsic_matrices[i]
    
        masked_depth_img = np.zeros((height, width))
        mask_img = np.zeros((height, width))
        
        for j in range(len(lists_3D_points)):
            point_3D_position = np.asarray(lists_3D_points[j])
            point_3D_position_camera = np.asarray(extrinsic_matrix).dot(point_3D_position)
            point_3D_position_camera = point_3D_position_camera / point_3D_position_camera[3]
            point_3D_position_camera[0:3] = point_3D_position_camera[0:3] * scale

            point_projected_undistorted = np.asarray(projection_matrix).dot(point_3D_position)
            point_projected_undistorted[0] = point_projected_undistorted[0] / point_projected_undistorted[2]
            point_projected_undistorted[1] = point_projected_undistorted[1] / point_projected_undistorted[2]
            round_u = int(round(point_projected_undistorted[0]))
            round_v = int(round(point_projected_undistorted[1]))
            # We will treat this point as valid if it is projected onto the mask region
            if(round_u < width and round_v < height and round_u >= 0 and round_v >= 0):
                if(img_mask[round_v, round_u] > 220 and binary_image[round_v, round_u] < 130 and point_3D_position_camera[2] > 0.0):
                    if(j not in contaminated_point_cloud_indexes):
                        mask_img[round_v][round_u] = 1.0
                        masked_depth_img[round_v][round_u] = np.log(4.0 * (point_3D_position_camera[2] - 0.25) + 4.0) / 7.0
#                        cv2.circle(img, (round_u, round_v), 1, (0, 255, 0))
#        cv2.imshow("", img)
        cv2.imshow("masked depth img", masked_depth_img * 80)
        cv2.waitKey(1)
        training_sv_imgs.append(noisy(img_hsv[start_h:end_h+2, start_w-2:end_w+2, 1:3]))
        training_mask_imgs.append(mask_img[start_h:end_h+2, start_w-2:end_w+2])
        training_masked_depth_imgs.append(masked_depth_img[start_h:end_h+2, start_w-2:end_w+2])
            
    stream = open(prefix_seq + "motion.yml", 'r')
    doc = yaml.load(stream)
    keys, values = doc.items()
    poses = values[1]
    for n in range(len(poses)):
        rigid_transform = transformations.quaternion_matrix([poses[n]['orientation']['w'], poses[n]['orientation']['x'], 
                                                             poses[n]['orientation']['y'], poses[n]['orientation']['z']])
        rigid_transform[0][3] = poses[n]['position']['x']
        rigid_transform[1][3] = poses[n]['position']['y']
        rigid_transform[2][3] = poses[n]['position']['z']

#        rotation_matrix = np.zeros((3, 3), dtype = 'float32')
#        translation_vector = np.zeros((3, 3), dtype = 'float32')
        
        rotation_matrix = rigid_transform[:3, :3]
        translation_vector = rigid_transform[:3, 3]
        
        rotation_matrices.append(rotation_matrix)
        translation_vectors.append(translation_vector)

    
#    ## Generate image pairs with corresponding affine tranform matrix for training
    for n in range(0, len(poses) - 7):
        num = np.min([7, len(poses) - n - 7])
        if(num == 7):
            for plus in range(num - 2, num):
                choice = random.randint(0, 1)
                if(choice == 0):
                    rotation_1 = rotation_matrices[n]
                    rotation_2 = rotation_matrices[n + plus]
                    translation_1 = translation_vectors[n] * scale
                    translation_2 = translation_vectors[n + plus] * scale
                    training_masked_depth_imgs_1.append(training_masked_depth_imgs[n])
                    training_masked_depth_imgs_2.append(training_masked_depth_imgs[n + plus])
                    training_mask_imgs_1.append(training_mask_imgs[n])
                    training_mask_imgs_2.append(training_mask_imgs[n + plus])
                    training_input_color_image_1.append(training_sv_imgs[n])
                    training_input_color_image_2.append(training_sv_imgs[n + plus])
                else:
                    rotation_2 = rotation_matrices[n]
                    rotation_1 = rotation_matrices[n + plus]
                    translation_2 = translation_vectors[n] * scale
                    translation_1 = translation_vectors[n + plus] * scale
                    training_masked_depth_imgs_2.append(training_masked_depth_imgs[n])
                    training_masked_depth_imgs_1.append(training_masked_depth_imgs[n + plus])
                    training_mask_imgs_2.append(training_mask_imgs[n])
                    training_mask_imgs_1.append(training_mask_imgs[n + plus])
                    training_input_color_image_2.append(training_sv_imgs[n])
                    training_input_color_image_1.append(training_sv_imgs[n + plus])   

                rotation_2_I = np.transpose(rotation_2)
                R = np.matmul(rotation_1, rotation_2_I)
                P = -np.matmul(np.matmul(rotation_1, rotation_2_I), translation_2) + translation_1
                R_I = np.transpose(R)
                P_I = -np.matmul(R_I, P)

                training_rotation_matrices.append(R)
                training_rotation_matrices_I.append(R_I)
                training_translation_vectors.append(P)
                training_translation_vectors_I.append(P_I)
                
#    for n in range(0, 1):
#        plus = len(poses) - 1
#        rotation_1 = rotation_matrices[n]
#        rotation_2 = rotation_matrices[n + plus]
#        translation_1 = translation_vectors[n] * scale
#        translation_2 = translation_vectors[n + plus] * scale
#        rotation_2_I = np.transpose(rotation_2)
#        
#        R = np.matmul(rotation_1, rotation_2_I)
#        P = -np.matmul(np.matmul(rotation_1, rotation_2_I), translation_2) + translation_1
#        R_I = np.transpose(R)
#        P_I = -np.matmul(R_I, P)
#    
#        training_masked_depth_imgs_1.append(training_masked_depth_imgs[n])
#        training_masked_depth_imgs_2.append(training_masked_depth_imgs[n + plus])
#        training_mask_imgs_1.append(training_mask_imgs[n])
#        training_mask_imgs_2.append(training_mask_imgs[n + plus])
#        training_input_color_image_1.append(training_sv_imgs[n])
#        training_input_color_image_2.append(training_sv_imgs[n + plus])
#        
#        training_rotation_matrices.append(R)
#        training_rotation_matrices_I.append(R_I)
#        training_translation_vectors.append(P)
#        training_translation_vectors_I.append(P_I)
        
#    temp = sv_images[0]
#    cv2.imshow("", temp[:, :, 0])
#    cv2.waitKey(10)
    
indexes = np.array(range(len(training_rotation_matrices)))
random.shuffle(indexes)

training_rotation_matrices = np.array(training_rotation_matrices)
training_rotation_matrices = training_rotation_matrices[indexes]

training_rotation_matrices_I = np.array(training_rotation_matrices_I)
training_rotation_matrices_I = training_rotation_matrices_I[indexes]

training_translation_vectors = np.array(training_translation_vectors)
training_translation_vectors = training_translation_vectors[indexes]

training_translation_vectors_I = np.array(training_translation_vectors_I)
training_translation_vectors_I = training_translation_vectors_I[indexes]

training_input_color_image_1 = np.array(training_input_color_image_1)
training_input_color_image_1 = training_input_color_image_1[indexes]
#
training_input_color_image_2 = np.array(training_input_color_image_2)
training_input_color_image_2 = training_input_color_image_2[indexes]

training_mask_imgs_2 = np.array(training_mask_imgs_2)
training_mask_imgs_2 = training_mask_imgs_2[indexes]

training_mask_imgs_1 = np.array(training_mask_imgs_1)
training_mask_imgs_1 = training_mask_imgs_1[indexes]

training_masked_depth_imgs_1 = np.array(training_masked_depth_imgs_1)
training_masked_depth_imgs_1 = training_masked_depth_imgs_1[indexes]

training_masked_depth_imgs_2 = np.array(training_masked_depth_imgs_2)
training_masked_depth_imgs_2 = training_masked_depth_imgs_2[indexes]




np.save(prefix + "depth estimation R", training_rotation_matrices)
np.save(prefix + "depth estimation R_I", training_rotation_matrices_I)
np.save(prefix + "depth estimation P", training_translation_vectors)
np.save(prefix + "depth estimation P_I", training_translation_vectors_I)
np.save(prefix + "depth estimation sv img_1", training_input_color_image_1)
np.save(prefix + "depth estimation sv img_2", training_input_color_image_2)      
np.save(prefix + "depth estimation mask img_1", training_mask_imgs_1)
np.save(prefix + "depth estimation mask img_2", training_mask_imgs_2)   
np.save(prefix + "depth estimation masked depth img_1", training_masked_depth_imgs_1)
np.save(prefix + "depth estimation masked depth img_2", training_masked_depth_imgs_2)   
  