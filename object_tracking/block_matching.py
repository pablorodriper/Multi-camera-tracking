import cv2
import numpy as np
from matplotlib import pyplot as plt
import timeit


def compute_difference_blocks(block1, block2, error_function, best_match):
    if error_function == 'SSD':
        s = np.sum((block1[:, :, 0:3] - block2[:, :, 0:3]) ** 2)
        if s < best_match:
            return s, True
    if error_function == 'SAD':
        s = np.sum(abs(block1[:, :, 0:3] - block2[:, :, 0:3]))
        if s < best_match:
            return s, True
    if error_function == 'MSD':
        s = np.mean(abs(block1 - block2) ** 2)
        if s < best_match:
            return s, True
    return best_match, False


def find_match(i, j, block, image, block_size, search_area, step_size, error_function):
    h, w = image.shape[:2]
    best_match = float('inf')
    best_col = i
    best_row = j
    for row in range(max(0, i-search_area), min(i + search_area+block_size, h-block_size+1), step_size):
        for col in range(max(0, j-search_area), min(j + search_area+block_size, w-block_size+1), step_size):
            block2 = image[row:row+block_size, col:col+block_size]
            best_match, match = compute_difference_blocks(block, block2, error_function, best_match)
            if match:
                best_row = row
                best_col = col

    return best_row-i, best_col-j


def block_matching_optical_flow(image1, image2, block_size=4, search_area=4, step_size=2, error_function='SSD', compensation='Forward'):
    if compensation == 'Backward':
        ref_image = image2
        compare_image = image1
    elif compensation == 'Forward':
        ref_image = image1
        compare_image = image2

    h, w = ref_image.shape[:2]
    optical_flow = np.zeros((image1.shape[1], image1.shape[0], 2))
    start = timeit.default_timer()
    for i in range(0, h-block_size+1):
        for j in range(0, w-block_size+1):
            block1 = ref_image[i:i+block_size, j:j+block_size]
            displ_i, displ_j = find_match(i, j, block1, compare_image, block_size, search_area, step_size, error_function)
            if compensation == 'Forward':
                optical_flow[j:j+block_size, i:i+block_size, 0] = displ_i
                optical_flow[j:j+block_size, i:i+block_size, 1] = displ_j
            elif compensation == 'Backward':
                optical_flow[j+displ_j:j+displ_j + block_size, i+displ_i:i+displ_i + block_size, 0] = -displ_i
                optical_flow[j+displ_j:j+displ_j + block_size, i+displ_i:i+displ_i + block_size, 1] = -displ_j
    finish = timeit.default_timer()
    print('Time: {}'.format(finish-start))

    return np.transpose(optical_flow,(1,0,2))
