import numpy as np
import cv2


def ColorTransfer(source, target):
    # image sizes
    src_h, src_w = source.shape[0], source.shape[1]
    trg_h, trg_w = target.shape[0], target.shape[1]

    RGBtoLMS_matrix = np.array([[0.3811, 0.5783, 0.0402],
                                [0.1967, 0.7244, 0.0782],
                                [0.0241, 0.1288, 0.8444]])

    # source image rgb to lms
    source_lms = np.zeros_like(source, dtype=float)
    for y in range(src_h):
        for x in range(src_w):
            source_lms[y, x] = source[y, x][::-1]  # bgr to rgb
            source_lms[y, x] = np.dot(RGBtoLMS_matrix, source_lms[y, x])
            # log operation
            if source_lms[y, x][0] == 0:
                source_lms[y, x][0] = 1.0
            if source_lms[y, x][1] == 0:
                source_lms[y, x][1] = 1.0
            if source_lms[y, x][2] == 0:
                source_lms[y, x][2] = 1.0

            source_lms[y, x] = np.log10(source_lms[y, x])

    # target image rgb to lms
    target_lms = np.zeros_like(target, dtype=float)
    for y in range(trg_h):
        for x in range(trg_w):
            target_lms[y, x] = target[y, x][::-1]  # bgr to rgb
            target_lms[y, x] = np.dot(RGBtoLMS_matrix, target_lms[y, x])
            # log operation
            if target_lms[y, x][0] == 0:
                target_lms[y, x][0] = 1.0
            if target_lms[y, x][1] == 0:
                target_lms[y, x][1] = 1.0
            if target_lms[y, x][2] == 0:
                target_lms[y, x][2] = 1.0

            target_lms[y, x] = np.log10(target_lms[y, x])

    LAB_t1 = np.array([[(1 / np.sqrt(3)), 0, 0],
                       [0, (1 / np.sqrt(6)), 0],
                       [0, 0, (1 / np.sqrt(2))]])

    LAB_t2 = np.array([[1, 1, 1],
                       [1, 1, -2],
                       [1, -1, 0]])

    LMStoLAB_matrix = np.dot(LAB_t1, LAB_t2)

    # source image lms to lab
    source_lab = np.zeros_like(source, dtype=float)
    src_l, src_a, src_b = [], [], []
    for y in range(src_h):
        for x in range(src_w):
            source_lab[y, x] = np.dot(LMStoLAB_matrix, source_lms[y, x])
            src_l.append(source_lab[y, x][0])
            src_a.append(source_lab[y, x][1])
            src_b.append(source_lab[y, x][2])

    # target image lms to lab
    target_lab = np.zeros_like(target, dtype=float)
    trg_l, trg_a, trg_b = [], [], []
    for y in range(src_h):
        for x in range(src_w):
            target_lab[y, x] = np.dot(LMStoLAB_matrix, target_lms[y, x])
            trg_l.append(target_lab[y, x][0])
            trg_a.append(target_lab[y, x][1])
            trg_b.append(target_lab[y, x][2])

    src_l_mean, src_a_mean, src_b_mean = np.mean(src_l), np.mean(src_a), np.mean(src_b)
    trg_l_mean, trg_a_mean, trg_b_mean = np.mean(trg_l), np.mean(trg_a), np.mean(trg_b)

    src_l_var, src_a_var, src_b_var = np.var(src_l), np.var(src_a), np.var(src_b)
    trg_l_var, trg_a_var, trg_b_var = np.var(trg_l), np.var(trg_a), np.var(trg_b)

    # step 5-6-7
    source_lab_f = np.zeros_like(source, dtype=float)
    for y in range(src_h):
        for x in range(src_w):
            source_lab_f[y, x][0] = ((source_lab[y, x][0] - src_l_mean) * (trg_l_var / src_l_var)) + trg_l_mean
            source_lab_f[y, x][1] = ((source_lab[y, x][1] - src_a_mean) * (trg_a_var / src_a_var)) + trg_a_mean
            source_lab_f[y, x][2] = ((source_lab[y, x][2] - src_b_mean) * (trg_b_var / src_b_var)) + trg_b_mean

    LMS_t1 = np.array([[1, 1, 1],
                       [1, 1, -1],
                       [1, -2, 0]])

    LMS_t2 = np.array([[((np.sqrt(3)) / 3), 0, 0],
                       [0, ((np.sqrt(6)) / 6), 0],
                       [0, 0, ((np.sqrt(2)) / 2)]])

    LABtoLMS_matrix = np.dot(LMS_t1, LMS_t2)

    # source image lab to lms
    source_lms_f = np.zeros_like(source, dtype=float)
    for y in range(src_h):
        for x in range(src_w):
            source_lms_f[y, x] = np.dot(LABtoLMS_matrix, source_lab_f[y, x])
            # go back to linear space
            source_lms_f[y, x][0] = 10 ** source_lms_f[y, x][0]
            source_lms_f[y, x][1] = 10 ** source_lms_f[y, x][1]
            source_lms_f[y, x][2] = 10 ** source_lms_f[y, x][2]
            if source_lms_f[y, x][0] == 1.0:
                source_lms_f[y, x][0] = 0.
            if source_lms_f[y, x][1] == 1.0:
                source_lms_f[y, x][1] = 0.
            if source_lms_f[y, x][2] == 1.0:
                source_lms_f[y, x][2] = 0.

    LMStoRGB_matrix = np.array([[4.4679, -3.5873, 0.1193],
                                [-1.2186, 2.3809, -0.1624],
                                [0.0497, -0.2439, 1.2045]])

    # source image LMS to RGB
    source_rgb = np.zeros_like(source, dtype=float)
    final_image = source.copy()
    for y in range(src_h):
        for x in range(src_w):
            source_rgb[y, x] = np.dot(LMStoRGB_matrix, source_lms_f[y, x])
            # rgb to bgr
            source_rgb[y, x] = source_rgb[y, x][::-1]
            if source_rgb[y, x][0] > 255:
                source_rgb[y, x][0] = 255
            if source_rgb[y, x][0] < 0:
                source_rgb[y, x][0] = 0
            if source_rgb[y, x][1] > 255:
                source_rgb[y, x][1] = 255
            if source_rgb[y, x][1] < 0:
                source_rgb[y, x][1] = 0
            if source_rgb[y, x][2] > 255:
                source_rgb[y, x][2] = 255
            if source_rgb[y, x][2] < 0:
                source_rgb[y, x][2] = 0
            final_image[y, x] = source_rgb[y, x]

    return final_image
