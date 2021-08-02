import os
import torch
import cv2
import json
import time
import numpy as np
import math
import torch.nn.functional as F

from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from config import system_configs
from collections import deque

from utils import crop_image, normalize_

from sample.vis import *


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255  # <-- This line altered for grayscale.

    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def pers_transform(img, nx=9, ny=6):
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[190, 720], [582, 457], [701, 457], [1145, 720]])
    offset = [150, 0]
    dst = np.float32(
        [
            src[0] + offset,
            np.array([src[0, 0], 0]) + offset,
            np.array([src[3, 0], 0]) - offset,
            src[3] - offset,
        ]
    )
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # Return the resulting image and matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, M, Minv


def hls_thresh(img, thresh_min=200, thresh_max=255):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 1]

    # Creating image masked in S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
    return s_binary


def sobel_thresh(img, sobel_kernel=3, orient="x", thresh_min=20, thresh_max=100):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == "x":
        sobelx = cv2.Sobel(
            gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel
        )  # Take the derivative in x
        abs_sobelx = np.absolute(
            sobelx
        )  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    else:
        sobely = cv2.Sobel(
            gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel
        )  # Take the derivative in x
        abs_sobely = np.absolute(
            sobely
        )  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    # Creathing img masked in x gradient
    grad_bin = np.zeros_like(scaled_sobel)
    grad_bin[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return grad_bin


def mag_thresh(img, sobel_kernel=3, thresh_min=100, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1

    # Return the binary image
    return binary_output


def dir_thresh(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi / 2):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh_min) & (absgraddir <= thresh_max)] = 1

    # Return the binary image
    return binary_output


def mask_image(image):
    img_y, img_x = (image.shape[0], image.shape[1])
    offset = 50

    source = np.float32(
        [  # MASK
            [img_y - offset, offset],  # bottom left
            [img_y - offset, img_x - offset],  # bottom right
            [offset, offset],  # top left
            [offset, img_x - offset],
        ]
    )  # top right
    masked_image = np.copy(image)
    mask = np.zeros_like(masked_image)
    vertices = np.array([[source[1], source[0], source[2], source[3]]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, [255, 255, 255])
    masked_edges = cv2.bitwise_and(masked_image, mask)
    return masked_edges


def lab_b_channel(img, thresh=(190, 255)):
    # Normalises and thresholds to the B channel
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:, :, 2]
    # Don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b * (255 / np.max(lab_b))
    #  Apply a threshold
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    return binary_output


def window_search(binary_warped):
    # Take a histogram of the bottom half of the image
    bottom_half_y = binary_warped.shape[0] / 2
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2) :, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(
            out_img,
            (win_xleft_low, win_y_low),
            (win_xleft_high, win_y_high),
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            out_img,
            (win_xright_low, win_y_low),
            (win_xright_high, win_y_high),
            (0, 255, 0),
            2,
        )
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        ).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(lefty) == 0 or len(leftx) == 0 or len(righty) == 0 or len(rightx) == 0:
        success = False
    else:
        success = True

    if success:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

    return left_lane_inds, right_lane_inds, out_img, success


class Line:
    def __init__(self, maxSamples=4):

        self.maxSamples = maxSamples
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=self.maxSamples)
        # Polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # Polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # Average x values of the fitted line over the last n iterations
        self.bestx = None
        # Was the line detected in the last iteration?
        self.detected = False
        # Radius of curvature of the line in some units
        self.radius_of_curvature = None
        # Distance in meters of vehicle center from the line
        self.line_base_pos = None

    def update_lane(self, ally, allx):
        # Updates lanes on every new frame
        # Mean x value
        self.bestx = np.mean(allx, axis=0)
        # Fit 2nd order polynomial
        new_fit = np.polyfit(ally, allx, 2)
        # Update current fit
        self.current_fit = new_fit
        # Add the new fit to the queue
        self.recent_xfitted.append(self.current_fit)
        # Use the queue mean as the best fit
        self.best_fit = np.mean(self.recent_xfitted, axis=0)
        # meters per pixel in y dimension
        ym_per_pix = 30 / 720
        # meters per pixel in x dimension
        xm_per_pix = 3.7 / 700
        # Calculate radius of curvature
        fit_cr = np.polyfit(ally * ym_per_pix, allx * xm_per_pix, 2)
        y_eval = np.max(ally)
        self.radius_of_curvature = (
            (1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5
        ) / np.absolute(2 * fit_cr[0])


def margin_search(binary_warped):
    # Performs window search on subsequent frame, given previous frame.
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 30

    left_lane_inds = (
        nonzerox
        > (
            left_line.current_fit[0] * (nonzeroy ** 2)
            + left_line.current_fit[1] * nonzeroy
            + left_line.current_fit[2]
            - margin
        )
    ) & (
        nonzerox
        < (
            left_line.current_fit[0] * (nonzeroy ** 2)
            + left_line.current_fit[1] * nonzeroy
            + left_line.current_fit[2]
            + margin
        )
    )
    right_lane_inds = (
        nonzerox
        > (
            right_line.current_fit[0] * (nonzeroy ** 2)
            + right_line.current_fit[1] * nonzeroy
            + right_line.current_fit[2]
            - margin
        )
    ) & (
        nonzerox
        < (
            right_line.current_fit[0] * (nonzeroy ** 2)
            + right_line.current_fit[1] * nonzeroy
            + right_line.current_fit[2]
            + margin
        )
    )

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(lefty) == 0 or len(leftx) == 0 or len(righty) == 0 or len(rightx) == 0:
        success = False

        return 1, 2, 3, success
    else:
        success = True

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Generate a blank image to draw on
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(out_img)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))]
    )
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx - margin, ploty]))]
    )
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))]
    )
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.intc([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.intc([right_line_pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]

    # Draw polyline on image
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
    cv2.polylines(out_img, [right], False, (1, 1, 0), thickness=5)
    cv2.polylines(out_img, [left], False, (1, 1, 0), thickness=5)

    return left_lane_inds, right_lane_inds, out_img, success


def validate_lane_update(img, left_lane_inds, right_lane_inds):
    # Checks if detected lanes are good enough before updating
    img_size = (img.shape[1], img.shape[0])

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Extract left and right line pixel positions
    left_line_allx = nonzerox[left_lane_inds]
    left_line_ally = nonzeroy[left_lane_inds]
    right_line_allx = nonzerox[right_lane_inds]
    right_line_ally = nonzeroy[right_lane_inds]

    # Discard lane detections that have very little points,
    # as they tend to have unstable results in most cases
    if len(left_line_allx) <= 1800 or len(right_line_allx) <= 1800:
        left_line.detected = False
        right_line.detected = False
        return

    left_x_mean = np.mean(left_line_allx, axis=0)
    right_x_mean = np.mean(right_line_allx, axis=0)
    lane_width = np.subtract(right_x_mean, left_x_mean)

    # Discard the detections if lanes are not in their repective half of their screens
    if left_x_mean > 740 or right_x_mean < 740:
        left_line.detected = False
        right_line.detected = False
        return

    # Discard the detections if the lane width is too large or too small
    if lane_width < 300 or lane_width > 800:
        left_line.detected = False
        right_line.detected = False
        return

    # If this is the first detection or
    # the detection is within the margin of the averaged n last lines
    if (
        left_line.bestx is None
        or np.abs(np.subtract(left_line.bestx, np.mean(left_line_allx, axis=0))) < 100
    ):
        left_line.update_lane(left_line_ally, left_line_allx)
        left_line.detected = True
    else:
        left_line.detected = False
    if (
        right_line.bestx is None
        or np.abs(np.subtract(right_line.bestx, np.mean(right_line_allx, axis=0))) < 100
    ):
        right_line.update_lane(right_line_ally, right_line_allx)
        right_line.detected = True
    else:
        right_line.detected = False

    # Calculate vehicle-lane offset
    xm_per_pix = (
        3.7 / 610
    )  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    car_position = img_size[0] / 2
    l_fit = left_line.current_fit
    r_fit = right_line.current_fit
    left_lane_base_pos = l_fit[0] * img_size[1] ** 2 + l_fit[1] * img_size[1] + l_fit[2]
    right_lane_base_pos = (
        r_fit[0] * img_size[1] ** 2 + r_fit[1] * img_size[1] + r_fit[2]
    )
    lane_center_position = (left_lane_base_pos + right_lane_base_pos) / 2
    left_line.line_base_pos = (car_position - lane_center_position) * xm_per_pix + 0.2
    right_line.line_base_pos = left_line.line_base_pos


def find_lanes(img):
    if (
        left_line.detected and right_line.detected
    ):  # Perform margin search if exists prior success.
        # Margin Search
        left_lane_inds, right_lane_inds, out_img, success = margin_search(img)
        # Update the lane detections
        if success:
            validate_lane_update(img, left_lane_inds, right_lane_inds)

    else:  # Perform a full window search if no prior successful detections.
        # Window Search
        left_lane_inds, right_lane_inds, out_img, success = window_search(img)
        # Update the lane detections
        if success:
            validate_lane_update(img, left_lane_inds, right_lane_inds)
    return out_img, success


left_line = Line()
right_line = Line()


def process_img(img):

    # Undistorting image
    # undist = camera.undistort(img)
    undist = img

    # Masking image
    masked = mask_image(undist)

    # Perspective transform image
    warped, M, Minv = pers_transform(undist)

    # Colour thresholding in S channel
    s_bin = hls_thresh(warped)

    # Colour thresholding in B channel of LAB
    b_bin = lab_b_channel(warped, thresh=(185, 255))

    # Combining both thresholds
    combined = np.zeros_like(s_bin)
    combined[(s_bin == 1) | (b_bin == 1)] = 1

    # Find Lanes
    output_img, success = find_lanes(combined)

    # Draw lanes on image
    # lane_img = draw_lane(undist, combined, Minv)

    # result = assemble_img(warped, combined, output_img, lane_img)

    # return result
    return np.random.randn(1, 7, 9), success


def lane_detection(image):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32),
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25,
    )

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if math.fabs(slope) < 0.5:
                    continue
                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

        min_y = int(image.shape[0] * (3 / 5))
        max_y = int(image.shape[0])

        if len(left_line_y) > 0 and len(left_line_x) > 0:
            poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))

            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
        else:
            left_x_start = left_x_end = 0

        if len(right_line_y) > 0 and len(right_line_x) > 0:
            poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))

            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))

    # return result
    return np.random.randn(1, 7, 9)


def kp_detection(
    db,
    nnet,
    result_dir,
    debug=False,
    evaluator=None,
    repeat=1,
    isEncAttn=False,
    isDecAttn=False,
):
    if db.split != "train":
        db_inds = db.db_inds if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds
    num_images = db_inds.size

    multi_scales = db.configs["test_scales"]

    input_size = db.configs["input_size"]  # [h w]

    total_t = 0
    n = 0
    for ind in tqdm(range(0, num_images), ncols=67, desc="locating kps"):
        db_ind = db_inds[ind]
        # image_id      = db.image_ids(db_ind)
        image_file = db.image_file(db_ind)
        image = cv2.imread(image_file)
        raw_img = image.copy()

        t0 = time.time()
        lanes, success = process_img(raw_img)
        t = time.time() - t0

        if success:
            total_t += t
            n += 1

        if evaluator is not None:
            evaluator.add_prediction(ind, lanes, t / repeat)

    if not debug:
        exp_name = "opencv"
        evaluator.exp_name = exp_name
        eval_str, _ = evaluator.eval(label="{}".format(os.path.basename(exp_name)))
        print(eval_str)

    print(f"fps: {n / total_t}")

    return 0


def testing(
    db,
    nnet,
    result_dir,
    debug=False,
    evaluator=None,
    repeat=1,
    debugEnc=False,
    debugDec=False,
):
    return globals()[system_configs.sampling_function](
        db,
        nnet,
        result_dir,
        debug=debug,
        evaluator=evaluator,
        repeat=repeat,
        isEncAttn=debugEnc,
        isDecAttn=debugDec,
    )
