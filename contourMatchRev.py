"""
    This module is designed to split the images of multiple sherds into single images using the
    RGB and depth image of each sherd. This method is based on contour based shape matching. This approach
    prioritizes matching the RGB and Depth image from the Depth mask.
"""
import os
import math
import cv2 as cv
import numpy as np
import re


def first_agr(tup):
    """
    This function returns the first element in the given tuple.

    Args:
        tup: the specified tuple.
    """
    return tup[0]


def contour_match(rbg_contours, depth_contours):
    """
    This function maps the RGB contours to depth images using the shape matching method to compare contours.

    Args:
        rbg_contours: the contours from the rgb image
        depth_contours: the contours from the depth image
    """
    c_results = []

    for j in rbg_contours:
        temp = []

        for d in range(len(depth_contours)):
            sim = cv.matchShapes(j, depth_contours[d][0], cv.CONTOURS_MATCH_I3, 0)
            temp.append((sim, depth_contours[d][1], depth_contours[d][0], d))

        # Maps the rgb to the depth mask with the lowest similarity
        if temp:
            dp_ob = min(temp, key=first_agr)
            c_results.append((j, dp_ob[1], dp_ob[0], dp_ob[2]))
            # Deletes depth values that have already been assigned
            del depth_contours[dp_ob[3]]

    return c_results


def rotate_image(img, theta):
    """
    This function rotates the given image to the given angle.

    Args:
        img: the image to be rotated
        theta: the angle (in degree) to rotate the image
    """
    [h, w, _] = img.shape
    img_c = (w / 2, h / 2)

    rot = cv.getRotationMatrix2D(img_c, theta, 1)

    rad = math.radians(theta)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    result = cv.warpAffine(img, rot, (b_w, b_h), flags=cv.INTER_LINEAR)
    return result


def crop(img, contours, dir="", name=" ", end="", num=-1):
    """
    This function crops the image based on the given contours.

    Args:
        end: Ending for the segmented image
        dir: string name of the directory to store the image
        img: the RGB image to crop from
        contours: the sherd contours
        name: the name of the file
        num: the index of the sherd contour
    """
    x, y, w, h = cv.boundingRect(contours)
    new_img = img[y:y + h, x:x + w]

    if name != " ":
        # Checks for an external or internal suffix
        if end:
            cv.imwrite(f'{dir}/{name}/{name}_{end}.png', new_img)
        else:
            cv.imwrite(f'{dir}/{name}/{name}.png', new_img)
    else:
        (x, y), (Ma, ma), angle = cv.fitEllipse(contours)
        theta = angle - 90
        new_img = rotate_image(new_img, theta)

    return new_img


def create_result(sherd_img, card_img=None):
    """ This function creates the final image containing the cropped rgb, depth and measurement card.

    Args:
        sherd_img: cropped Sherd image
        card_img: cropped measure card
    """

    padding = 100

    RGB_B = cv.copyMakeBorder(sherd_img, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=(0, 0, 0))

    rr, rc, _ = RGB_B.shape
    cr, cc, _ = card_img.shape

    height, width = rr + cr, rc + cc

    card_start_h = rr
    card_start_w = int((width - cc) / 2)
    sherd_start = int((width - rc) / 2)

    result = np.zeros((height, width, 3), dtype=np.uint8)

    result[0:rr, sherd_start:sherd_start + rc, :] = RGB_B
    result[card_start_h:cr + card_start_h, card_start_w:cc + card_start_w, :] = card_img

    return result


# Regular Expression for retrieving the sherd ids
get_id = re.compile('\d.*\d+')


def splitRGB(filename, in_dir, out_dir):
    # Retrieves the sherd's ID from the filename.
    ending = ""

    if 'ext' in filename:
        ending = 'ext'
    elif 'int' in filename:
        ending = 'int'

    input_file = os.path.join(in_dir, filename)

    # Reads the image
    A = cv.imread(input_file)

    if 'SCAN' in filename:
        A = cv.rotate(A, cv.ROTATE_180)

    # Adding Gaussian Blur
    blur_img = cv.GaussianBlur(A, (31, 31), 0)

    # Converts the image to a binary image.
    [_, BW] = cv.threshold(cv.cvtColor(blur_img, cv.COLOR_BGR2GRAY), 20, 255, cv.THRESH_BINARY)

    # Mask used to block out non-sherd contours.
    mask = np.zeros(A.shape, np.uint8)

    # Quadrilateral shape detection
    [t_conts, _] = cv.findContours(BW, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    # Holds the contours of non-sherd shapes
    square_cont = []

    # The measurement scale card
    card = None

    # Checks each contours' area, arc-length and number of vertices.
    for cnt in t_conts:
        area = cv.contourArea(cnt)

        if area > 10000:
            per = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.01 * per, True)

            if len(approx) == 4:
                square_cont.append(cnt)

    if len(square_cont) > 0:
        cv.drawContours(mask, square_cont, -1, (255, 255, 255), cv.FILLED)
        # print(f"S num: {len(square_cont)}")

        square_cont = sorted(square_cont, key=cv.contourArea, reverse=False)
        card = square_cont[0]

    # Mask with quadrilaterals blocked out.
    block_out = cv.bitwise_and(A, mask)

    # Creates new binary image with out non-sherd shapes.
    BW = cv.cvtColor(BW, cv.COLOR_GRAY2BGR)
    combine = cv.subtract(BW, mask)
    combine = cv.cvtColor(combine, cv.COLOR_BGR2GRAY)

    # Finds sherd contours in binary image
    [temp_contours, _] = cv.findContours(combine, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Sort contours
    temp_contours = sorted(temp_contours, key=cv.contourArea, reverse=True)

    # Filter Contour list
    final_contours = [cont for cont in temp_contours if
                      cv.contourArea(cont) > 100000 and (
                              cv.contourArea(cont) < 4700000.0 or cv.contourArea(cont) > 6000000)]

    # Retrieves depth mask from subdirectories
    temp_folders = os.listdir(out_dir)
    mask_folder = []

    name = filename.split('.')[0]

    for i in temp_folders:
        if name in i:
            mask_folder.append(i)

    # Array to hold tuple of depth contours and directory name
    dp_contours = []

    for x in range(len(mask_folder)):
        dp_input = os.path.join(out_dir, mask_folder[x], 'mask.png')

        # Reads image
        dp = cv.imread(dp_input)

        # Convert depth image to binary image.
        [_, dp_bw] = cv.threshold(cv.cvtColor(dp, cv.COLOR_BGR2GRAY), 15, 255, cv.THRESH_BINARY)

        # Finds depth contour.
        [dp_temp_contours, _] = cv.findContours(dp_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Sort contours
        dp_temp_contours = sorted(dp_temp_contours, key=cv.contourArea, reverse=True)

        # Grabs biggest contour area
        dp_final_contour = dp_temp_contours[0]
        dp_contours.append((dp_final_contour, mask_folder[x]))

    # Calls the contour matching function
    results = contour_match(final_contours, dp_contours)

    for r in range(len(results)):

        # Print Result -----------------------------------------------------------------
        # Retrieves a fresh copy of the RGB image for cropping
        some_img = cv.imread(input_file)
        if 'SCAN' in input_file:
            some_img = cv.rotate(some_img, cv.ROTATE_180)

        # Retrieves the angle need to rotate the RGB image
        (x1, y1), (Ma1, ma1), d_angle = cv.fitEllipse(results[r][3])
        (x2, y2), (Ma2, ma2), rgb_angle = cv.fitEllipse(results[r][0])
        theta = rgb_angle - d_angle

        # Retrieves the cropped image of the Sherd
        result_img_rgb = crop(some_img, results[r][0], out_dir, results[r][1], ending, num=r)
        result_img_rgb = rotate_image(result_img_rgb, theta)
        card_img = crop(some_img, card)

        # Creates Final result image
        final_img = create_result(result_img_rgb, card_img)

        if ending:
            save_file = f'{out_dir}/{results[r][1]}/card_{ending}.png'
        else:
            save_file = f'{out_dir}/{results[r][1]}/card.png'

        cv.imwrite(save_file, final_img)
        # -------------------------------------------------------------------------------
