"""
    This module is designed to split the images of multiple sherds into single images using the
    RGB and depth image of each sherd. This method is derived from contour based shape matching. This approach
    prioritizes matching RGB and Depth images from generated Depth masks.
"""
import math
import os
import re
import cv2 as cv
import numpy as np


def resize_img(img):
    """
    This method downsizes the image.
    Args:
        img: An image.

    Returns: resized image.

    """

    scale = 20
    dim = (int(img.shape[1] * scale / 100), int(img.shape[0] * scale / 100))
    n_img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)

    return n_img


def first_agr(tup):
    """
    This function returns the first element in the given tuple.

    Args:
        tup: the specified tuple.
    """
    return tup[0]


def match_helper(sherd_shape, depth_shape, inx):
    """
    This method is a helper method for the contour_match method.
    Args:
        sherd_shape: contour from RGB
        depth_shape: contour from depth
        inx: index

    Returns: the similarity, depth name, the depth shape, index

    """
    sim = cv.matchShapes(sherd_shape, depth_shape[inx][0], cv.CONTOURS_MATCH_I3, 0)
    return sim, depth_shape[inx][1], depth_shape[inx][0], inx


def contour_match(rbg_contours, depth_contours):
    """
    This function maps the RGB contours to depth images using the shape matching method to compare contours.

    Args:
        rbg_contours: the contours from the rgb image
        depth_contours: the contours from the depth image
    """
    c_results = []

    for j in rbg_contours:
        temp = [match_helper(j, depth_contours, d) for d in range(len(depth_contours))]

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
    h, w, _ = np.shape(img)
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
        (_, _), (_, _), angle = cv.minAreaRect(contours)
        theta = 0
        if angle < 3:
            theta = -93
        elif angle > 80:
            theta = int(angle) - 5

        new_img = cv.rotate(new_img, theta, new_img)
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


def find_square(cnt):
    """
    This method determines if the contour is a square.
    Args:
        cnt: the contour

    Returns: a contour

    """
    area = cv.contourArea(cnt)

    if area > 10000:
        per = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.01 * per, True)

        if len(approx) == 4:
            return cnt


def process_depth(out_dir, mask, name):
    """
    This method detects the contours in the depth image
    Args:
        out_dir: the output directory path
        mask: the binary mask
        name: name of the file

    Returns: the depth contours and a binary mask

    """
    dp_input = os.path.join(out_dir, mask, name)

    # Reads image
    dp = cv.imread(dp_input)

    # Convert depth image to binary image.
    _, dp_bw = cv.threshold(cv.cvtColor(dp, cv.COLOR_BGR2GRAY), 15, 255, cv.THRESH_BINARY)

    # Finds depth contour.
    dp_temp_contours, _ = cv.findContours(dp_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Sort contours
    dp_temp_contours = sorted(dp_temp_contours, key=cv.contourArea, reverse=True)

    # Grabs biggest contour area
    dp_final_contour = dp_temp_contours[0]
    return dp_final_contour, mask


# Regular Expression for retrieving the sherd ids
get_id = re.compile('\d.*\d+')


def splitRGB(filename, in_dir, out_dir):

    # Retrieves the sherd's ID from the filename.
    ending = 'ext' if 'ext' in filename else 'int' if 'int' in filename else None

    input_file = os.path.join(in_dir, filename)

    # Retrieves depth mask from subdirectories
    temp_folders = os.listdir(out_dir)

    # Retrieves filename and sherd ID.
    # scan_name = filename.split('.')[0][: -4] if ending else filename.split('.')[0]
    scan_name = filename.split('_')[0]
    temp_id = filename.split('_')[2] if ending else filename.split('_')[1]
    scan_id = temp_id.split('.')[0]

    print(f'Looking for ID: {scan_id}')

    mask_folder = tuple(i for i in temp_folders if scan_id in i)

    if len(mask_folder) <= 0 or len(mask_folder) == 1:
        print(f'No splitting necessary. There is {len(mask_folder)} sherds to split.')
        return

    # Reads the image
    A = cv.rotate(resize_img(cv.imread(input_file)), cv.ROTATE_180) if 'SCAN' in filename \
        else resize_img(cv.imread(input_file))

    # Adding Gaussian Blur
    # blur_img = cv.GaussianBlur(A, (31, 31), 0) <-- Default for large images

    # Converts the image to a binary image.
    _, BW = cv.threshold(cv.cvtColor(cv.GaussianBlur(A, (5, 5), 0), cv.COLOR_BGR2GRAY), 20, 255, cv.THRESH_BINARY)

    # Mask used to block out non-sherd contours.
    mask = np.zeros(A.shape, np.uint8)

    # Quadrilateral shape detection
    t_conts, _ = cv.findContours(BW, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    # Holds the contours of non-sherd shapes
    square_cont = tuple(map(find_square, t_conts))
    square_cont = tuple(i for i in square_cont if i is not None)

    # The measurement scale card
    card = None

    if len(square_cont) > 0:
        cv.drawContours(mask, square_cont, -1, (255, 255, 255), cv.FILLED)
        # print(f"S num: {len(square_cont)}")

        square_cont = sorted(square_cont, key=cv.contourArea, reverse=False)
        card = square_cont[0]

    # Creates new binary image with out non-sherd shapes.
    BW = cv.cvtColor(BW, cv.COLOR_GRAY2BGR)
    combine = cv.subtract(BW, mask)
    combine = cv.cvtColor(combine, cv.COLOR_BGR2GRAY)

    # Finds sherd contours in binary image
    temp_contours, _ = cv.findContours(combine, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Sort contours
    temp_contours = sorted(temp_contours, key=cv.contourArea, reverse=True)

    # Filter Contour list
    # final_contours = (cont for cont in temp_contours if
    #                   cv.contourArea(cont) > 100000 and (
    #                           cv.contourArea(cont) < 4700000.0 or cv.contourArea(cont) > 6000000))

    final_contours = [cont for cont in temp_contours if cv.contourArea(cont) > 1000]

    # Array to hold tuple of depth contours and directory name
    dp_contours = [process_depth(out_dir, mask_folder[x], 'mask.png') for x in range(len(mask_folder))]

    # Calls the contour matching function
    results = contour_match(final_contours, dp_contours)

    for r in range(len(results)):

        # Print Result -----------------------------------------------------------------
        # Retrieves a fresh copy of the RGB image for cropping
        some_img = cv.rotate(resize_img(cv.imread(input_file)), cv.ROTATE_180) if 'SCAN' in input_file \
            else resize_img(cv.imread(input_file))

        # Retrieves the angle need to rotate the RGB image
        # (_, _), (_, _), d_angle = cv.minAreaRect(results[r][3])
        # (_, _), (_, _), rgb_angle = cv.minAreaRect(results[r][0])
        # theta = rgb_angle - d_angle
        # theta = int(rgb_angle + (rgb_angle - d_angle))
        theta = 0

        # Retrieves the cropped image of the Sherd
        result_img_rgb = crop(some_img, results[r][0], out_dir, results[r][1], ending, num=r)
        result_img_rgb = cv.rotate(result_img_rgb, theta, result_img_rgb)
        card_img = crop(some_img, card)

        # Creates Final result image
        final_img = create_result(result_img_rgb, card_img)

        if ending:
            save_file = f'{out_dir}/{results[r][1]}/{scan_name}_card_{ending}.png'
        else:
            save_file = f'{out_dir}/{results[r][1]}/{scan_name}_card.png'

        print(f'Saving {save_file}')
        cv.imwrite(save_file, final_img)
        # -------------------------------------------------------------------------------
