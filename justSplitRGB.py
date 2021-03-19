"""
    This module is designed to split the images of multiple sherds into single images using the
    RGB and depth image of each sherd. This method is based on contour based shape matching.
"""
import os
import math
import cv2 as cv
import numpy as np
import re


# Helper Functions
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

        for d in depth_contours:
            sim = cv.matchShapes(j, d[0], cv.CONTOURS_MATCH_I3, 0)
            temp.append((sim, d[1], d[0]))

        dp_ob = min(temp, key=first_agr)
        # TEST LOG
        # print("Min Similarity: ", dp_ob[0])
        c_results.append((j, dp_ob[1], dp_ob[0], dp_ob[2]))

    # return results
    # print("Result: ", len(c_results))
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
    This function crops the image based on the given contours
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
        # Creates the directory for the Sherd
        if end:
            if not os.path.exists(f'{dir}/{name.split(".")[0][:-4]}-{num+1}'):
                os.mkdir(f'{dir}/{name.split(".")[0][:-4]}-{num+1}')

            cv.imwrite(f'{dir}/{name.split(".")[0][:-4]}-{num+1}/{end}.png', new_img)
        else:
            if not os.path.exists(f'{dir}/{name.split(".")[0]}-{num+1}'):
                os.mkdir(f'{dir}/{name.split(".")[0]}-{num+1}')

            cv.imwrite(f'{dir}/{name.split(".")[0]}-{num+1}/{name.split(".")[0]}.png', new_img)
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

    sherd_id = get_id.findall(filename)[0]
    # sherd_id = sherd_id.lstrip('0')

    input_file = os.path.join(in_dir, filename)

    # Reads the image
    A = cv.imread(input_file)

    if 'SCAN' in filename:
        A = cv.rotate(A, cv.ROTATE_180)

    # Adding Gaussian Blur
    blur_img = cv.GaussianBlur(A, (31, 31), 0)

    # Converts the image to a binary image.
    [threshold, BW] = cv.threshold(cv.cvtColor(blur_img, cv.COLOR_BGR2GRAY), 20, 255, cv.THRESH_BINARY)

    # Mask used to block out non-sherd contours.
    mask = np.zeros(A.shape, np.uint8)

    # Quadrilateral shape detection
    t_conts, hierarchy = cv.findContours(BW, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

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

                # print(f"Area: {cv.contourArea(cnt)}")

                # if 940000.0 > area > 860000.0:
                #     card = cnt
                square_cont.append(cnt)

    if len(square_cont) > 0:
        cv.drawContours(mask, square_cont, -1, (255, 255, 255), cv.FILLED)
        # print(f"S num: {len(square_cont)}")

        square_cont = sorted(square_cont, key=cv.contourArea, reverse=False)
        card = square_cont[0]

        # cv.imshow("Mask", mask)
        # cv.waitKey(0)

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

    # TEST LOGS
    # print("Contours: ", len(final_contours))

    # Drawing the contours (Might not be needed for production)(TEST)
    contImage = cv.drawContours(A, final_contours, -1, (0, 255, 0), 5, cv.FILLED)

    if card is not None:
        # Create Bounding Box for measurement card
        rect = cv.minAreaRect(card)
        card_BB = cv.boxPoints(rect)
        card_BB = card_BB.astype(int)

        # Drawing Bounding Box (TEST)
        # boxImage = cv.drawContours(contImage, [card_BB], -1, (255, 0, 0), 5)

        # cv.imshow("Result", boxImage)
        # cv.waitKey(0)

    some_img = cv.imread(input_file)

    if 'SCAN' in filename:
        some_img = cv.rotate(some_img, cv.ROTATE_180)

    for c in range(len(final_contours)):
        result_img_rgb = crop(some_img, final_contours[c], out_dir, filename, ending, num=c)

        card_img = None

        if card is not None:
            card_img = crop(some_img, card)

        # Creates Final result image
        final_img = create_result(result_img_rgb, card_img)
        if ending:
            save_file = f'{out_dir}/{filename.split(".")[0][:-4]}-{c + 1}/card_{ending}.png'
        else:
            save_file = f'{out_dir}/{filename.split(".")[0]}-{c + 1}/card.png'

        cv.imwrite(save_file, final_img)
