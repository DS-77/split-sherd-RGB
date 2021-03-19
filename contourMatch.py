"""
    This module is designed to split the images of multiple sherds into single images using the
    RGB and depth image of each sherd. This method is based on contour based shape matching. This approach
    prioritizes matching the RGB and Depth image from the RGB.
"""
import os
import math
import cv2 as cv
import numpy as np
import re


# TODO: Optimize code to run better. (Create functions for redundant code)
# TODO: Clean up code. (Remove all log statements and irrelevant comments)
# TODO: Blackout background not enclosed by sherd contour (makes image crops look better)

# TODO (optional): create logger for debugging

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
            if not os.path.exists(f'output/{name.split(".")[0][:-4]}-{num+1}'):
                os.mkdir(f'output/{name.split(".")[0][:-4]}-{num+1}')

            cv.imwrite(f'output/{name.split(".")[0][:-4]}-{num+1}/{end}.png', new_img)
        else:
            if not os.path.exists(f'output/{name.split(".")[0]}-{num+1}'):
                os.mkdir(f'output/{name.split(".")[0]}-{num+1}')

            cv.imwrite(f'output/{name.split(".")[0]}-{num+1}/{name.split(".")[0]}.png', new_img)
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


# Keeps track of files that produce error
problem_sherds = []

# Regular Expression for retrieving the sherd ids
get_id = re.compile('\d.*\d+')

# Grabs all files in the "rgb" directory.
Files = [f for f in os.listdir('rgb') if os.path.isfile(os.path.join('rgb', f)) and f != '.DS_Store']
Files.sort()

# TEST LOG
# print("Num Files: ", len(Files))

for fileIndex in range(len(Files)):
    # Retrieves the sherd's ID from the filename.
    ending = ""

    if 'ext' in Files[fileIndex]:
        ending = 'ext'
    elif 'int' in Files[fileIndex]:
        ending = 'int'

    sherd_id = get_id.findall(Files[fileIndex])[0]
    # sherd_id = sherd_id.lstrip('0')

    input_file = f'rgb/{Files[fileIndex]}'

    # Reads the image
    A = cv.imread(input_file)
    A_rotate = cv.rotate(A, cv.ROTATE_180)

    # Adding Gaussian Blur
    blur_img = cv.GaussianBlur(A_rotate, (31, 31), 0)

    # Converts the image to a binary image.
    [threshold, BW] = cv.threshold(cv.cvtColor(blur_img, cv.COLOR_BGR2GRAY), 25, 255, cv.THRESH_BINARY)

    # Contours is a Python list of all the contours in the image. Each individual contour is a Numpy 
    # array of (x,y) coordinates of boundary points of the object.

    # Mask used to block out non-sherd contours.
    mask = np.zeros(A_rotate.shape, np.uint8)

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

                if 940000.0 > area > 860000.0:
                    card = cnt

                square_cont.append(cnt)

    if len(square_cont) > 0:
        cv.drawContours(mask, square_cont, -1, (255, 255, 255), cv.FILLED)

    # Mask with quadrilaterals blocked out.
    block_out = cv.bitwise_and(A_rotate, mask)

    # Creates new binary image with out non-sherd shapes.
    BW = cv.cvtColor(BW, cv.COLOR_GRAY2BGR)
    combine = cv.subtract(BW, mask)
    combine = cv.cvtColor(combine, cv.COLOR_BGR2GRAY)

    # Finds sherd contours in binary image
    [temp_contours, hierarchy] = cv.findContours(combine, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Sort contours
    temp_contours = sorted(temp_contours, key=cv.contourArea, reverse=True)

    # Filter Contour list 
    final_contours = [cont for cont in temp_contours if
                      cv.contourArea(cont) > 100000 and (
                                  cv.contourArea(cont) < 4700000.0 or cv.contourArea(cont) > 6000000)]

    # TEST LOGS
    # print("Contours: ", len(final_contours))

    # Drawing the contours (Might not be needed for production)(TEST)
    # contImage = cv.drawContours(A_rotate, final_contours, -1, (0, 255, 0), 5, cv.FILLED)

    if card is not None:
        # Create Bounding Box for measurement card
        rect = cv.minAreaRect(card)
        card_BB = cv.boxPoints(rect)
        card_BB = card_BB.astype(int)

        # Drawing Bounding Box (TEST)
        # boxImage = cv.drawContours(contImage, [card_BB], -1, (255, 0, 0), 5)
        # TEST LOG
        # print("Box Coordinates: ", card_BB)

    # TEST ------------------------------------------------------------
    # Produces an image with the contours drawn on to the RGB image.
    # temp_dir = 'bw_images/rgb_contours'
    # if not os.path.exists(temp_dir):
    #     os.mkdir(temp_dir)
    #
    # bwTestFile = temp_dir + '/' + 'SCAN' + sherd_id + '.png'
    # cv.imwrite(bwTestFile, contImage)
    # -----------------------------------------------------------------

    # Checks if there is the same number of Depth images as Sherds featured in the RGB images.
    temp_depth_Files = os.listdir('depth/')
    temp_depth_Files.sort()

    depth_files = []

    for file in temp_depth_Files:

        if re.search(sherd_id, file):
            depth_files.append(file)

    # TEST LOG
    # print("Depth Num: ", len(depth_files))

    if len(final_contours) != len(depth_files):
        # TEST LOG
        # print("Number of Sherds and Depth images do not match.")

        problem_sherds.append(input_file)
        continue
    elif len(depth_files) == 0:
        # TEST LOG
        # print("No depth images were found")

        problem_sherds.append(input_file)
        continue

    # Holds depth contours with matching filenames
    dp_contours = []

    for x in range(len(depth_files)):
        depth_input_file = f'depth/{depth_files[x]}'

        # TEST LOG
        # print(depth_input_file)
        dp = cv.imread(depth_input_file)

        # Convert depth image to binary image.
        [dp_threshold, DPBW] = cv.threshold(cv.cvtColor(dp, cv.COLOR_BGR2GRAY), 15, 255, cv.THRESH_BINARY)

        # Finds depth contour.
        [dp_temp_contours, dp_hierarchy] = cv.findContours(DPBW, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Sort contours
        dp_temp_contours = sorted(dp_temp_contours, key=cv.contourArea, reverse=True)

        # Grabs biggest contour area
        dp_final_contour = dp_temp_contours[0]

        # TEST LOG
        # print("Depth Contour Area: ", cv.contourArea(dp_final_contour))
        dp_contours.append((dp_final_contour, depth_files[x]))

    # TEST LOG
    # print("Processed Depths: ", len(dp_contours))
    results = contour_match(final_contours, dp_contours)

    for r in range(len(results)):
        # Print Result -----------------------------------------------------------------
        # Retrieves a fresh copy of the RGB image for cropping
        some_img = cv.imread(input_file)
        rot_img = cv.rotate(some_img, cv.ROTATE_180)

        # Retrieves the angle need to rotate the RGB image
        (x1, y1), (Ma1, ma1), d_angle = cv.fitEllipse(results[r][3])
        (x2, y2), (Ma2, ma2), rgb_angle = cv.fitEllipse(results[r][0])
        theta = rgb_angle - d_angle

        # Retrieves the cropped image of the Sherd
        result_img_rgb = crop(rot_img, results[r][0], sherd_id, Files[fileIndex], ending, num=r)
        result_img_rgb = rotate_image(result_img_rgb, theta)
        result_img_dp = cv.imread(f'depth/{results[r][1]}')
        card_img = crop(rot_img, card)

        # Save the corresponding depth image to the Sherd's directory
        if ending:
            cv.imwrite(f'output/{Files[fileIndex].split(".")[0][:-4]}-{r+1}/depth.png', result_img_dp)
        else:
            cv.imwrite(f'output/{Files[fileIndex].split(".")[0]}-{r+1}/depth.png', result_img_dp)

        # Creates Final result image
        final_img = create_result(result_img_rgb, card_img)

        if ending:
            save_file = f'output/{Files[fileIndex].split(".")[0][:-4]}-{r+1}/card_{ending}.png'
        else:
            save_file = f'output/{Files[fileIndex].split(".")[0]}-{r+1}/card.png'

        cv.imwrite(save_file, final_img)
        # -------------------------------------------------------------------------------

# TEST LOG
# print("Match Errors: ", len(problem_sherds))
