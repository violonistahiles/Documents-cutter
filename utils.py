import numpy as np
import cv2
from PIL import Image


def unit_vector(vector):
    """ Returns the unit vector of the vector """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if v2_u[1] < 0:
        angle = 6.28 - angle
    return angle


def get_intersection(line1, line2):
    """ Count intersection of two lines """
    p1, p2 = line1
    p3, p4 = line2

    k1 = (p1[1] - p2[1]) / (p1[0] - p2[0] + 0.1)
    b1 = p1[1] - p1[0] * k1

    k2 = (p3[1] - p4[1]) / (p3[0] - p4[0] + 0.1)
    b2 = p3[1] - p3[0] * k2

    x = (b2 - b1) / (k1 - k2)
    y = x * k1 + b1

    return int(x), int(y)


def rotate(p, origin=(0, 0), degrees=0):
    """ Rotate coordinates to some angle """
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def get_max_hw(points):
    """ Calculate maximum height and width from four clockwise sorted (start top left) points """
    w1 = np.linalg.norm(points[0] - points[1])
    w2 = np.linalg.norm(points[2] - points[3])
    h1 = np.linalg.norm(points[0] - points[2])
    h2 = np.linalg.norm(points[1] - points[3])
    hf = int(np.max([h1, h2]))
    wf = int(np.max([w1, w2]))

    return hf, wf


def sort_points(points):
    """
    Sort four points by its distance from top left corner
    Input:
        - points - four points [[x,y],...]
    Output:
        - points - sorted points
    """

    points = points.reshape(4, 2)

    distances = [np.sqrt(np.power(point[0], 2) + np.power(point[1], 2)) for point in points]
    dist_ind = np.argsort(distances)
    points = points[dist_ind]

    # Fix 2-nd and 3-d points
    if points[2][1] < points[1][1]:
        points = points[[0, 2, 1, 3]]

    # Swap the last two points to create correct polygon
    points = points[[0, 1, 3, 2]]

    return points


def fix_points(i, points, h_p, w_p):
    """
    Fix cases where points don't align to boundaries
    Input:
        - i - number of quarter
        - points - four points [[x,y],...]
        - h_p - height of quarter
        - w_p - width of quarter
    Output:
        - points - fixed points
    """

    if i == 0:
        points[1][0] = w_p
        points[3][1] = h_p
        points[2] = np.array([w_p, h_p])

    elif i == 1:
        points[0][0] = 0.
        points[3] = np.array([0., h_p])
        points[2][1] = h_p

    elif i == 2:
        points[0][1] = 0.
        points[1] = np.array([w_p, 0.])
        points[2][0] = w_p

    elif i == 3:
        points[0] = np.array([0., 0.])
        points[1][1] = 0.
        points[3][0] = 0.

    return np.array(points).astype(np.int32)


def get_intersection_point(points, center, degrees):
    """ Calculate intersection point related to each quarter """
    # Prepare coordinated for angle calculation
    point_vectors = np.array(
        [[point[0][0], center[1] * 2 - point[0][1]] for point in points])  # Reverse height coordinates
    point_vectors = [point - center for point in point_vectors]  # Center coordinates
    # Rotate coordinates for quarter quality
    point_vectors = [rotate(point, origin=(0, 0), degrees=degrees) for point in point_vectors]
    start_vector = np.array([1., 0.])  # Vector relative to which the angle is calculated
    thetas = [angle_between(start_vector, np.array(point_vector)) for point_vector in point_vectors]  # Angles
    sorted_thetas = np.argsort(thetas)  # Sort angles counterclockwise
    points_cur = points[sorted_thetas].squeeze()  # Sort points by angles
    first_line = [points_cur[0], points_cur[1]]
    second_line = [points_cur[-2], points_cur[-3]]

    int_point = get_intersection(first_line, second_line)

    return int_point, points_cur


def sort_clockwise(i, points, h_p, w_p):
    """
    Get intersection of first from start and third from end lines and
    sort points as [tl, tr, bt, bl]
    Input:
        - i - number of quarter
        - points - mask approximation points
        - h_p - quarter height
        - w_p - quarter width
    Output:
        - coords - four sorted coordinates [tl, tr, bt, bl]
    """

    center = np.array([w_p // 2, h_p // 2])  # Center of quarter

    if i == 0:
        int_point, points_cur = get_intersection_point(points, center, 0)
        # Write coordinates relative to it's quarter
        coords = [int_point, points_cur[0], points_cur[-1], points_cur[-2]]

    elif i == 1:
        int_point, points_cur = get_intersection_point(points, center, 90)
        # Write coordinates relative to it's quarter
        coords = [points_cur[-2], int_point, points_cur[0], points_cur[-1]]

    elif i == 2:
        int_point, points_cur = get_intersection_point(points, center, -90)
        # Write coordinates relative to it's quarter
        coords = [points_cur[0], points_cur[-1], points_cur[-2], int_point]

    elif i == 3:
        int_point, points_cur = get_intersection_point(points, center, -180)
        # Write coordinates relative to it's quarter
        coords = [points_cur[-1], points_cur[-2], int_point, points_cur[0]]

    return coords


def process_quarters(parts, start_epsilon):
    '''
    Approximate quarter masks by four points
    Input:
        - parts - quarters [tl, tr, bl, br]
        - start_epsilon - epsilon for start approximation
    Output:
        - new_parts - approximated quarters [tl, tr, bl, br]
    '''
    new_parts = []

    for i, part in enumerate(parts):
        h_p, w_p = part.shape[0], part.shape[1]
        part_area = h_p * w_p
        # Get main contour of quarter
        p_contours, hierarchy = cv2.findContours(part, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_cnt_p = p_contours[np.argmax([cv2.contourArea(contour) for contour in p_contours])]
        # Approximate contour
        peri = cv2.arcLength(max_cnt_p, True)
        approx = cv2.approxPolyDP(max_cnt_p, start_epsilon * peri, True).astype(np.int32)

        # If after approximation contour can't be discribed by 4 lines
        if len(approx) > 4:
            # Check what is more valuable convexity defects correction or approximation cutting
            hull = cv2.convexHull(max_cnt_p, returnPoints=True)
            tmp_part = cv2.fillPoly(np.zeros_like(part), [hull.squeeze()], [255])
            tmp_part_approx = cv2.fillPoly(np.zeros_like(part), [approx], [255])
            # Area of convexity defects correction
            unintersect_area = np.sum(np.clip(np.abs(part - tmp_part), 0., 1.)) / part_area
            # Area of approximation
            unintersect_area_approx = np.sum(np.clip(part - tmp_part_approx, 0., 1.)) / part_area
            # print(unintersect_area, unintersect_area_approx)
            if unintersect_area > unintersect_area_approx:
                peri = cv2.arcLength(hull, True)
                approx_h = cv2.approxPolyDP(hull, 0.005 * peri, True).astype(np.int32)
            else:
                approx_h = cv2.approxPolyDP(max_cnt_p, 0.02 * peri, True).astype(np.int32)

            if len(approx_h) > 4:
                # Get intersection of first from start and third from end lines
                s_approx = sort_clockwise(i, approx_h, h_p, w_p)  # Sort points [tl, tr, bt, bl]
                f_approx = fix_points(i, s_approx, h_p, w_p)  # Fix points alignment to boundaries
                tmp_part = cv2.fillPoly(np.zeros_like(part), [f_approx], [255])
                new_parts.append(tmp_part)

            else:
                s_approx = sort_points(approx_h)  # Sort points [tl, tr, bt, bl]
                f_approx = fix_points(i, s_approx, h_p, w_p)  # Fix points alignment to boundaries
                tmp_part = cv2.fillPoly(np.zeros_like(part), [f_approx], [255])
                new_parts.append(tmp_part)

        else:

            s_approx = sort_points(approx)  # Sort points [tl, tr, bt, bl]
            f_approx = fix_points(i, s_approx, h_p, w_p)  # Fix points alignment to boundaries
            tmp_part = cv2.fillPoly(np.zeros_like(part), [f_approx], [255])
            new_parts.append(tmp_part)

    return new_parts


def get_list_coords(mask):
    """
    Determine the four approximate coordinates of the document
    Input:
        - mask - grayscale image of document mask
    Output:
        - cut_coords - four coordinates sorted clockwise starting from top left corner
    """

    # Get main contour of mask
    h, w = mask.shape[0], mask.shape[1]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
    # Get four quarters from main contour for better defects processing
    test_img = cv2.fillPoly(np.zeros_like(mask), [max_cnt], [255])
    tl = test_img[:h // 2, :w // 2].copy()
    tr = test_img[:h // 2, w // 2:].copy()
    bl = test_img[h // 2:, :w // 2].copy()
    br = test_img[h // 2:, w // 2:].copy()
    parts = [tl, tr, bl, br]

    new_parts = process_quarters(parts.copy(), 0.01)

    # Connect all quarters together
    full_mask = np.concatenate((np.concatenate((new_parts[0], new_parts[1]), axis=1),
                                np.concatenate((new_parts[2], new_parts[3]), axis=1)), axis=0)
    # Approximate final contour
    f_contours, hierarchy = cv2.findContours(full_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    peri = cv2.arcLength(f_contours[0], True)
    approx_f = cv2.approxPolyDP(f_contours[0], 0.1 * peri, True).astype(np.int32)

    # Sort points [tl, tr, bt, bl]
    coords = sort_points(approx_f[:4].reshape(4, 2))
    tmp_mask = cv2.fillPoly(np.zeros_like(mask), [coords], [255])
    tmp_mask_cut_area = np.sum(np.clip(mask - tmp_mask, 0., 1.)) / np.sum(np.clip(mask, 0., 1.))
    # If after processing main part of the mask was cut, then repeat it more accurate
    if tmp_mask_cut_area > 0.05:
        # print(tmp_mask_cut_area)
        new_parts = process_quarters(parts.copy(), 0.001)

        # Connect all quarters together
        full_mask = np.concatenate((np.concatenate((new_parts[0], new_parts[1]), axis=1),
                                    np.concatenate((new_parts[2], new_parts[3]), axis=1)), axis=0)
        # Approximate final contour
        f_contours, hierarchy = cv2.findContours(full_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        peri = cv2.arcLength(f_contours[0], True)
        approx_f = cv2.approxPolyDP(f_contours[0], 0.1 * peri, True).astype(np.int32)

        # Sort points [tl, tr, br, bl]
        coords = sort_points(approx_f[:4].reshape(4, 2))

    cut_coords = [coords[0], coords[1], coords[3], coords[2]]

    return cut_coords


def correct_orientation(file_path):

    pix = Image.open(file_path)
    # get correction based on 'Orientation' from Exif (==Tag 274)
    try:
        deg = {3: 180, 6: 270, 8: 90}.get(pix._getexif().get(274, 0), 0)
    except:
        deg = 0
    if deg != 0:
        pix = pix.rotate(deg, expand=False)
    # convert PIL -> opencv
    img = np.array(pix)

    return img


def predict(file_path, pred_path, model):
    """ Process photo of document and cut it """
    INPUT_SHAPE = 512

    img_st = cv2.imread(file_path, flags=cv2.IMREAD_UNCHANGED)
    h, w = img_st.shape[0], img_st.shape[1]
    if w > h:
        img_st = cv2.rotate(img_st, cv2.ROTATE_90_CLOCKWISE)
        h, w = w, h
    # Reshape for neural net input size
    img = cv2.resize(img_st[:, :, ::-1], (INPUT_SHAPE, INPUT_SHAPE), interpolation=cv2.INTER_AREA)

    # Prepare image for prediction
    std = np.array([0.229, 0.224, 0.225]).T
    mean = np.array([0.485, 0.456, 0.406]).T
    img = (img / 255 - mean) / std

    # Generate set of vertically and horizontally flipped images for better prediction accuracy
    imgs = np.array([img, img[:, ::-1, :].copy(), img[::-1, :, :].copy(), img[::-1, ::-1, :].copy()])

    imgs = imgs.astype(np.float32).transpose(0,3,1,2)
    imgs = imgs if isinstance(imgs, list) else [imgs]
    feed = dict([(inputs.name, imgs[n]) for n, inputs in enumerate(model.get_inputs())])
    masks_0 = model.run(None, feed)[0].squeeze()

    # Postprocessing
    mask = (masks_0[0] + masks_0[1][:, ::-1] + masks_0[2][::-1, :] + masks_0[3][::-1, ::-1]) / 4
    _, mask = cv2.threshold(np.uint8(mask * 255), 150, 255, cv2.THRESH_BINARY)
    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)

    cut_coords = get_list_coords(mask)  # Get coordinates for document
    # Prepare coordinates for perspective transformation
    pts1 = np.float32(cut_coords)
    hf, wf = get_max_hw(pts1)  # Height and width for image after perspective transformation
    pts2 = np.float32([[0, 0], [wf, 0], [0, hf], [wf, hf]])
    M = cv2.getPerspectiveTransform(pts1, pts2)  # Get transformation matrix
    document = cv2.warpPerspective(img_st, M, (wf, hf))

    cv2.imwrite(pred_path, document)

    return img_st[:, :, ::-1]