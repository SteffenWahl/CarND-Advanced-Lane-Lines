from src.find_lane_pixels import *
from src.search_around_poly import *

def fit_polynomial(binary_warped, left_lane, right_lane):
    if left_lane.detected is True and right_lane.detected is True:
        leftx, lefty, rightx, righty, out_img = search_around_poly(binary_warped, left_lane, right_lane)
        
    else:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_lane.update(leftx,lefty)
    right_lane.update(rightx,righty)

    # Generate x and y values for plotting
    try:
        left_fitx,ploty_left = left_lane.get_vals()
        right_fitx,ploty_right = right_lane.get_vals()
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty_left**2 + 1*ploty_left
        right_fitx = 1*ploty_right**2 + 1*ploty_right

    ## Visualization ##
    out_img[lefty, leftx] = [255, 150, 150]
    out_img[righty, rightx] = [150, 150, 255]

    out_img[np.int_(ploty_left), np.int_(left_fitx)] = [255, 0, 0]
    out_img[np.int_(ploty_right), np.int_(right_fitx)] = [0, 0, 255]

    return out_img, left_fitx, right_fitx, ploty_left, ploty_right

