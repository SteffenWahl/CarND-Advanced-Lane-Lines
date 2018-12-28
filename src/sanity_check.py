from src.Line import *
def sanity_check(left_lane, right_lane, xm_per_pix, ym_per_pix):
    l_range = left_lane.y_range()
    r_range = right_lane.y_range()
    all_range = [ max(l_range[0],r_range[0]),min(l_range[1],r_range[1])]
    xl0 = left_lane.get_val(all_range[0])
    xr0 = right_lane.get_val(all_range[0])
    xl1 = left_lane.get_val(all_range[1])
    xr1 = right_lane.get_val(all_range[1])
    diff0 = abs(xl0-xr0)*xm_per_pix 
    diff1 = abs(xl1-xr1)*xm_per_pix
    print('Diff0: '+ str(diff0))
    print('Diff1: '+ str(diff1))
    if (diff0 < 2.7) | (diff0 > 4.7):
        return False 
    if (diff1 < 3.2) | (diff1 > 4.2):
        return False 
    return True

