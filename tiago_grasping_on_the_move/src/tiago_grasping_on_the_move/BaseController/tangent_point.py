import math
import numpy as np

def tangent_point(x_b,y_b,x_t,y_t,r_c,debug=False): 
    iters = 360
    thetas = np.linspace(0, 2*np.pi, iters)
    x_c = 0
    y_c = 0
    old_diff = np.inf

    for i in range(iters):
        candidate_theta = thetas[i]
        candidate_x_tang = x_t + r_c * math.cos(candidate_theta)
        candidate_y_tang = y_t + r_c * math.sin(candidate_theta)
        
        if math.sin(candidate_theta) == 0.0:
            m = np.pi/2
        else:
            m = -math.cos(candidate_theta)/math.sin(candidate_theta)
        b = candidate_y_tang - m*candidate_x_tang
        y_test = m * x_b + b
        
        if abs(y_test - y_b) < old_diff and abs(x_c - candidate_x_tang) > 0.2:
            if debug:
                print(f"x = {x_c} || y = {y_c} || theta = {candidate_theta}")
            x_c = candidate_x_tang
            y_c = candidate_y_tang
            old_diff = abs(y_test - y_b)
    
    return x_c,y_c

# tangent_point(x_b=0,y_b=0,x_t=3.93,y_t=1.98,r_c=0.6,debug=True)