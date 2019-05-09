'''
Convert map coordinates to simulator coordinates
'''

from math import pi, degrees

# Scaling constants
A_x = 0.758725341426
B_x = -1038.694992412747
A_y = -0.759878419452888
B_y = + 79.787234042553209

def map_to_sim(x_map, y_map, head_map):
    '''Convert x,y,heading map to simulator coordinates'''
    x_sim = A_x*x_map + B_x
    y_sim = A_y*y_map + B_y
    head_sim = degrees(head_map - pi/2)
    head_sim = (head_sim + 360) % 360
    return (x_sim, y_sim, head_sim)
