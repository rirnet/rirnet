import numpy as np
import shapely.geometry as geo

def generate_floor(min_side, max_side):
    p_extrude = .5 #else subtract
    box_x1 =box_y1 = 0
    box_x2, box_y2 = np.random.uniform(min_side, max_side, 2)
    box = geo.box(box_x1, box_y1, box_x2, box_y2)

    if np.random.uniform() < p_extrude:
        x_coords, y_coords = extrude_box(box, box_x1, box_y1, box_x2, box_y2, min_side, max_side)
    else:
        x_coords, y_coords = intrude_box(box, box_x1, box_y1, box_x2, box_y2, min_side, max_side)
    return x_coords, y_coords


def extrude_box(box1, box1_x1, box1_y1, box1_x2, box1_y2, min_side, max_side):

    box2_x1 = np.random.uniform(box1_x1, box1_x2)
    box2_y1 = np.random.uniform(box1_y1, box1_y2)

    box2_x2 = np.random.uniform(box2_x1+min_side, max_side)
    box2_y2 = np.random.uniform(box2_y1+min_side, max_side)

    box2 = geo.box(box2_x1, box2_y1, box2_x2, box2_y2)

    vertices = box1.union(box2).exterior.coords

    x_coords = vertices.xy[0][:-1]
    y_coords = vertices.xy[1][:-1]
    return x_coords, y_coords


def intrude_box(box1, box1_x1, box1_y1, box1_x2, box1_y2, min_side, max_side):
    border_thickness = min_side/3
    p_L_shape = .5 #else make U_shape

    box2_x1 = np.random.uniform(border_thickness, box1_x2-border_thickness)
    box2_y1 = np.random.uniform(border_thickness, box1_y2-border_thickness)
    print(box1_x2)
    if np.random.uniform() < p_L_shape:
        box2_x2 = box2_y2 = max_side
    else:
        box2_x2 = np.random.uniform(box2_x1+border_thickness, box1_x2-border_thickness)
        box2_y2 = max_side

    box2 = geo.box(box2_x1, box2_y1, box2_x2, box2_y2)
    vertices = box1.difference(box2).exterior.coords

    x_coords = vertices.xy[0][:-1]
    y_coords = vertices.xy[1][:-1]
    return x_coords, y_coords
