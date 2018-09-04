import numpy as np
import shapely.geometry as geo
import pyroomacoustics as pra
import matplotlib.pyplot as plt

#TODO fix height control of source and receiver
#TODO clean up code!!!

def generate(min_side, max_side, min_height, max_height):
    floor_shape = generate_floor_shape(min_side, max_side)
    height = np.random.uniform(min_height, max_height)

    mic_pos = find_valid_pos(floor_shape)
    source_pos = find_valid_pos(floor_shape)

    vertices = floor_shape.exterior.coords

    x_coords = vertices.xy[0][:-1]
    y_coords = vertices.xy[1][:-1]


    room = pra.Room.from_corners([x_coords, y_coords], fs=16000, max_order=2, absorption=0.1)
    room.extrude(height)

    print(source_pos[0])
    room.add_source([source_pos[0],source_pos[1], 1])

    room.plot()
    plt.show()

def find_valid_pos(floor_shape):
    point_limits = floor_shape.bounds[2:4]
    print(point_limits)
    pos_found = False
    while not pos_found:
        point_x = np.random.uniform(0, point_limits[0])
        point_y = np.random.uniform(0, point_limits[1])
        diluted_point = geo.Point(point_x, point_y).buffer(.1)
        if floor_shape.contains(diluted_point):
            pos_found = True
    return point_x, point_y




def generate_floor_shape(min_side, max_side):
    p_add = .5 #else subtract
    box_x1 = box_y1 = 0
    box_x2, box_y2 = np.random.uniform(min_side, max_side, 2)
    box = geo.box(box_x1, box_y1, box_x2, box_y2)

    if np.random.uniform() < p_add:
        floor_shape = add_box(box, box_x1, box_y1, box_x2, box_y2, min_side, max_side)
    else:
        floor_shape = subtract_box(box, box_x1, box_y1, box_x2, box_y2, min_side, max_side)

    return floor_shape


def add_box(box1, box1_x1, box1_y1, box1_x2, box1_y2, min_side, max_side):

    box2_x1 = np.random.uniform(box1_x1, box1_x2)
    box2_y1 = np.random.uniform(box1_y1, box1_y2)

    box2_x2 = np.random.uniform(box2_x1+min_side, max_side)
    box2_y2 = np.random.uniform(box2_y1+min_side, max_side)

    box2 = geo.box(box2_x1, box2_y1, box2_x2, box2_y2)

    floor_shape = box1.union(box2)

    return floor_shape


def subtract_box(box1, box1_x1, box1_y1, box1_x2, box1_y2, min_side, max_side):
    border_thickness = min_side/3
    p_L_shape = .5 #else make U_shape

    box2_x1 = np.random.uniform(border_thickness, box1_x2-border_thickness)
    box2_y1 = np.random.uniform(border_thickness, box1_y2-border_thickness)
    if np.random.uniform() < p_L_shape:
        box2_x2 = box2_y2 = max_side
    else:
        box2_x2 = np.random.uniform(box2_x1+border_thickness, box1_x2-border_thickness)
        box2_y2 = max_side

    box2 = geo.box(box2_x1, box2_y1, box2_x2, box2_y2)
    floor_shape = box1.difference(box2)

    return floor_shape
