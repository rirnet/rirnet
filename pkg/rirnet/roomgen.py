import numpy as np
import shapely.geometry as geo
import pyroomacoustics as pra


def generate_from_dict(db_setup):
    np.random.seed()
    min_side, max_side = db_setup['side']
    min_height, max_height = db_setup['height']
    n_mics = db_setup['n_mics']
    fs = db_setup['fs']
    max_order = db_setup['max_order']
    min_abs, max_abs = db_setup['absorption']
    return generate(min_side, max_side, min_height, max_height, n_mics, fs, max_order, min_abs, max_abs)


def generate(min_side, max_side, min_height, max_height, n_mics, fs=16000, max_order=2, min_abs=0.1, max_abs=0.9):
    np.random.seed()
    floor_shape = generate_floor_shape(min_side, max_side)
    n_walls = len(floor_shape.boundary.xy[0])-1
    height = np.random.uniform(min_height, max_height)
    absorption = np.random.rand(n_walls, 1)*(max_abs-min_abs)+min_abs
    vertices = floor_shape.exterior.coords
    x_coords = vertices.xy[0][:-1]
    y_coords = vertices.xy[1][:-1]

    room = pra.Room.from_corners([x_coords, y_coords], fs=fs, max_order=max_order, absorption=absorption)
    room.extrude(height, absorption=absorption)

    visible = False
    while not visible:

        mic_pos = find_valid_pos(floor_shape, height, n_mics)
        source_pos = find_valid_pos(floor_shape, height, n_pos=1)
        source_pos = sum(source_pos, [])

        room.mic_array = pra.MicrophoneArray(mic_pos, room.fs)
        room.sources = [pra.SoundSource(source_pos, signal=None, delay=0)]

        visibility_list = []
        for pos in np.array(mic_pos).T:
            visibility_list.append(room.is_visible(room.sources[0], pos))
        visible = all(visibility_list)

    return room


def find_valid_pos(floor_shape, height, n_pos):
    np.random.seed()
    min_gap = .5
    point_limits = floor_shape.bounds[2:4]
    x_array = []
    y_array = []
    for n in range(n_pos):
        pos_found = False
        while not pos_found:
            point_x = np.random.uniform(0, point_limits[0])
            point_y = np.random.uniform(0, point_limits[1])
            diluted_point = geo.Point(point_x, point_y).buffer(min_gap)
            if floor_shape.contains(diluted_point):
                pos_found = True
                x_array.append(point_x)
                y_array.append(point_y)
    z_array = np.random.uniform(min_gap, height-min_gap, n_pos).tolist()
    return x_array, y_array, z_array


def generate_floor_shape(min_side, max_side):
    p_add = .5 #else subtract
    box_x1 = box_y1 = 0
    box_x2, box_y2 = np.random.uniform(min_side, max_side, 2)
    box = geo.box(box_x1, box_y1, box_x2, box_y2)

    if np.random.uniform() < p_add:
        floor_shape = add_box(box, box_x1, box_y1, box_x2, box_y2, min_side,
                max_side)
    else:
        floor_shape = subtract_box(box, box_x2, box_y2,
                min_side, max_side)

    return floor_shape


def add_box(box1, box1_x1, box1_y1, box1_x2, box1_y2, min_side, max_side):
    np.random.seed()

    box2_x1 = np.random.uniform(box1_x1, box1_x2)
    box2_y1 = np.random.uniform(box1_y1, box1_y2)

    box2_x2 = np.random.uniform(box2_x1+min_side, max_side)
    box2_y2 = np.random.uniform(box2_y1+min_side, max_side)

    box2 = geo.box(box2_x1, box2_y1, box2_x2, box2_y2)

    floor_shape = box1.union(box2)

    return floor_shape


def subtract_box(box1, box1_x2, box1_y2, min_side, max_side):
    np.random.seed()
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
