import navpy
import numpy as np

def reduce_to_2d(points, min_height=0.3, col_hits=2, res=0.2):
    # points = xs, ys, zs
    [xs, ys, zs] = points
    if min_height > 0:
        xs = xs[zs > min_height]
        ys = ys[zs > min_height]
        zs = zs[zs > min_height]

    xs = np.ceil(xs / res) * res
    ys = np.ceil(ys / res) * res
    points_to_keep = []
    while len(xs) > 0 and len(ys) > 0:
        x = xs[0]
        y = ys[0]
        point_bool_index = np.logical_and(xs == x, ys == y)
        count = np.sum(point_bool_index)
        if [x,y] not in points_to_keep and count > col_hits - 1:
            points_to_keep.append([x,y])

        xs = xs[np.logical_not(point_bool_index)]
        ys = ys[np.logical_not(point_bool_index)]
    
#     return np.array(points_to_keep)[:,0], np.array(points_to_keep)[:,1]
    return points_to_keep


def movement_correct_velo(data, min_height=0.3, col_hits=2, res=0.2):
    data = data.copy()
    print('converting velo points to birds eye view, and orienting NE')
    N = []
    E = []
    initial_lla = [data.iloc[0]['lat'], data.iloc[0]['lon'], data.iloc[0]['alt']]
    counter = 0
    for data_point in data.iterrows():
        data_point = data_point[1]
        points = reduce_to_2d(data_point['velodyne_points'][:, 0:3].copy().T, min_height=min_height, col_hits=col_hits, res=res)
        c, s = np.cos(np.array((data_point['yaw'] - np.pi / 2))), np.sin(np.array((data_point['yaw'] - np.pi / 2)))
        local_NE_rotation = np.array(((c, -s), (s, c)))
        current_lla = [data_point['lat'], data_point['lon'], data_point['alt']]
        NED_displacement = navpy.lla2ned(*(current_lla + initial_lla))
        NED_points = (local_NE_rotation @ np.array(points).T).T + NED_displacement[0:2]
        N.append(NED_points[:, 0])
        E.append(-NED_points[:, 1])
        print(f'proccing point cloud: {counter}')
        counter += 1
        
    data['North'] = N
    data['East'] = E
    
    return data