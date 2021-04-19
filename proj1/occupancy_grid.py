import navpy
import numpy as np

def initialize_grid(data, res):
    global_max_N = 0
    global_max_E = 0
    global_min_N = 0
    global_min_E = 0
    for data_point in data.iterrows():
        data_point = data_point[1]
        max_N = np.max(data_point['North']) # meters
        max_E = np.max(data_point['East'])
        min_N = np.min(data_point['North'])
        min_E = np.min(data_point['East'])
        
        global_max_N = max_N if max_N > global_max_N else global_max_N
        global_max_E = max_E if max_E > global_max_E else global_max_E
        global_min_N = min_N if min_N < global_min_N else global_min_N
        global_min_E = min_E if min_E < global_min_E else global_min_E
        
    NED_bounds = [global_min_E, global_min_N, global_max_E, global_max_N]
    width = int(np.ceil((global_max_E - global_min_E) / 0.2))
    hight = int(np.ceil((global_max_N - global_min_N) / 0.2))
    grid = np.zeros((hight, width))
    # min must be less then zero
    return grid, NED_bounds


def proceess_logit_grid(
            logit_grid,
            current_occupancy_grid,
            prop_upper_saturation_threshold=0.98,
            prop_lower_saturation_threshold=0.02,
            occupied_threshold = 0.8,
            frees_threshold = 0.35
            ):

    probability_grid  = 1 - 1 / (1 + np.exp(logit_grid))
    probability_grid[probability_grid > prop_upper_saturation_threshold] = prop_upper_saturation_threshold
    probability_grid[probability_grid < prop_lower_saturation_threshold] = prop_lower_saturation_threshold
    occupancy_grid = 0.5 * np.ones(current_occupancy_grid.shape)
    occupancy_grid[probability_grid > occupied_threshold] = 1
    occupancy_grid[probability_grid < frees_threshold] = 0
    keep_old_value = np.logical_and(occupancy_grid == 0.5, current_occupancy_grid != 0.5)
    occupancy_grid[keep_old_value] = occupancy_grid[keep_old_value]
    logit_grid = np.log(probability_grid / (1 - probability_grid))
    return [occupancy_grid, logit_grid]


def occupancy_map(data,
                  res=0.2,
                  alpha=0.4,
                  locc=0.85,
                  lfree=-0.4,
                  max_range=60,
                  prop_upper_saturation_threshold=0.98,
                  prop_lower_saturation_threshold=0.02,
                  occupied_threshold = 0.8,
                  frees_threshold = 0.35):

    grid, [min_E, min_N, _, _] = initialize_grid(data, res)
    car_initial_lla = [data['lat'][0], data['lon'][0], data['alt'][0]]
    car_initial_pixel_location = [-np.round(min_N / 0.2), -np.round(min_E / 0.2)]
    cell_E, cell_N = np.meshgrid(np.array(range(grid.shape[1])), np.array(range(grid.shape[0])))
    NE_grid = (np.stack([cell_N, cell_E]).T - car_initial_pixel_location).T * 0.2
    print(f'NE shape: {NE_grid.shape}')
    
    current_occupancy_grid = 0.5 * np.ones(NE_grid.shape[1:])
    
    update_grids = []
    grids = []
    post_grids = []
    occupancies = []
    measurements = []
    counter = 0

    for data_point in data.iterrows():
        data_point = data_point[1]
        car_current_lla = [data_point['lat'], data_point['lon'], data_point['alt']]
        current_location_NED = navpy.lla2ned(*(car_current_lla + car_initial_lla))[0:2]
       # Displacement from initial location
        current_pixel_location = np.round(current_location_NED / 0.2) + car_initial_pixel_location
        relative_grid = (NE_grid.T - current_location_NED).T                                         # Relative locations to the cards
        cell_distance = np.sqrt(relative_grid[0, :, :] ** 2 + relative_grid[1, :, :] ** 2)

        # looking only on close grid to overcome memory and time constraints
        min_row = int(max(current_pixel_location[0] - max_range / res, 0)) 
        max_row = int(min(current_pixel_location[0] + max_range / res, grid.shape[0]))
        min_col = int(max(current_pixel_location[1] - max_range / res, 0))
        max_col = int(min(current_pixel_location[1] + max_range / res, grid.shape[1]))
        local_grid = relative_grid[:, min_row: max_row, min_col: max_col]
        local_cell_distance = cell_distance[min_row: max_row, min_col: max_col]
        local_cell_center_directions = local_grid / local_cell_distance

        # mesurements
        measurments = np.vstack([data_point['North'], data_point['East']])
        measurement_correction = current_location_NED
        measurement_correction[1] = -measurement_correction[1]
        corrected_measurments = (measurments.T - current_location_NED).T
        measured_ranges = np.sqrt(corrected_measurments[0] ** 2 + corrected_measurments[1] ** 2)
        lidar_rays = corrected_measurments / measured_ranges #These are only measurments that hit above the ground twice
        measurements.append(corrected_measurments)

        # matching rays and cells:
        ray_cell_inner_products = np.einsum('kij, kl -> ijl', local_cell_center_directions, lidar_rays) # shape of local grid X rays
        # taking care of numerical errors
        ray_cell_inner_products[ray_cell_inner_products > 1] = 1
        ray_cell_inner_products[ray_cell_inner_products < -1] = -1
        
        ray_cell_angles = np.arccos(ray_cell_inner_products) # angles between rays and cell center
        closest_angle = np.min(ray_cell_angles, axis=2)      # for each cell the angle between in and the closest ray
        distance_from_closest_ray = np.sin(closest_angle) * local_cell_distance
        cells_touched_by_ray = distance_from_closest_ray 
        cells_touched_by_ray[cells_touched_by_ray >= res ] = 0
        cells_touched_by_ray[cells_touched_by_ray < res ] = 1 # A binary array for, for each cell if there is an active ray touching it
        cell_closest_ray_range = measured_ranges[np.argmin(ray_cell_angles, axis=2)]

        update_grid = np.zeros(local_grid.shape[1:])
        update_grid[np.logical_and(cell_closest_ray_range < max_range, np.abs(local_cell_distance - cell_closest_ray_range) < alpha / 2)] = locc 
        update_grid[local_cell_distance < (cell_closest_ray_range - alpha / 2)] = lfree
        update_grid[np.logical_not(cells_touched_by_ray)] = 0
        update_grids.append(update_grid)

        grid[min_row: max_row, min_col: max_col] = grid[min_row: max_row, min_col: max_col] + update_grid
        grids.append(grid.copy())

        [occupancy_grid, grid] = proceess_logit_grid(grid, current_occupancy_grid, prop_upper_saturation_threshold, prop_lower_saturation_threshold, occupied_threshold, frees_threshold)

        occupancies.append(occupancy_grid)
        post_grids.append(grid.copy())
        counter += 1
        print(f'frame: {counter} processed')
        
    return update_grids, grids, occupancies