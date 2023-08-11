# pythonw vis_fru.py
# from operator import gt
import pickle
import numpy as np
# from omegaconf import DictConfig
from mayavi import mlab
from collections import Counter
# path = r'n008-2018-08-28-16-16-48-0400__LIDAR_TOP__1535488206297315.pcd.bin'
# points = np.fromfile(path, dtype=np.float16).reshape(-1, 5)
# print(points.shape)
import argparse
point_cloud_range = [-50, -50, -2, 50, 50, 5]
voxel_size=[0.2, 0.2, 0.2]
voxel_shape=(int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0]),
             int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1]),
             int((point_cloud_range[5]-point_cloud_range[2])/voxel_size[2]))
map_label = {0: 0,
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 1,
                6: 1,
                7: 1,
                8: 1,
                9: 2,
                10: 2,
                11: 2,
                12: 2,
                13: 2,
                14: 3,
                15: 3,
                16: 3,
                17: 3,
                18: 3,
                19: 3,
                20: 3,
                21: 3,
                22: 3,
                23: 3,
                24: 4,
                25: 4,
                26: 4,
                27: 4,
                28: 4,
                29: 4,
                30: 4,
                31: 3}
def remove_far(points, point_cloud_range):
    mask = (points[:, 0]>point_cloud_range[0]) & (points[:, 0]<point_cloud_range[3]) & (points[:, 1]>point_cloud_range[1]) & (points[:, 1]<point_cloud_range[4]) \
            & (points[:, 2]>point_cloud_range[2]) & (points[:, 2]<point_cloud_range[5])
    return points[mask, :]

def voxelize(voxel: np.array, label_count: np.array):
    '''
    '''
    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                if label_count[x, y, z] == 0:
                    continue
                labels = voxel[x, y, z]
                if np.unique(labels).shape[0] == 0:
                    # import ipdb; ipdb.set_trace()
                    assert False
                    continue
                # import ipdb
                # ipdb.set_trace()
                # print(np.argmax(np.bincount(labels[labels!=0])))
                try:
                    label_count[x, y, z] = np.argmax(np.bincount(labels[labels!=0]))
                except:
                    print(labels)
    return label_count

def points2voxel(points, voxel_shape, voxel_size, max_points=5, specific_category=None):
    voxel = np.zeros((*voxel_shape, max_points), dtype=np.int64)
    label_count = np.zeros((voxel_shape), dtype=np.int64)
    index = points[:, 4].argsort()
    points = points[index]
    for point in points:
      
        x, y, z = point[0], point[1], point[2]
        x = round((x - point_cloud_range[0]) / voxel_size[0])
        y = round((y - point_cloud_range[1]) / voxel_size[1])
        z = round((z - point_cloud_range[2]) / voxel_size[2])
        if point[4] == 31:
            continue
        if specific_category and int(point[4]) not in  specific_category:
            continue
        try:
            voxel[x, y, z, label_count[x, y, z]] = int(point[4])  # map_label[int(point[4])]
            label_count[x, y, z] += 1
        except:
            # import ipdb
            # ipdb.set_trace()
            continue

    voxel = voxelize(voxel, label_count)
    label_count[label_count==max_points] = 0
    voxel = voxel.astype(np.float64)
    # from IPython import embed
    # embed()
    # exit()
    return voxel



# voxel = points2voxel(points, voxel_shape, voxel_size, 100)
def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def draw(
    voxels,
    T_velo_2_cam,
    vox_origin,
    fov_mask,
    img_size,
    f,
    voxel_size=0.2,
    d=7,  # 7m - determine the size of the mesh representing the camera
):
    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ]
    )
    # tri_points = np.hstack([tri_points, np.ones((5, 1))])
    # tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0]
    y = tri_points[:, 1]
    z = tri_points[:, 2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # # Get the voxels outside FOV
    # outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)
    ]
    # outfov_voxels = outfov_grid_coords[
    #     (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)
    # ]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))

    # Draw the camera
    # mlab.triangular_mesh(
    #     x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5
    # )

    
    # counter = Counter(list(fov_voxels[:,3].reshape(-1)))
    # for key in counter:
    #     if counter[key] < 100:
    #         index = fov_voxels[:,3] != key
    #         fov_voxels = fov_voxels[index]
    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    # Draw occupied outside FOV voxels
    # plt_plot_outfov = mlab.points3d(
    #     outfov_voxels[:, 0],
    #     outfov_voxels[:, 1],
    #     outfov_voxels[:, 2],
    #     outfov_voxels[:, 3],
    #     colormap="viridis",
    #     scale_factor=voxel_size - 0.05 * voxel_size,
    #     mode="cube",
    #     opacity=1.0,
    #     vmin=1,
    #     vmax=19,
    # )

    classname_to_color = {  # RGB.
        "noise": (0, 0, 0),  # Black.
        "animal": (70, 130, 180),  # Steelblue
        "human.pedestrian.adult": (0, 0, 230),  # Blue
        "human.pedestrian.child":(0, 0, 230),  # Skyblue,
        "human.pedestrian.construction_worker":(0, 0, 230),  # Cornflowerblue
        "human.pedestrian.personal_mobility": (0, 0, 230),  # Palevioletred
        "human.pedestrian.police_officer":(0, 0, 230),  # Navy,
        "human.pedestrian.stroller": (0, 0, 230),  # Lightcoral
        "human.pedestrian.wheelchair": (0, 0, 230),  # Blueviolet
        "movable_object.barrier": (112, 128, 144),  # Slategrey
        "movable_object.debris": (112, 128, 144),  # Chocolate
        "movable_object.pushable_pullable":(112, 128, 144),  # Dimgrey
        "movable_object.trafficcone":(112, 128, 144),  # Darkslategrey
        "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
        "vehicle.bicycle": (220, 20, 60),  # Crimson
        "vehicle.bus.bendy":(255, 158, 0),  # Coral
        "vehicle.bus.rigid": (255, 158, 0),  # Orangered
        "vehicle.car": (255, 158, 0),  # Orange
        "vehicle.construction":(255, 158, 0),  # Darksalmon
        "vehicle.emergency.ambulance":(255, 158, 0),
        "vehicle.emergency.police": (255, 158, 0),  # Gold
        "vehicle.motorcycle": (255, 158, 0),  # Red
        "vehicle.trailer":(255, 158, 0),  # Darkorange
        "vehicle.truck": (255, 158, 0),  # Tomato
        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
        "flat.other":(0, 207, 191),
        "flat.sidewalk": (75, 0, 75),
        "flat.terrain": (0, 207, 191),
        "static.manmade": (222, 184, 135),  # Burlywood
        "static.other": (0, 207, 191),  # Bisque
        "static.vegetation": (0, 175, 0),  # Green
        "vehicle.ego": (255, 240, 245)
    }
    
    classname_to_color= {'ignore_class': (0, 0, 0),  # Black.
                'barrier': (112, 128, 144),  # Slategrey
                'bicycle': (220, 20, 60),  # Crimson
                'bus': (255, 127, 80),  # Coral
                'car': (255, 158, 0),  # Orange
                'construction_vehicle': (233, 150, 70),  # Darksalmon
                'motorcycle': (255, 61, 99),  # Red
                'pedestrian': (0, 0, 230),  # Blue
                'traffic_cone': (47, 79, 79),  # Darkslategrey
                'trailer': (255, 140, 0),  # Darkorange
                'truck': (255, 99, 71),  # Tomato
                'driveable_surface': (0, 207, 191),  # nuTonomy green
                'other_flat': (175, 0, 75),
                'sidewalk': (75, 0, 75),
                'terrain': (112, 180, 60),
                'manmade': (222, 184, 135),  # Burlywood
                'vegetation': (0, 175, 0)}
    colors = np.array(list(classname_to_color.values())).astype(np.uint8)
    alpha = np.ones((colors.shape[0], 1), dtype=np.uint8) * 255
    colors = np.hstack([colors, alpha])



    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    # plt_plot_outfov.glyph.scale_mode = "scale_by_vector"

    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    plt_plot_fov.module_manager.scalar_lut_manager.data_range = [0, 17]

    mlab.show()

def voxel_exist(voxels, x,y,z):
    if x < 0 or y < 0 or z < 0 or x >= voxels.shape[0] or y >= voxels.shape[1] or z >= voxels.shape[2]:
        return False
    else:
        return voxels[x,y,z]

def max_connected(voxels, distance=3):
    """ Keep the max connected component of the voxels (a boolean matrix). 
    distance is the distance considered as neighbors, i.e. if distance = 2, 
    then two blocks are considered connected even with a hole in between"""
    assert(distance > 0)
    component_list = []
    # max_component = np.zeros(voxels.shape)
    voxels_copy = np.copy(voxels)
    for startx in range(voxels.shape[0]):
        for starty in range(voxels.shape[1]):
            for startz in range(voxels.shape[2]):
                if not voxels_copy[startx,starty,startz]:
                    continue
                # start a new component
                component = np.zeros(voxels.shape, dtype=bool)
                stack = [[startx,starty,startz]]
                component[startx,starty,startz] = True
                voxels_copy[startx,starty,startz] = False
                while len(stack) > 0:
                    x,y,z = stack.pop()
                    category = voxels[x,y,z]
                    for i in range(x-distance, x+distance + 1):
                        for j in range(y-distance, y+distance + 1):
                            for k in range(z-distance, z+distance + 1):
                                if (i-x)**2+(j-y)**2+(k-z)**2 > distance * distance:
                                    continue
                                category = voxels[x,y,z]
                                if voxel_exist(voxels_copy, i,j,k) and voxels[i,j,k] == category:
                                    voxels_copy[i,j,k] = False
                                    component[i,j,k] = True
                                    stack.append([i,j,k])
                component_list.append(component)
                # if component.sum() > max_component.sum():
                #     max_component = component
                    

    max_component = np.zeros(voxels.shape,  dtype=bool)
    for each in component_list:
        if each.sum()>10:
            max_component |= each
    return max_component 

# points = remove_far(points, point_cloud_range)
def main(filepath='*.npz'):

    vox_origin = np.array([0, 0, -2])


    # y_pred = points2voxel(points, voxel_shape, voxel_size, 20)
    # y_del = ~max_connected(y_pred)
    # y_pred[y_del] = 0
   
    if filepath.endswith('npy'):
        y_pred = np.load(filepath)
    elif filepath.endswith('npz'):
        y_pred = np.load(filepath)['pred']# ['semantics']

    # y_pred: shape 200x200x16
    draw(
        y_pred,
        None,
        vox_origin,
        None,
        voxel_size=0.2,
        f=552.55426,
        img_size=(1600, 900),
        d=7,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vis occ')
    parser.add_argument('path', help='path to npz')
    args = parser.parse_args()
    main(args.path)