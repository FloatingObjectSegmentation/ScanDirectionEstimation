import common

lidar_folder = 'E:/workspaces/LIDAR_WORKSPACE/lidar'
name = "386_95"

names = common.get_dataset_names(lidar_folder)
names.sort()
for name in names:
    print(name)
    dataset = common.RawLidarDatasetNormXYZRGBAngle(lidar_folder, name)
    bmp = common.Visualization().transform_dataset_to_scananglebmp(dataset, bmpsize=4000)
    common.HoughTransform().visualize_scananglematrix(bmp)