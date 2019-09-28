import common

lidar_folder = 'E:/workspaces/LIDAR_WORKSPACE/lidar'
name = "391_38"

for name in common.get_dataset_names(lidar_folder):
    name = "391_38"
    dataset = common.LidarDatasetNormXYZRGBAngle(lidar_folder, name)
    bmp = common.PointOperations.transform_dataset_to_scananglebmp(dataset, bmpsize=4000)
    common.HoughTransform.visualize_scananglematrix(bmp)