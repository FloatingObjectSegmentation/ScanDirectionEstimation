import common

lidar_folder = 'E:/workspaces/LIDAR_WORKSPACE/lidar'
name = "391_38"

for name in common.get_dataset_names(lidar_folder):
    dataset = common.LidarDatasetNormXYZRGBAngle(lidar_folder, name)
    bmp, minx, miny = common.PointOperations.transform_dataset_to_bmp(dataset, bmpsize=4000)
    common.HoughTransform.visualize_matrix(bmp)