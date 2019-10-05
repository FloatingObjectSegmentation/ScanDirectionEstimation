import common
import pickle
import os.path

parallel = 1
crisscross = 2
perpendicular = 3

class DatasetCharacteristics:

    def __init__(self, numlines, style, sparse, badprojection, emptyspace, chaos):
        self.numlines = numlines
        self.style = style
        self.sparse = sparse
        self.badprojection = badprojection
        self.emptyspace = emptyspace
        self.chaos = chaos



specs = [
    DatasetCharacteristics(numlines=[1,2], style=1, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=3, sparse=0, badprojection=0, emptyspace=1, chaos=0),
    DatasetCharacteristics(numlines=[1,2], style=3, sparse=0, badprojection=0, emptyspace=1, chaos=0),
    DatasetCharacteristics(numlines=[1,2], style=3, sparse=0, badprojection=0, emptyspace=0, chaos=0),

    DatasetCharacteristics(numlines=[2], style=3, sparse=1, badprojection=1, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[3], style=2, sparse=1, badprojection=0, emptyspace=0, chaos=1),
    DatasetCharacteristics(numlines=[3], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[1], style=0, sparse=2, badprojection=0, emptyspace=0, chaos=0),

    DatasetCharacteristics(numlines=[2,3], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[1], style=1, sparse=1, badprojection=0, emptyspace=1, chaos=0),
    DatasetCharacteristics(numlines=[2], style=1.5, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=1, sparse=0, badprojection=1, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[1], style=1, sparse=1, badprojection=1, emptyspace=0, chaos=0),

    DatasetCharacteristics(numlines=[2], style=3, sparse=0, badprojection=1, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[1,2,3], style=2, sparse=0, badprojection=1, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[1,2], style=1, sparse=0, badprojection=0, emptyspace=0, chaos=0),

    DatasetCharacteristics(numlines=[5], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=1),
    DatasetCharacteristics(numlines=[2], style=1.5, sparse=0, badprojection=0, emptyspace=2, chaos=0),
    DatasetCharacteristics(numlines=[1,2,3], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=1),
    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),

    DatasetCharacteristics(numlines=[2,3], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[1,2,3], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[1,2], style=1, sparse=0, badprojection=0, emptyspace=0, chaos=0),

    DatasetCharacteristics(numlines=[1,2], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=1, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2,3], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=1.5, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=3, sparse=0, badprojection=1, emptyspace=0, chaos=0),

    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2,3], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[5], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=1),
    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=1, sparse=0, badprojection=1, emptyspace=0, chaos=0),

    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=1, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=1, sparse=0, badprojection=1, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=1, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=1, sparse=0, badprojection=0.5, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=1, sparse=1, badprojection=0, emptyspace=1, chaos=0),

    DatasetCharacteristics(numlines=[2], style=1.5, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[1,2], style=3, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[2], style=1, sparse=0, badprojection=0, emptyspace=0, chaos=0),
    DatasetCharacteristics(numlines=[3], style=2, sparse=0, badprojection=0, emptyspace=0, chaos=0),

    DatasetCharacteristics(numlines=[2], style=1.5, sparse=0, badprojection=0, emptyspace=0, chaos=0)
]

points = []
lidar_folder = 'E:/workspaces/LIDAR_WORKSPACE/lidar'
tempfile = 'E:\\workspaces\\LIDAR_WORKSPACE\\temp\\coorcheenah.bin'
augmentable_folder_swath_sols = 'E:/workspaces/LIDAR_WORKSPACE/augmentation/augmentables_scantraces_solutions'

# if not os.path.isfile(tempfile):
#     names = common.get_dataset_names(lidar_folder)
#     names.sort()
#
#     for name in names:
#         dataset = common.RawLidarDatasetNormXYZRGBAngle(lidar_folder, name)
#         points.append(len(dataset.points))
#         dataset = None
#     pickle.dump(points, open(tempfile, 'wb'))
# points = pickle.load(open(tempfile, 'rb'))

names = common.get_dataset_names(lidar_folder)
names.sort()



for i in range(len(names)):
    line = ''
    augs = common.AugmentableSet(augmentable_folder_swath_sols, names[i])

    a = names[i].split('_')
    line += a[0] + '\\_' + a[1] + ' & '
    line += str(len(augs.augmentables)) + ' & '
    line += ','.join([str(l) for l in specs[i].numlines]) + ' & '
    line += str(specs[i].style) + ' & '
    line += str(specs[i].sparse) + ' & '
    line += str(specs[i].badprojection) + ' & '
    line += str(specs[i].emptyspace) + ' & '
    line += str(specs[i].chaos) + '\\\\ \n'
    line += '\\hline'
    print(line)