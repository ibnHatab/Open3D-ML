import numpy as np
import logging
from glob import glob
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import Config, make_dir, DATASET


log = logging.getLogger(__name__)

class KITTI_RAW(BaseDataset):
    """This class is used to create a dataset based on the KITTI RAW dataset"""
    def __init__(self,
                dataset_path,
                name='KITTI_RAW',
                cache_dir='./logs/cache',
                use_cache=False,
                val_split=3712,
                test_result_folder='./test',
                **kwargs):

        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         val_split=val_split,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg
        self.num_classes = 3
        self.label_to_names = self.get_label_to_names()

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.all_files = glob(
            join(cfg.dataset_path, 'velodyne_points', 'data', '*.bin'))

        self.all_files.sort()

        self.train_files = []
        self.val_files = []

        for f in self.all_files:
            idx = int(Path(f).name.replace('.bin', ''))
            if idx < cfg.val_split:
                self.train_files.append(f)
            else:
                self.val_files.append(f)

        self.test_files = []

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and values are the corresponding
            names.
        """
        label_to_names = {
            0: 'Pedestrian',
            1: 'Cyclist',
            2: 'Car',
            3: 'Van',
            4: 'Person_sitting',
            5: 'DontCare'
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    def is_tested(self): pass

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return KITTI_RAWSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split in ['train', 'training']:
            return self.train_files
        elif split in ['test', 'testing']:
            return self.test_files
        elif split in ['val', 'validation']:
            return self.val_files
        elif split in ['all']:
            return self.train_files + self.val_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def save_test_result(self, results, attrs): pass

class KITTI_RAWSplit():

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]

        # label_path = pc_path.replace('velodyne',
        #                              'label_2').replace('.bin', '.txt')
        # calib_path = label_path.replace('label_2', 'calib')

        pc = self.dataset.read_lidar(pc_path)
        # calib = self.dataset.read_calib(calib_path)
        # label = self.dataset.read_label(label_path, calib)

        # reduced_pc = DataProcessing.remove_outside_points(
        #     pc, calib['world_cam'], calib['cam_img'], [375, 1242])

        data = {
            'point': pc,
            'full_point': pc,
            'feat': None,
            'calib': None,
            'bounding_boxes': [],
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr

DATASET._register_module(KITTI_RAW)