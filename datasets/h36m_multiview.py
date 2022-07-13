import gc
import os
import tqdm

import h5py
import torch
from torch.utils.data import Dataset



class H36M_Dataset(Dataset):

    def __init__(self, subjects, annotation_file,device='cpu', annotation_path=None,
                 train=False, filter_actions="all"):
        """[summary]

        Arguments:
            subjects {list} -- IDs of subjects to include in this dataset
            annotation_file {str} -- file name (debug_h36m_17, h36m17 etc)


        Keyword Arguments:           
            device {str} -- (default: {'cpu'})
            annotation_path {str} -- path to the annotation_file
            train {bool} -- triggers data augmentation during training (default: {False})
        """

        
        self.device = device
        self.train = train


        # Data Specific Information
        self.skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                         (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.joint_names = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                            'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.action_names = ["Directions", "Discussion", "Eating", "Greeting", "Phoning", "Photo", "Posing", "Purchases",
                             "Sitting", "SittingDown", "Smoking", "Waiting", "WalkDog", "Walking", "WalkTogether"]

      
        self.flipped_indices = [0, 4, 5, 6, 1, 2, 3,
                                7, 8, 9, 10, 14, 15, 16, 11, 12, 13]

        self.root_idx = self.joint_names.index('Pelvis')

        ignore_data = [] # "bbox", "pose3d_global", "cam_f", "cam_c", "cam_R", "cam_T"

        # get labels and metadata including camera parameters
        subj_name = "".join(str(sub) for sub in subjects)
        if annotation_path:
            annotations_h5 = h5py.File(
                f'{annotation_path}/{annotation_file}_{subj_name}.h5', 'r')
        else:
            annotations_h5 = h5py.File(
                f'{os.path.dirname(os.path.abspath(__file__))}/data/{annotation_file}_{subj_name}.h5', 'r')
            

        # store only the subjects of interest
        self.annotations = {}
        for key in tqdm.tqdm(annotations_h5.keys()):
            if key not in ignore_data:
                self.annotations[key] = annotations_h5[key][:]

        print(f'[INFO]: processing subjects: {subjects}')


        # get keys to avoid query them for every __getitem__
        self.annotation_keys = self.annotations.keys()
        
        for key in self.annotation_keys:
            self.annotations[key] = torch.tensor(
                self.annotations[key], dtype=torch.float32)

      

        # clear the HDF5 datasets
        annotations_h5.close()
        #f.close()
        del annotations_h5
        #del f
        gc.collect()


     
    def __len__(self):
        # idx - index of the image files
        return len(self.annotations['camera'])

    def __getitem__(self, idx):

        # Get all data for a sample
        sample = {}
        for key in self.annotation_keys:
            sample[key] = self.annotations[key][idx]


        return sample


    
    
    def flip(self, sample):
        sample['pose2d'] = sample['pose2d'][self.flipped_indices]
        sample['pose3d'] = sample['pose3d'][self.flipped_indices]
        sample['pose2d'][:, 0] *= -1
        sample['pose3d'][:, 0] *= -1
        # TODO add image flipping
        return sample





if __name__ == "__main__":
   
    
    annotation_file = f'h36m17'
    image_path = f'/media/bouazia/bouazia/Data/h36m_dataset/h36m_pickles.h5'

    dataset = H36M_Dataset([9,11], annotation_file,  train=True)
