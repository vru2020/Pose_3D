
import time
import numpy as np
import torch


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))
        
        
        
def zero_the_root(pose, root_idx):
    if isinstance(pose, np.ndarray):
        # center at root
        root_pose = []
        for i in range(pose.shape[0]):
            root_pose.append (pose[i, root_idx, :])
            pose[i, :, :] = pose[i, :, :] - pose[i, root_idx, :]              
        # remove root
        pose = np.delete(pose, root_idx, 1)  # axis [n, j, x/y]
        root_pose = np.asarray (root_pose)
        
        return pose, root_pose
    
    elif torch.is_tensor(pose):
       pose1 = pose.clone ()
       root_pose = pose1[:,root_idx,:].reshape (pose.shape[0],-1,pose.shape[2])

       for i in range(pose.shape[0]):
           pose1[i, :, :] = pose1[i, :, :] - pose1[i, root_idx, :]          
       pose1 = pose1 [:,1:,:]   
       return pose1#, root_pose
        
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")
        

def zero_the_root_torch(pose, root_idx):
      
   for i in range(pose.shape[0]):
       pose[i, :, :] = pose[i, :, :] - pose[i, root_idx, :]          
       
       return pose
        
    
def add_the_root(pose, root,root_idx):
    pose_new = torch.cat ([root,pose],1)
    pose_new [:,1:,:] = pose_new [:,1:,:] + root
    return pose_new
