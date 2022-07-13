import numpy as np
import torch


#%%
def get_all_views_2d (batch):
    
    pose2d_4 = batch ['pose2d'].reshape (-1,4,17,2)#.to(config.device)#.detach().numpy ()
      
    pose2d_view1 = pose2d_4[:,0,:,:]
    pose2d_view2 = pose2d_4[:,1,:,:]
    pose2d_view3 = pose2d_4[:,2,:,:]
    pose2d_view4 = pose2d_4[:,3,:,:]
    
    return pose2d_view1,pose2d_view2,pose2d_view3,pose2d_view4


def get_all_views_3d (batch):
    
    #pose3d_4 = batch ['pose3d'].to(config.device).reshape (-1,4,17,3)#.detach().numpy ()
    pose3d_4 = batch ['pose3d'].reshape (-1,4,17,3)
      
    pose3d_view1 = pose3d_4[:,0,:,:]
    pose3d_view2 = pose3d_4[:,1,:,:]
    pose3d_view3 = pose3d_4[:,2,:,:]
    pose3d_view4 = pose3d_4[:,3,:,:]
    
    return pose3d_view1,pose3d_view2,pose3d_view3,pose3d_view4



def get_all_views_3d_tri (batch):
    
    #pose3d_4 = batch ['pose3d'].to(config.device).reshape (-1,4,17,3)#.detach().numpy ()
    pose3d_4 = batch ['pose3d_tri'].reshape (-1,4,17,3)
      
    pose3d_view1 = pose3d_4[:,0,:,:]
    pose3d_view2 = pose3d_4[:,1,:,:]
    pose3d_view3 = pose3d_4[:,2,:,:]
    pose3d_view4 = pose3d_4[:,3,:,:]
    
    return pose3d_view1,pose3d_view2,pose3d_view3,pose3d_view4



#%%
def rotation_matrices_batch (batch,rotation_matrices):
     subjects = batch ['subject'].reshape (-1,4)
     
     rot_matrix_batch = []
     for i in range (len(subjects)):
         item_sbj = int (subjects[i,0])
         rot_matrix = torch.tensor([rotation_matrices[key] for key in \
                                    [(item_sbj,1),(item_sbj,2),(item_sbj,3),(item_sbj,4)]])
         rot_matrix_batch.append (rot_matrix)
    
     rot_matrix_batch = torch.stack (rot_matrix_batch)
     return rot_matrix_batch
 
def translation_vectors_batch (batch,translation_vectors):
     subjects = batch ['subject'].reshape (-1,4)
     
     trans_vec_batch = []
     for i in range (len(subjects)):
         item_sbj = int (subjects[i,0])
         trans_vec = torch.tensor([translation_vectors[key] for key in \
                                    [(item_sbj,1),(item_sbj,2),(item_sbj,3),(item_sbj,4)]])
         trans_vec_batch.append (trans_vec)
    
     trans_vec_batch = torch.stack (trans_vec_batch)
     return trans_vec_batch
 
#%%

def homogeneous_to_euclidean(points):
    
    'https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/utils/multiview.py'
    
    """Converts homogeneous points to euclidean
    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M
    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")
        
def euclidean_to_homogeneous(points):
    
    'https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/utils/multiview.py'
    
    """Converts euclidean points to homogeneous
    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M
    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")
        
        

def project_3d_points_to_image_plane_without_distortion(proj_matrix, points_3d, convert_back_to_euclidean=True):
    
    'https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/utils/multiview.py'
    
    """Project 3D points to image plane not taking into account distortion
    # points3d must be in the gloabl (world) coordinate frame
    Args:
        proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
        points_3d numpy array or torch tensor of shape (N, 3): 3D points
        convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                        NOTE: division by zero can be here if z = 0
    Returns:
        numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
        #proj_matrix = proj_matrix.float ()
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


#%%    

def rotate_3d_view (points_batch,rotation_batch, translation_batch,view_i,view_j):
     batch_size = points_batch.shape[0]
     poses_output = []
     for i in range(batch_size):
         
         rot = rotation_batch[i][view_j,:,:] @ (rotation_batch[i][view_i,:,:].t())
         
         tran = rotation_batch[i][view_i,:,:] @ (translation_batch[i][view_j,:,:] - translation_batch[i][view_i,:,:])
         
         view2 = rot @ (points_batch[i].t() - tran)
         poses_output.append (view2.t())
   
     poses_output = torch.stack(poses_output, dim=0)
     
     return poses_output



def project_by_view (pose3d,proj_matrix):
    
    pose3d_11,pose3d_12,pose3d_13,pose3d_14 = ([] for i in range(4))

    for i in range(len(pose3d)):
        pose3d_11.append(project_3d_points_to_image_plane_without_distortion (proj_matrix [0],pose3d[i]))
    
        pose3d_12.append(project_3d_points_to_image_plane_without_distortion (proj_matrix [1],pose3d[i]))
    
        pose3d_13.append(project_3d_points_to_image_plane_without_distortion (proj_matrix [2],pose3d[i]))
    
        pose3d_14.append(project_3d_points_to_image_plane_without_distortion (proj_matrix [3],pose3d[i]))

    
    pose3d_11 = torch.stack(pose3d_11, dim=0)
    pose3d_12 = torch.stack(pose3d_12, dim=0)
    pose3d_13 = torch.stack(pose3d_13, dim=0)
    pose3d_14 = torch.stack(pose3d_14, dim=0)
    
    
    return pose3d_11,pose3d_12,pose3d_13,pose3d_14



def project_by_view_batch (pose3d,proj_matrices_batch):
    
    pose3d_11,pose3d_12,pose3d_13,pose3d_14 = ([] for i in range(4))

    for i in range(len(pose3d)):
        pose3d_11.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][0],pose3d[i]))
    
        pose3d_12.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][1],pose3d[i]))
    
        pose3d_13.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][2],pose3d[i]))
    
        pose3d_14.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][3],pose3d[i]))

    
    pose3d_11 = torch.stack(pose3d_11, dim=0)
    pose3d_12 = torch.stack(pose3d_12, dim=0)
    pose3d_13 = torch.stack(pose3d_13, dim=0)
    pose3d_14 = torch.stack(pose3d_14, dim=0)
    
    
    return pose3d_11,pose3d_12,pose3d_13,pose3d_14



 
#%%
    
def camera_to_world_torch(P, R, T):
  #P = 
  X_cam = R.t()  @  P.t().double() + T  
  return X_cam. t()

def world_to_camera_torch(P, R, T):
  X_cam = R @  (P.t() - T) .double()   
  return X_cam. t()


def camera_to_world_torch_batch(pose3d, rotation, translation):
    pose3d_glob = []
    
    for i in range(len(pose3d)):
        pose3d_glob.append(camera_to_world_torch (pose3d[i],rotation,translation))
      
    pose3d_glob = torch.stack(pose3d_glob, dim=0)
  
    return pose3d_glob


def world_to_camera_torch_batch(pose3d, rotation, translation):
    pose3d_cam = []
    
    for i in range(len(pose3d)):
        pose3d_cam.append(world_to_camera_torch(pose3d[i],rotation,translation))
     
    pose3d_cam = torch.stack(pose3d_cam, dim=0)
  
    return pose3d_cam


def world_to_camera_batch (points_batch, rotation_batch, translation_batch):
    batch_size = points_batch.shape[0]
    
    poses_camera1, poses_camera2, poses_camera3, poses_camera4 = ( [] for i in range (4))
    for i in range(batch_size):
        poses_camera1.append (world_to_camera_torch(points_batch[i], rotation_batch[i][0],  translation_batch[i][0]))
        poses_camera2.append (world_to_camera_torch(points_batch[i], rotation_batch[i][1],  translation_batch[i][1]))
        poses_camera3.append (world_to_camera_torch(points_batch[i], rotation_batch[i][2],  translation_batch[i][2]))
        poses_camera4.append (world_to_camera_torch(points_batch[i], rotation_batch[i][3],  translation_batch[i][3]))
    
    poses_camera1 = torch.stack(poses_camera1, dim=0)
    poses_camera2 = torch.stack(poses_camera2, dim=0)
    poses_camera3 = torch.stack(poses_camera3, dim=0)
    poses_camera4 = torch.stack(poses_camera4, dim=0)
    
    return poses_camera1, poses_camera2, poses_camera3, poses_camera4




def camera_to_world_batch (points_batch,rotation_batch, translation_batch):
    batch_size = points_batch.shape[0]
    
    poses_world1, poses_world2, poses_world3, poses_world4 = ( [] for i in range (4))
    for i in range(batch_size):
        poses_world1.append (camera_to_world_torch(points_batch[i], rotation_batch[i][0],  translation_batch[i][0]))
        poses_world2.append (camera_to_world_torch(points_batch[i], rotation_batch[i][1],  translation_batch[i][1]))
        poses_world3.append (camera_to_world_torch(points_batch[i], rotation_batch[i][2],  translation_batch[i][2]))
        poses_world4.append (camera_to_world_torch(points_batch[i], rotation_batch[i][3],  translation_batch[i][3]))
    
    poses_world1 = torch.stack(poses_world1, dim=0)
    poses_world2 = torch.stack(poses_world2, dim=0)
    poses_world3 = torch.stack(poses_world3, dim=0)
    poses_world4 = torch.stack(poses_world4, dim=0)
    
    return poses_world1, poses_world2, poses_world3, poses_world4
        






































