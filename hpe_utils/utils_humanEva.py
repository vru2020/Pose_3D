
import torch

from camera_humaneva import Camera, load_cameras
from triangulate import project_3d_points_to_image_plane_without_distortion
import numpy as np

from utils_multiview import camera_to_world_torch,world_to_camera_torch
#%%
def get_all_views_2d (batch):
    
    pose2d_4 = batch ['pose2d'].reshape (-1,3,15,2)
      
    pose2d_view1 = pose2d_4[:,0,:,:]
    pose2d_view2 = pose2d_4[:,1,:,:]
    pose2d_view3 = pose2d_4[:,2,:,:]
    
    return pose2d_view1,pose2d_view2,pose2d_view3

def get_all_views_2d_det (batch):
    
    pose2d_4 = batch ['pose2d_det'].reshape (-1,3,15,2)
      
    pose2d_view1 = pose2d_4[:,0,:,:]
    pose2d_view2 = pose2d_4[:,1,:,:]
    pose2d_view3 = pose2d_4[:,2,:,:]
    
    return pose2d_view1,pose2d_view2,pose2d_view3


def get_all_3d_tri (batch):
    
    pose3d_4 = batch ['pose3d_tri'].reshape (-1,3,15,3)
      
    pose3d_view1 = pose3d_4[:,0,:,:]
    pose3d_view2 = pose3d_4[:,1,:,:]
    pose3d_view3 = pose3d_4[:,2,:,:]
    
    return pose3d_view1,pose3d_view2,pose3d_view3

def get_all_3d_tri_det (batch):
    
    pose3d_4 = batch ['pose3d_tri_det'].reshape (-1,3,15,3)
      
    pose3d_view1 = pose3d_4[:,0,:,:]
    pose3d_view2 = pose3d_4[:,1,:,:]
    pose3d_view3 = pose3d_4[:,2,:,:]
    
    return pose3d_view1,pose3d_view2,pose3d_view3



def get_all_views_3d (batch):
    
    pose3d_4 = batch ['pose3d'].reshape (-1,3,15,3)
      
    pose3d_view1 = pose3d_4[:,0,:,:]
    pose3d_view2 = pose3d_4[:,1,:,:]
    pose3d_view3 = pose3d_4[:,2,:,:]
    
    return pose3d_view1,pose3d_view2,pose3d_view3


def get_3d_global (batch):
    
    pose3d_4 = batch ['pose3d_global'].reshape (-1,3,15,3)
      
    pose3d_view1 = pose3d_4[:,0,:,:]
    
    return pose3d_view1

#%%
# def generate_projection_matrices ():
    
#     keys = list(load_cameras('/external/10g/tk/fs1/HPE3D_data/HumanEva_I/preprocessed/cameras.h5', 
#                              subjects=[1,2,3,4] ).keys ())
#     projections = []
#     projection_matrices = {}
    
#     for sbj in ([1,2,3,4]):
#         for cam in [1,2,3]: 
#             camera_params = load_cameras('/external/10g/tk/fs1/HPE3D_data/HumanEva_I/preprocessed/cameras.h5', subjects=[sbj] )[(sbj,cam)]
#             Cam_param = Camera (camera_params)
#             projection_matrix = Cam_param.projection_matrix
#             projections.append (projection_matrix)   
    
#     for (i,key) in enumerate (keys):
#         projection_matrices[key] = projections[i][:]
#     return projection_matrices


def generate_projection_matrices ():
    Rotations = generate_cam_extrinsics() [0]
    Translations = generate_cam_extrinsics() [1]
    
    fx1 = 765.789418 
    fy1 = 765.330306 
    
    cx1 = 299.773675 
    cy1 = 232.347455 
    
    K1 = np.array([[fx1, 0., cx1],[0., fy1, cy1],[0., 0., 1.]]).astype(np.double)
    
    fx2 = 688.411220  
    fy2 = 686.786141  
    
    cx2 = 273.089872 
    cy2 = 221.974930  
    
    K2 = np.array([[fx2, 0., cx2],[0., fy2, cy2],[0., 0., 1.]]).astype(np.double)
    
    
    fx3 = 724.267662 
    fy3 = 723.569727 
    
    cx3 = 302.629652 
    cy3 = 204.912033 
    
    K3 = np.array([[fx3, 0., cx3],[0., fy3, cy3],[0., 0., 1.]]).astype(np.double)
    
    
    R11 = Rotations [(1,1)]   
    T11 = Translations [(1,1)]   
    
    R12 = Rotations [(1,2)]   
    T12 = Translations [(1,2)]  
    
    R13 = Rotations [(1,3)]   
    T13 = Translations [(1,3)]  
    
    
    R21 = Rotations [(2,1)]   
    T21 = Translations [(2,1)]   
    
    R22 = Rotations [(2,2)]   
    T22 = Translations [(2,2)]  
    
    R23 = Rotations [(2,3)]   
    T23 = Translations [(2,3)] 
    
    
    R31 = Rotations [(3,1)]   
    T31 = Translations [(3,1)]   
    
    R32 = Rotations [(3,2)]   
    T32 = Translations [(3,2)]  
    
    R33 = Rotations [(3,3)]   
    T33 = Translations [(3,3)] 
    
    R41 = Rotations [(4,1)]   
    T41 = Translations [(4,1)]   
    
    R42 = Rotations [(4,2)]   
    T42 = Translations [(4,2)]  
    
    R43 = Rotations [(4,3)]   
    T43 = Translations [(4,3)] 
    
    
    P11 = np.dot(K1, np.concatenate((R11, T11), axis=1))
    P12 = np.dot(K1, np.concatenate((R12, T12), axis=1))
    P13 = np.dot(K1, np.concatenate((R13, T13), axis=1))
    
    P21 = np.dot(K2, np.concatenate((R21, T21), axis=1))
    P22 = np.dot(K2, np.concatenate((R22, T22), axis=1))
    P23 = np.dot(K2, np.concatenate((R23, T23), axis=1))
    
    P31 = np.dot(K3, np.concatenate((R31, T31), axis=1))
    P32 = np.dot(K3, np.concatenate((R32, T32), axis=1))
    P33 = np.dot(K3, np.concatenate((R33, T33), axis=1))
    
    P41 = np.dot(K3, np.concatenate((R41, T41), axis=1))
    P42 = np.dot(K3, np.concatenate((R42, T42), axis=1))
    P43 = np.dot(K3, np.concatenate((R43, T43), axis=1))
    
    
    Projections = {(1, 1): P11, (1, 2): P12,(1, 3): P13,
                   (2, 1): P21, (2, 2): P22,(2, 3): P23,
                   (3, 1): P31, (3, 2): P32,(3, 3): P33,
                   (4, 1): P41, (4, 2): P42,(4, 3): P43}
    
    return Projections

def generate_cam_extrinsics ():
    
    keys = list(load_cameras('/external/10g/tk/fs1/HPE3D_data/HumanEva_I/preprocessed/cameras.h5', 
                             subjects=[1,2,3,4] ).keys ())
    Rotations, Translations = ([] for i in range(2))
    Rot_matrices, Trans_matrices = ({} for i in range(2))
    
    for sbj in ([1,2,3,4]): 
        for cam in [1,2,3]: 
            camera_params = load_cameras('/external/10g/tk/fs1/HPE3D_data/HumanEva_I/preprocessed/cameras.h5', subjects=[sbj] )[(sbj,cam)]
            Cam_param = Camera (camera_params)
            Rotations.append (Cam_param.R)
            Translations.append (Cam_param.T)
    
    
    for (i,key) in enumerate (keys):
        Rot_matrices[key] = Rotations[i][:]
        Trans_matrices[key] = Translations[i][:]
    return Rot_matrices, Trans_matrices

def projection_matrices_batch (batch,projection_matrices):
     subjects = batch ['subject'].reshape (-1,3)
     
     proj_matrix_batch = []
     for i in range (len(subjects)):
         item_sbj = int (subjects[i,0])
         proj_matrix = torch.tensor([projection_matrices[key] for key in \
                                    [(item_sbj,1),(item_sbj,2),(item_sbj,3)]])
         proj_matrix_batch.append (proj_matrix)
    
     proj_matrix_batch = torch.stack (proj_matrix_batch)
     return proj_matrix_batch
 

def rotation_matrices_batch (batch,rotation_matrices):
     subjects = batch ['subject'].reshape (-1,3)
     
     rot_matrix_batch = []
     for i in range (len(subjects)):
         item_sbj = int (subjects[i,0])
         rot_matrix = torch.tensor([rotation_matrices[key] for key in \
                                    [(item_sbj,1),(item_sbj,2),(item_sbj,3)]])
         rot_matrix_batch.append (rot_matrix)
    
     rot_matrix_batch = torch.stack (rot_matrix_batch)
     return rot_matrix_batch
 
def translation_vectors_batch (batch,translation_vectors):
     subjects = batch ['subject'].reshape (-1,3)
     
     trans_vec_batch = []
     for i in range (len(subjects)):
         item_sbj = int (subjects[i,0])
         trans_vec = torch.tensor([translation_vectors[key] for key in \
                                    [(item_sbj,1),(item_sbj,2),(item_sbj,3)]])
         trans_vec_batch.append (trans_vec)
    
     trans_vec_batch = torch.stack (trans_vec_batch)
     return trans_vec_batch
 

#%%
def project_by_view (pose3d,proj_matrix):
    
    pose3d_11,pose3d_12,pose3d_13 = ([] for i in range(3))

    for i in range(len(pose3d)):
        pose3d_11.append(project_3d_points_to_image_plane_without_distortion (proj_matrix [0],pose3d[i]))
    
        pose3d_12.append(project_3d_points_to_image_plane_without_distortion (proj_matrix [1],pose3d[i]))
    
        pose3d_13.append(project_3d_points_to_image_plane_without_distortion (proj_matrix [2],pose3d[i]))
    
    
    pose3d_11 = torch.stack(pose3d_11, dim=0)
    pose3d_12 = torch.stack(pose3d_12, dim=0)
    pose3d_13 = torch.stack(pose3d_13, dim=0)
    
    
    return pose3d_11,pose3d_12,pose3d_13



def project_by_view_batch (pose3d,proj_matrices_batch):
    
    pose3d_11,pose3d_12,pose3d_13 = ([] for i in range(3))

    for i in range(len(pose3d)):
        pose3d_11.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][0],pose3d[i]))
    
        pose3d_12.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][1],pose3d[i]))
    
        pose3d_13.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][2],pose3d[i]))
    

    
    pose3d_11 = torch.stack(pose3d_11, dim=0)
    pose3d_12 = torch.stack(pose3d_12, dim=0)
    pose3d_13 = torch.stack(pose3d_13, dim=0)
    
    
    return pose3d_11,pose3d_12,pose3d_13

#%%


def camera_to_world_batch (points_batch,rotation_batch, translation_batch):
    batch_size = points_batch.shape[0]
    
    poses_world1, poses_world2, poses_world3 = ( [] for i in range (3))
    for i in range(batch_size):
        poses_world1.append (camera_to_world_torch(points_batch[i], rotation_batch[i][0],  translation_batch[i][0]))
        poses_world2.append (camera_to_world_torch(points_batch[i], rotation_batch[i][1],  translation_batch[i][1]))
        poses_world3.append (camera_to_world_torch(points_batch[i], rotation_batch[i][2],  translation_batch[i][2]))
    
    poses_world1 = torch.stack(poses_world1, dim=0)
    poses_world2 = torch.stack(poses_world2, dim=0)
    poses_world3 = torch.stack(poses_world3, dim=0)
    
    return poses_world1, poses_world2, poses_world3




def world_to_camera_batch (points_batch, rotation_batch, translation_batch):
    batch_size = points_batch.shape[0]
    
    poses_camera1, poses_camera2, poses_camera3 = ( [] for i in range (3))
    for i in range(batch_size):
        poses_camera1.append (world_to_camera_torch(points_batch[i], rotation_batch[i][0],  translation_batch[i][0]))
        poses_camera2.append (world_to_camera_torch(points_batch[i], rotation_batch[i][1],  translation_batch[i][1]))
        poses_camera3.append (world_to_camera_torch(points_batch[i], rotation_batch[i][2],  translation_batch[i][2]))
    
    poses_camera1 = torch.stack(poses_camera1, dim=0)
    poses_camera2 = torch.stack(poses_camera2, dim=0)
    poses_camera3 = torch.stack(poses_camera3, dim=0)
    
    return poses_camera1, poses_camera2, poses_camera3


#%%

def homogeneous_to_euclidean(points):
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


def triangulate_point_from_multiple_views_linear_torch(proj_matricies, points, confidences=None):
    """Similar as triangulate_point_from_multiple_views_linear() but for PyTorch.
    For more information see its documentation.
    Args:
        proj_matricies torch tensor of shape (N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (N, 2): sequence of points' coordinates
        confidences None or torch tensor of shape (N,): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
    Returns:
        point_3d numpy torch tensor of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)

    if confidences is None:
        confidences = torch.ones(n_views, dtype=torch.float32, device=points.device)

    A = proj_matricies[:, 2:3].expand(n_views, 2, 4) * points.view(n_views, 2, 1)
    A -= proj_matricies[:, :2]
    A *= confidences.view(-1, 1, 1)

    u, s, vh = torch.svd(A.view(-1, 4))

    point_3d_homo = -vh[:, 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]

    return point_3d


def triangulate_batch_of_points(proj_matricies_batch, points_batch, confidences_batch=None):
    batch_size, n_views, n_joints = points_batch.shape[:3]
    
    print ("batch_size, n_views, n_joints",batch_size, n_views, n_joints)
    point_3d_batch = torch.zeros(batch_size, n_joints, 3, dtype=torch.float32, device=points_batch.device)

    for batch_i in range(batch_size):
        for joint_i in range(n_joints):
            points = points_batch[batch_i, :, joint_i, :]

            confidences = confidences_batch[batch_i, :, joint_i] if confidences_batch is not None else None
            point_3d = triangulate_point_from_multiple_views_linear_torch(proj_matricies_batch[batch_i], points, confidences=confidences)
            point_3d_batch[batch_i, joint_i] = point_3d

    return point_3d_batch

 