
import torch
import h5py
from camera_mpii import Camera
from triangulate import project_3d_points_to_image_plane_without_distortion
import numpy as np
import os
from utils_multiview import camera_to_world_torch,world_to_camera_torch


def generate_projection_matrices(cameras=None, subjects=list(range(1, 9)), cams=list(range(14))):
    if cameras == None:
        h5path = os.path.join('/external/10g/tk/fs1/HPE3D_data/poselifter_mpii_inf_3dhp', "cameras.h5")

        cameras = load_mpii_cams(h5path)

    projection_matrices = {}

    for sbj in subjects:
        for cam in cams:
            camera_params = [cameras[sbj, cam][k] for k in ["R", "T", "f", "c", "k", "p", "Name"]]
            Cam_param = Camera(camera_params)
            projection_matrix = Cam_param.projection_matrix
            projection_matrices[sbj, cam] = projection_matrix

    return projection_matrices


def generate_cam_extrinsics(cameras=None, subjects=list(range(1, 9)), cams=list(range(14))):

    if cameras == None:
        h5path = os.path.join('/external/10g/tk/fs1/HPE3D_data/poselifter_mpii_inf_3dhp', "cameras.h5")

        cameras = load_mpii_cams(h5path)
    
    Rotations, Translations = ({} for i in range(2))

    for sbj in subjects:  # 1,5,6,7,8,9,11
        for cam in cams:  # 1,2,3,4
            camera_params = [cameras[sbj, cam][k] for k in ["R", "T", "f", "c", "k", "p", "Name"]]
            Cam_param = Camera(camera_params)
            Rotations[sbj, cam] = Cam_param.R
            Translations[sbj, cam] = Cam_param.T

    return Rotations, Translations


def load_mpii_cams(h5path=None):
    if h5path == None:
        h5path = os.path.join('/external/10g/tk/fs1/HPE3D_data/poselifter_mpii_inf_3dhp', "cameras.h5")

    cams_raw = h5py.File(h5path, "r")

    cams = {}

    for sk, sv in cams_raw.items():
        s_id = int(sk.replace("subject", ""))
        for ck, cv in sv.items():
            c_id = int(ck.replace("camera", ""))
            params = {}
            for pk, pv in cv.items():
                if pk == "k" or pk == "p":
                    pv = None
                else:
                    pv = pv[:]
                params[pk] = pv
            cams[(s_id, c_id)] = params

    return cams


def projection_matrices_batch (batch,projection_matrices):
     subjects = batch ['subject'].reshape (-1,8)
     
     proj_matrix_batch = []
     for i in range (len(subjects)):
         item_sbj = int (subjects[i,0])
         proj_matrix = torch.tensor([projection_matrices[key] for key in \

                                     [(item_sbj, cid) for cid in [0, 1, 2, 4, 5, 6, 7, 8]]])
    
         proj_matrix_batch.append (proj_matrix)
    
     proj_matrix_batch = torch.stack (proj_matrix_batch)
     return proj_matrix_batch
 

def rotation_matrices_batch (batch,rotation_matrices):
     subjects = batch ['subject'].reshape (-1,8)
     
     rot_matrix_batch = []
     for i in range (len(subjects)):
         item_sbj = int (subjects[i,0])
         rot_matrix = torch.tensor([rotation_matrices[key] for key in \
                                    [(item_sbj, cid) for cid in [0, 1, 2, 4, 5, 6, 7, 8]]])
         rot_matrix_batch.append (rot_matrix)
    
     rot_matrix_batch = torch.stack (rot_matrix_batch)
     return rot_matrix_batch


def translation_vectors_batch (batch,translation_vectors):
     subjects = batch ['subject'].reshape (-1,8)
     
     trans_vec_batch = []
     for i in range (len(subjects)):
         item_sbj = int (subjects[i,0])
         trans_vec = torch.tensor([translation_vectors[key] for key in \
                                   [(item_sbj, cid) for cid in [0, 1, 2, 4, 5, 6, 7, 8]]])
         trans_vec_batch.append (trans_vec)
    
     trans_vec_batch = torch.stack (trans_vec_batch)
     return trans_vec_batch
 


def project_by_view_batch (pose3d,proj_matrices_batch):
    
    pose3d_11,pose3d_12,pose3d_13,pose3d_14 = ([] for i in range(4))
    pose3d_15,pose3d_16,pose3d_17,pose3d_18 = ([] for i in range(4))

    for i in range(len(pose3d)):
        pose3d_11.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][0],pose3d[i]))   
        pose3d_12.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][1],pose3d[i]))   
        pose3d_13.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][2],pose3d[i]))
        pose3d_14.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][3],pose3d[i]))
        pose3d_15.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][4],pose3d[i]))
        pose3d_16.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][5],pose3d[i]))
        pose3d_17.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][6],pose3d[i]))
        pose3d_18.append(project_3d_points_to_image_plane_without_distortion (proj_matrices_batch [i][7],pose3d[i]))
   
    pose3d_11 = torch.stack(pose3d_11, dim=0)
    pose3d_12 = torch.stack(pose3d_12, dim=0)
    pose3d_13 = torch.stack(pose3d_13, dim=0)
    pose3d_14 = torch.stack(pose3d_14, dim=0)
    pose3d_15 = torch.stack(pose3d_15, dim=0)
    pose3d_16 = torch.stack(pose3d_16, dim=0)
    pose3d_17 = torch.stack(pose3d_17, dim=0)
    pose3d_18 = torch.stack(pose3d_18, dim=0)
       
    return pose3d_11,pose3d_12,pose3d_13,pose3d_14,\
           pose3d_15,pose3d_16,pose3d_17,pose3d_18

#%%


def camera_to_world_batch (points_batch,rotation_batch, translation_batch):
    batch_size = points_batch.shape[0]
    
    poses_world1, poses_world2, poses_world3, poses_world4 = ( [] for i in range (4))
    poses_world5, poses_world6, poses_world7, poses_world8 = ( [] for i in range (4))


    for i in range(batch_size):
        poses_world1.append (camera_to_world_torch(points_batch[i], rotation_batch[i][0],  translation_batch[i][0]))
        poses_world2.append (camera_to_world_torch(points_batch[i], rotation_batch[i][1],  translation_batch[i][1]))
        poses_world3.append (camera_to_world_torch(points_batch[i], rotation_batch[i][2],  translation_batch[i][2]))
        poses_world4.append (camera_to_world_torch(points_batch[i], rotation_batch[i][3],  translation_batch[i][3]))
        poses_world5.append (camera_to_world_torch(points_batch[i], rotation_batch[i][4],  translation_batch[i][4]))
        poses_world6.append (camera_to_world_torch(points_batch[i], rotation_batch[i][5],  translation_batch[i][5]))
        poses_world7.append (camera_to_world_torch(points_batch[i], rotation_batch[i][6],  translation_batch[i][6]))
        poses_world8.append (camera_to_world_torch(points_batch[i], rotation_batch[i][7],  translation_batch[i][7]))

    poses_world1 = torch.stack(poses_world1, dim=0)
    poses_world2 = torch.stack(poses_world2, dim=0)
    poses_world3 = torch.stack(poses_world3, dim=0)
    poses_world4 = torch.stack(poses_world4, dim=0)
    poses_world5 = torch.stack(poses_world5, dim=0)
    poses_world6 = torch.stack(poses_world6, dim=0)
    poses_world7 = torch.stack(poses_world7, dim=0)
    poses_world8 = torch.stack(poses_world8, dim=0)
    
    
    return poses_world1, poses_world2, poses_world3, poses_world4,\
           poses_world5, poses_world6, poses_world7, poses_world8


def world_to_camera_batch (points_batch, rotation_batch, translation_batch):
    batch_size = points_batch.shape[0]
    
    poses_camera1, poses_camera2, poses_camera3, poses_camera4 = ( [] for i in range (4))
    poses_camera5, poses_camera6, poses_camera7, poses_camera8 = ( [] for i in range (4))
    
    for i in range(batch_size):
        poses_camera1.append (world_to_camera_torch(points_batch[i], rotation_batch[i][0],  translation_batch[i][0]))
        poses_camera2.append (world_to_camera_torch(points_batch[i], rotation_batch[i][1],  translation_batch[i][1]))
        poses_camera3.append (world_to_camera_torch(points_batch[i], rotation_batch[i][2],  translation_batch[i][2]))
        poses_camera4.append (world_to_camera_torch(points_batch[i], rotation_batch[i][3],  translation_batch[i][3]))
        poses_camera5.append (world_to_camera_torch(points_batch[i], rotation_batch[i][4],  translation_batch[i][4]))
        poses_camera6.append (world_to_camera_torch(points_batch[i], rotation_batch[i][5],  translation_batch[i][5]))
        poses_camera7.append (world_to_camera_torch(points_batch[i], rotation_batch[i][6],  translation_batch[i][6]))
        poses_camera8.append (world_to_camera_torch(points_batch[i], rotation_batch[i][7],  translation_batch[i][7]))
        
    poses_camera1 = torch.stack(poses_camera1, dim=0)
    poses_camera2 = torch.stack(poses_camera2, dim=0)
    poses_camera3 = torch.stack(poses_camera3, dim=0)
    poses_camera4 = torch.stack(poses_camera4, dim=0)
    poses_camera5 = torch.stack(poses_camera5, dim=0)
    poses_camera6 = torch.stack(poses_camera6, dim=0)
    poses_camera7 = torch.stack(poses_camera7, dim=0)
    poses_camera8 = torch.stack(poses_camera8, dim=0)
      
    return poses_camera1, poses_camera2, poses_camera3, poses_camera4,\
           poses_camera5, poses_camera6, poses_camera7, poses_camera8


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

 
