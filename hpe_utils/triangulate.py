

import numpy as np


import torch

from cameras import Camera,load_cameras

from hpe_utils.utils_multiview import homogeneous_to_euclidean, euclidean_to_homogeneous


from hpe_utils.parser import parse_args


args = parse_args()


#%%


cam_file = args.camera_file#'/lhome/bouazia/hpe3d_open/cam/cameras.h5'#'/external/10g/tk/fs1/HPE3D_data/h36m/cameras.h5'

def generate_cam_extrinsics ():
    
    keys = list(load_cameras(cam_file, 
                             subjects=[1,5,6,7,8,9,11] ).keys ())
    Rotations, Translations = ([] for i in range(2))
    Rot_matrices, Trans_matrices = ({} for i in range(2))
    
    for sbj in ([1,5,6,7,8,9,11]): #1,5,6,7,8,9,11
        for cam in [1,2,3,4]: #1,2,3,4
            camera_params = load_cameras(cam_file, subjects=[sbj] )[(sbj,cam)]
            Cam_param = Camera (camera_params)
            Rotations.append (Cam_param.R)
            Translations.append (Cam_param.T)
    
    
    
    for (i,key) in enumerate (keys):
        #print (i,key)
        Rot_matrices[key] = Rotations[i][:]
        Trans_matrices[key] = Translations[i][:]
    return Rot_matrices, Trans_matrices




def generate_projection_matrices ():
    
    keys = list(load_cameras(cam_file, 
                             subjects=[1,5,6,7,8,9,11] ).keys ())
    projections = []
    projection_matrices = {}
    
    for sbj in ([1,5,6,7,8,9,11]): #1,5,6,7,8,9,11
        for cam in [1,2,3,4]: #1,2,3,4
            camera_params = load_cameras(cam_file, subjects=[sbj] )[(sbj,cam)]
            Cam_param = Camera (camera_params)
            projection_matrix = Cam_param.projection_matrix
            projections.append (projection_matrix)   
    
    for (i,key) in enumerate (keys):
        #print (i,key)
        projection_matrices[key] = projections[i][:]
    return projection_matrices



#%%
def projection_matrices_batch (batch,projection_matrices):
     subjects = batch ['subject'].reshape (-1,4)
     
     proj_matrix_batch = []
     for i in range (len(subjects)):
         item_sbj = int (subjects[i,0])
         proj_matrix = torch.tensor([projection_matrices[key] for key in \
                                    [(item_sbj,1),(item_sbj,2),(item_sbj,3),(item_sbj,4)]])
         proj_matrix_batch.append (proj_matrix)
    
     proj_matrix_batch = torch.stack (proj_matrix_batch)
     return proj_matrix_batch
 

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
def triangulate_point_from_multiple_views_linear(proj_matricies, points):
    """Triangulates one point from multiple (N) views using direct linear transformation (DLT).
    For more information look at "Multiple view geometry in computer vision",
    Richard Hartley and Andrew Zisserman, 12.2 (p. 312).
    Args:
        proj_matricies numpy array of shape (N, 3, 4): sequence of projection matricies (3x4)
        points numpy array of shape (N, 2): sequence of points' coordinates
    Returns:
        point_3d numpy array of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)
    A = np.zeros((2 * n_views, 4))
    for j in range(len(proj_matricies)):
        A[j * 2 + 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
        A[j * 2 + 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]

    u, s, vh =  np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]

    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d


    
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
    point_3d_batch = torch.zeros(batch_size, n_joints, 3, dtype=torch.float32, device=points_batch.device)

    for batch_i in range(batch_size):
        for joint_i in range(n_joints):
            points = points_batch[batch_i, :, joint_i, :]

            confidences = confidences_batch[batch_i, :, joint_i] if confidences_batch is not None else None
            point_3d = triangulate_point_from_multiple_views_linear_torch(proj_matricies_batch[batch_i], points, confidences=confidences)
            point_3d_batch[batch_i, joint_i] = point_3d

    return point_3d_batch


def triangulate_pose_torch (proj_matrix,poses_proj,n_joints=17):
    skel = []
    for i in range(n_joints):
        point = poses_proj [:,i,:]
        skel.append (triangulate_point_from_multiple_views_linear_torch (proj_matrix,point)[:,None])
    
    item_triang = torch.stack(skel, dim=0)
    item_triang.transpose_(0, 2)
    item_triang.transpose_(1, 2)

    return item_triang


def triangulate_pose_torch_batch (proj_matrix,poses):
    poses_triang = []
    for i in range(len(poses)):
       poses_triang.append(triangulate_pose_torch (proj_matrix,poses[i]))
    poses_triang = torch.stack(poses_triang, dim=0)
    poses_triang.transpose_(0, 1)
    return poses_triang [0]




     





























