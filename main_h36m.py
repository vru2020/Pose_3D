import time
import logging
import datetime
from tempfile import mkdtemp
from pathlib import Path
import os
import sys

import torch
import torch.nn as nn

from datasets.h36m_multiview import H36M_Dataset

from hpe_utils.eval_utils import mpjpe,p_mpjpe

from model import Encoder2D,Decoder3D


from hpe_utils.utils import zero_the_root,add_the_root
from hpe_utils.utils_multiview import project_by_view_batch,rotate_3d_view
from hpe_utils.utils_multiview import world_to_camera_batch, camera_to_world_batch
from hpe_utils.utils_multiview import get_all_views_2d, get_all_views_3d_tri

from hpe_utils.triangulate import generate_projection_matrices, generate_cam_extrinsics
from hpe_utils.triangulate import translation_vectors_batch, rotation_matrices_batch, projection_matrices_batch
from hpe_utils.triangulate import triangulate_batch_of_points


from hpe_utils.parser import parse_args
from hpe_utils.utils import Timer


args = parse_args()

#%%

dataset_train = H36M_Dataset(args.train_subjects, args.annotation_file_train,
                   args.device, args.annotation_path, "train")


train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_memory, shuffle= args.shuffle )
  
  
dataset_test = H36M_Dataset(args.test_subjects, args.annotation_file_test,
                   args.device, args.annotation_path, "test")

test_loader = torch.utils.data.DataLoader(
        dataset= dataset_test,  batch_size=len(dataset_test), num_workers=args.num_workers,
        pin_memory=args.pin_memory,shuffle=args.shuffle )


#%%
runId = datetime.datetime.now().isoformat()
experiment_dir = Path('./experiments/') # '../experiments/' 
experiment_dir.mkdir(parents=True, exist_ok=True)
runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
checkpoints_path = os.path.join(runPath,'checkpoints/')
os.mkdir(checkpoints_path)
#sys.stdout = Logger('{}/run.log'.format(runPath))
sys.stdout = open('{}/run.log'.format(runPath), 'w')

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

print('Expt:', runPath)
print('RunID:', runId)
print (args)

#%%
def train_model (epoch, encoder_2d, decoder_3d):
   
    projection_matrices = generate_projection_matrices ()
    Rotations = generate_cam_extrinsics() [0]
    Translations = generate_cam_extrinsics() [1]
     
    encoder_2d.train()
    decoder_3d.train()  
   
    criterion  = nn.L1Loss()
 
    train_loss = 0  
   
    for batch_idx, batch in enumerate(train_loader):
       
        pose2d_view1 = get_all_views_2d (batch)[0].to(args.device)
        pose2d_view2 = get_all_views_2d (batch)[1].to(args.device)
        pose2d_view3 = get_all_views_2d (batch)[2].to(args.device)
        pose2d_view4 = get_all_views_2d (batch)[3].to(args.device)
        
             
        pose3d_view1 = get_all_views_3d_tri (batch)[0].to(args.device)
        pose3d_view2 = get_all_views_3d_tri (batch)[1].to(args.device)
        pose3d_view3 = get_all_views_3d_tri (batch)[2].to(args.device)
        pose3d_view4 = get_all_views_3d_tri (batch)[3].to(args.device)
            
            
        pose3d_view1_root = pose3d_view1 [:,0,:].unsqueeze(1).float().to(args.device)
        pose3d_view2_root = pose3d_view2 [:,0,:].unsqueeze(1).float().to(args.device)
        pose3d_view3_root = pose3d_view3 [:,0,:].unsqueeze(1).float().to(args.device)
        pose3d_view4_root = pose3d_view4 [:,0,:].unsqueeze(1).float().to(args.device)
       
           
        out_enc_view1 = encoder_2d(zero_the_root (pose2d_view1 ,0))
        out_enc_view2 = encoder_2d(zero_the_root (pose2d_view2 ,0))
        out_enc_view3 = encoder_2d(zero_the_root (pose2d_view3 ,0))
        out_enc_view4 = encoder_2d(zero_the_root (pose2d_view4 ,0))
       
           
        pose3d_est_view1 = decoder_3d(out_enc_view1).reshape (-1,args.num_joints,3)
        pose3d_est_view2 = decoder_3d(out_enc_view2).reshape (-1,args.num_joints,3)
        pose3d_est_view3 = decoder_3d(out_enc_view3).reshape (-1,args.num_joints,3)
        pose3d_est_view4 = decoder_3d(out_enc_view4).reshape (-1,args.num_joints,3)
       
       
        pose3d_est_view1 = add_the_root(pose3d_est_view1, pose3d_view1_root,0)
        pose3d_est_view2 = add_the_root(pose3d_est_view2, pose3d_view2_root,0)
        pose3d_est_view3 = add_the_root(pose3d_est_view3, pose3d_view3_root,0)
        pose3d_est_view4 = add_the_root(pose3d_est_view4, pose3d_view4_root,0)
        
        pose3d_est_view1 = pose3d_est_view1.to (args.device)
        pose3d_est_view2 = pose3d_est_view2.to (args.device)
        pose3d_est_view3 = pose3d_est_view3.to (args.device)
        pose3d_est_view4 = pose3d_est_view4.to (args.device)
        
        
        ####################### compute the input triangulation loss ###################
         
        if args.tri_inp:
            
            loss3d_tri = criterion (pose3d_est_view1, pose3d_view1) \
                       + criterion (pose3d_est_view2, pose3d_view2) \
                       + criterion (pose3d_est_view3, pose3d_view3) \
                       + criterion (pose3d_est_view4, pose3d_view4)
               
        
        proj_matrices_batch = projection_matrices_batch (batch, projection_matrices).to (args.device)       
        rot_matrices_batch  = rotation_matrices_batch (batch, Rotations).to (args.device)
        trans_vectors_batch  =  translation_vectors_batch (batch, Translations ).to (args.device)
        
        ####################### compute the 3D consistency loss ###################
       
        if args.transform_3d:
            
            pose3d_est_view1_2 = rotate_3d_view (pose3d_est_view1,rot_matrices_batch, trans_vectors_batch,0,1)
            pose3d_est_view1_3 = rotate_3d_view (pose3d_est_view1,rot_matrices_batch, trans_vectors_batch,0,2)
            pose3d_est_view1_4 = rotate_3d_view (pose3d_est_view1,rot_matrices_batch, trans_vectors_batch,0,3)
           
            pose3d_est_view2_1 = rotate_3d_view (pose3d_est_view2,rot_matrices_batch, trans_vectors_batch,1,0)
            pose3d_est_view2_3 = rotate_3d_view (pose3d_est_view2,rot_matrices_batch, trans_vectors_batch,1,2)
            pose3d_est_view2_4 = rotate_3d_view (pose3d_est_view2,rot_matrices_batch, trans_vectors_batch,1,3)
           
            pose3d_est_view3_1 = rotate_3d_view (pose3d_est_view3,rot_matrices_batch, trans_vectors_batch,2,0)
            pose3d_est_view3_2 = rotate_3d_view (pose3d_est_view3,rot_matrices_batch, trans_vectors_batch,2,1)
            pose3d_est_view3_4 = rotate_3d_view (pose3d_est_view3,rot_matrices_batch, trans_vectors_batch,2,3)
           
            pose3d_est_view4_1 = rotate_3d_view (pose3d_est_view4,rot_matrices_batch, trans_vectors_batch,3,0)
            pose3d_est_view4_2 = rotate_3d_view (pose3d_est_view4,rot_matrices_batch, trans_vectors_batch,3,1)
            pose3d_est_view4_3 = rotate_3d_view (pose3d_est_view4,rot_matrices_batch, trans_vectors_batch,3,2)
            
        
            #zero center the 3d joints around the pelvis joint
        
            loss3d_view1 = criterion (zero_the_root(pose3d_est_view2_1,0), zero_the_root(pose3d_view1,0)) + \
                           criterion (zero_the_root(pose3d_est_view3_1,0), zero_the_root(pose3d_view1,0)) + \
                           criterion (zero_the_root(pose3d_est_view4_1,0), zero_the_root(pose3d_view1,0))
                           
            loss3d_view2 = criterion (zero_the_root(pose3d_est_view1_2,0), zero_the_root(pose3d_view2,0)) + \
                           criterion (zero_the_root(pose3d_est_view3_2,0), zero_the_root(pose3d_view2,0)) + \
                           criterion (zero_the_root(pose3d_est_view4_2,0), zero_the_root(pose3d_view2,0))
                                   
            loss3d_view3 = criterion (zero_the_root(pose3d_est_view1_3,0), zero_the_root(pose3d_view3,0)) + \
                           criterion (zero_the_root(pose3d_est_view2_3,0), zero_the_root(pose3d_view3,0)) + \
                           criterion (zero_the_root(pose3d_est_view4_3,0), zero_the_root(pose3d_view3,0))
                           
            loss3d_view4 = criterion (zero_the_root(pose3d_est_view1_4,0), zero_the_root(pose3d_view4,0)) + \
                           criterion (zero_the_root(pose3d_est_view2_4,0), zero_the_root(pose3d_view4,0)) + \
                           criterion (zero_the_root(pose3d_est_view3_4,0), zero_the_root(pose3d_view4,0))
                           
            loss3d_rigid = loss3d_view1 + loss3d_view2 + loss3d_view3 + loss3d_view4
         
       
        ################### compute the 2D re-projection loss #############
        if args.proj_2d :
            
            #Project the estimated 3d to global frame
            pose3d_est_world_view1 = camera_to_world_batch (pose3d_est_view1,rot_matrices_batch, trans_vectors_batch) [0]
            pose3d_est_world_view2 = camera_to_world_batch (pose3d_est_view2,rot_matrices_batch, trans_vectors_batch) [1]
            pose3d_est_world_view3 = camera_to_world_batch (pose3d_est_view3,rot_matrices_batch, trans_vectors_batch) [2]
            pose3d_est_world_view4 = camera_to_world_batch (pose3d_est_view4,rot_matrices_batch, trans_vectors_batch) [3]
       
           
            pose3d_est_11,pose3d_est_12,pose3d_est_13,pose3d_est_14 = project_by_view_batch (pose3d_est_world_view1,proj_matrices_batch)
            pose3d_est_21,pose3d_est_22,pose3d_est_23,pose3d_est_24 = project_by_view_batch (pose3d_est_world_view2,proj_matrices_batch)
            pose3d_est_31,pose3d_est_32,pose3d_est_33,pose3d_est_34 = project_by_view_batch (pose3d_est_world_view3,proj_matrices_batch)
            pose3d_est_41,pose3d_est_42,pose3d_est_43,pose3d_est_44 = project_by_view_batch (pose3d_est_world_view4,proj_matrices_batch)
            
            
            
            loss2d_view1 = criterion(zero_the_root (pose3d_est_11,0) ,zero_the_root (pose2d_view1,0) ) \
                         + criterion(zero_the_root (pose3d_est_21,0), zero_the_root (pose2d_view1,0) ) \
                         + criterion(zero_the_root (pose3d_est_31,0), zero_the_root (pose2d_view1,0) ) \
                         + criterion(zero_the_root (pose3d_est_41,0), zero_the_root (pose2d_view1,0) )
           
            loss2d_view2 = criterion(zero_the_root (pose3d_est_12,0), zero_the_root (pose2d_view2,0) ) \
                         + criterion(zero_the_root (pose3d_est_22,0), zero_the_root (pose2d_view2,0) )   \
                         + criterion(zero_the_root (pose3d_est_32,0), zero_the_root (pose2d_view2,0) )\
                         + criterion(zero_the_root (pose3d_est_42,0), zero_the_root (pose2d_view2,0) )
                       
            loss2d_view3 = criterion(zero_the_root (pose3d_est_13,0), zero_the_root (pose2d_view3,0) ) \
                         + criterion(zero_the_root (pose3d_est_23,0), zero_the_root (pose2d_view3,0) ) \
                         + criterion(zero_the_root (pose3d_est_33,0), zero_the_root (pose2d_view3,0) ) \
                         + criterion(zero_the_root (pose3d_est_43,0), zero_the_root (pose2d_view3,0) )
                         
                           
            loss2d_view4 = criterion(pose3d_est_14, pose2d_view4) + criterion(pose3d_est_24, pose2d_view4) \
                         + criterion(pose3d_est_34, pose2d_view4) + criterion(pose3d_est_44, pose2d_view4)
           
            loss2d = loss2d_view1 + loss2d_view2 + loss2d_view3 + loss2d_view4
            
              
       ################### compute the output triangulation loss #############
        if args.tri_out:
            
            poses2d_proj = torch.stack ([pose2d_view1, pose2d_view2, pose2d_view3, pose2d_view4]).transpose (0,1).to(args.device)
            pose3d_tri_proj = triangulate_batch_of_points (proj_matrices_batch,poses2d_proj)
           
            pose3d_view1_proj = world_to_camera_batch (pose3d_tri_proj,rot_matrices_batch,trans_vectors_batch)[0]
            pose3d_view2_proj = world_to_camera_batch (pose3d_tri_proj,rot_matrices_batch,trans_vectors_batch)[1]
            pose3d_view3_proj = world_to_camera_batch (pose3d_tri_proj,rot_matrices_batch,trans_vectors_batch)[2]
            pose3d_view4_proj = world_to_camera_batch (pose3d_tri_proj,rot_matrices_batch,trans_vectors_batch)[3]
            
            if epoch > args.epoch_out:
                
                loss3d_tri_proj = criterion (pose3d_est_view1, pose3d_view1_proj) \
                                + criterion (pose3d_est_view2, pose3d_view2_proj) \
                                + criterion (pose3d_est_view3, pose3d_view3_proj) \
                                + criterion (pose3d_est_view4, pose3d_view4_proj)

       
        ###################### compute loss ###################################
        if args.tri_inp:
            loss_all   =  loss3d_tri
        if args.transform_3d:
            loss_all   =  loss3d_rigid
        if args.proj_2d :
            loss_all = loss2d
        if args.proj_2d and args.tri_inp:
            loss_all   =  loss3d_tri + loss2d
        if args.tri_inp and args.transform_3d:
            loss_all   =   loss3d_tri + loss3d_rigid
        if args.proj_2d and args.tri_inp and args.transform_3d:
            loss_all   =  loss3d_rigid + loss3d_tri + loss2d
        if args.proj_2d and args.tri_inp and args.tri_out:
            loss_all   =  loss3d_tri + loss2d+ loss3d_tri_proj
       
        params = list(encoder_2d.parameters())+list(decoder_3d.parameters())          
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)        
        optimizer.zero_grad()  
        loss_all.backward ()      
        train_loss += loss_all.item()        
        optimizer.step()
             
    train_loss /=  len(train_loader.dataset)
       
    return train_loss, optimizer


#%%
def test_model (epoch, encoder_2d, decoder_3d):
    encoder_2d.eval()
    decoder_3d.eval()  
   
   
    with torch.no_grad():
       
        batch = next(iter(test_loader))
       
        pose2d = batch ["pose2d"].to(args.device)
        pose3d = batch ["pose3d"].to(args.device)
       
                   
        pose2d = zero_the_root(pose2d, 0)
        pose3d = zero_the_root(pose3d, 0)
         
        out_enc_2d = encoder_2d(pose2d)

        pose3d_est = decoder_3d( out_enc_2d).reshape (-1,args.num_joints,3)

    
        pose3d_est = torch.cat((torch.zeros(pose3d_est.shape[0], 1, 3, device=args.device), pose3d_est), dim=1)
        pose3d     = torch.cat((torch.zeros(pose3d.shape[0], 1, 3, device=args.device), pose3d), dim=1)
   
       
        mpjpe_all   = mpjpe(pose3d_est, pose3d)
        p_mpjpe_all = p_mpjpe(pose3d_est.cpu().detach().numpy(), pose3d.cpu().detach().numpy())
   
    return mpjpe_all, p_mpjpe_all


#%%

if __name__ == '__main__':
    with Timer('HPE-Self-Supervised') as t:
       
        encoder_2d = Encoder2D(args.num_latents,args.num_joints).to(args.device)
        decoder_3d = Decoder3D(args.num_latents,args.num_joints).to(args.device)
       
        for epoch in range(args.num_epochs):
       
            since = int(round(time.time()))
            train_loss, optimizer = train_model (epoch, encoder_2d, decoder_3d)        
            MPJPE_test, MPJPE_align_test, = test_model (epoch, encoder_2d, decoder_3d)
           
            print('====> Epoch: {} ETA: {}s training loss: {:.4f}  mpjpe_test: {:.2f} mpjpe_align_test: {:.2f}'.format (
            epoch, int(round(time.time())) - since , train_loss,MPJPE_test,MPJPE_align_test))
           
           
            torch.save({
                    'epoch': epoch ,
                    'encoder_2d_state_dict': encoder_2d.state_dict(),
                    'decoder_3d_state_dict': decoder_3d.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'mpjpe': MPJPE_test,
                    }, os.path.join(checkpoints_path,'model_{}.pt'.format (epoch)))
               










