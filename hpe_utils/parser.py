import argparse




def parse_args():
    
    
    parser = argparse.ArgumentParser(description='Training script')

    
    parser.add_argument('--train_subjects',type=list,default=[1, 5, 6, 7, 8],help='training subjects for H36M')
    parser.add_argument('--test_subjects',type=list,default=[9,11],help='test subjects for H36M')
    
    parser.add_argument('--annotation_path',type=str,default='/lhome/bouazia/hpe3d_open/processed_data',help="annotation path")
    parser.add_argument('--annotation_file_train',type=str,default='h36m17_cpn_ft_h36m_dbb_train_3dptri',help="annotation file train")
    parser.add_argument('--annotation_file_test',type=str,default='h36m17_cpn_ft_h36m_dbb_test',help="annotation file test")
    parser.add_argument('--camera_file',type=str,default='/lhome/bouazia/hpe3d_open/processed_data/cameras.h5',help="camera file")
    
    
    parser.add_argument('--batch_size',type=int,default=8192,help='batch size for the training ')
    parser.add_argument('--num_workers',type=int,default=4,help='num_workers for the training ')
    parser.add_argument('--pin_memory',type=bool,default=False,help='num_workers for the training ')
    parser.add_argument('--shuffle',type=bool,default=False,help='shuffle data during the training ')
    parser.add_argument('--device',type=str,default='cuda:0',help='training device ')
    
    parser.add_argument('--num_latents',type=int,default=1024,help='number of latents for the backbone ')
    parser.add_argument('--num_joints',type=int,default=16,help='number of joints')
    
    parser.add_argument('--num_epochs',type=int,default=700,help='number of training epochs')
    parser.add_argument('--learning_rate',type=float,default=1e-3,help='learning rate for the optimizer')
    
    
    parser.add_argument('--tri_inp',type=bool,default=True,help='use the trinagulated keypoints for the input triangulation loss')
    parser.add_argument('--transform_3d',type=bool,default=True,help='rotate the 3D keypoints for the consistency loss')
    parser.add_argument('--proj_2d',type=bool,default=False,help='proj_2d the 3D keypoints for the re-projection loss')
    parser.add_argument('--tri_out',type=bool,default=False,help='triangulate the projected keypoints for the output triangulation loss')

    parser.add_argument('--epoch_out',type=int,default=450,help='number of training epochs')
    
    
    
    args = parser.parse_args()
    
    return args





