import open3d as o3d
import numpy as np
import cv2 as cv
import re


 
def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    return rot_matrix

def extract_numbers_from_file(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Use regular expression to find all numbers in the file
    # This pattern matches both integers and floating point numbers, including negatives
    numbers = re.findall(r"-?\d+\.?\d*", content)
    n_len = len(numbers)
    for i in range(0, len(numbers)):
        numbers[i]= float(numbers[i])
    w = numbers[6]
    x = numbers[3]
    y = numbers[4]
    z = numbers[5]
    numbers[3] = w
    numbers[4] = x
    numbers[5] = y
    numbers[6] = z
    
    return numbers


def load_npy(npy_filepath):
    with open(npy_filepath, 'rb') as f:
        npy = np.load(f)
    return npy

def create_point_clouds(file_path, start_count, finish_count):
   
    pcd_list = []
    camera_x_mm = 0.0
    camera_z_mm = 0.000
    camera_translation = np.array([[camera_x_mm],
                                       [0],
                                       [camera_z_mm],
                                       [1]])
  
    for i in range(start_count, finish_count):
            
        depth_npy = load_npy(file_path+'depth/rs_'+ str(i) + '.npy')


       
        pose = extract_numbers_from_file(file_path+'pose/robot_pose_'+str((i))+'.txt')
        pose_np = np.array(pose[3:])

        pose_rot = o3d.geometry.get_rotation_matrix_from_quaternion(pose_np)
   
        rot_euler = quaternion_rotation_matrix(pose[3:])

        translation_vector_base = np.array([[pose[0]],
                                       [pose[1]],
                                       [pose[2]],
                                       [1]])
        
        
        base_transform = np.r_[rot_euler, np.array([[0, 0, 0]]) ]

        base_transform= np.c_[base_transform, translation_vector_base]

        camera_transform = np.array([ [1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, -1],
                                      [0, 0, 0]

        ])
        camera_transform = np.c_[ camera_transform, camera_translation ]  
         
        base_camera = base_transform @ camera_transform

        print('Base Camera: ')
        print(base_camera)

    
       
        depth = o3d.geometry.Image(depth_npy.astype(np.uint16))   
        color = cv.cvtColor(cv.imread(file_path+'rgb/rs_'+str(i)+'.jpg'), cv.COLOR_RGB2BGR)
        
        color = o3d.geometry.Image(color)
        # create rgbd
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False, depth_trunc=11.0)
   

        
        # create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        pcd.transform(base_camera)
        #pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        
        pcd_list.append(pcd)
        print(len(pcd_list))
        # show point cloud
        #o3d.visualization.draw_geometries([pcd])
        file_pc = file_path +'/rgbd_pcd/'  + str(i) + '.pcd'
        #o3d.visualization.draw_geometries([pcd])
        # save point cloud
        #o3d.io.write_point_cloud(file_name=file_pc, point_cloud=pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)
    return pcd_list

    # 5. Combine point clouds with multiway registration to create coherent spatial map with landmarks


    return pcds
start_im = 5
last_im = 7
pcds_l = create_point_clouds('/workspaces/ROB-8/docker/src/content/new_data/', start_im, last_im)
print(pcds_l)
print(f'PCD List: {len(pcds_l)}')
o3d.visualization.draw_geometries(pcds_l)
pcds_donw_l = []
for i in range(last_im-start_im):
    print(i)
    pcds_l[i].estimate_normals()
    pcd_down = pcds_l[i].voxel_down_sample(voxel_size=0.02)
    pcds_donw_l.append(pcd_down)
print(f'PCD DOwn:{len(pcds_donw_l)}')
voxel_size = 0.02









