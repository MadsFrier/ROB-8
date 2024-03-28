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
     

    print(numbers)
    # Convert the list of strings to a string with numbers separated by commas
    #numbers_string = ', '.join(numbers)



    return numbers


def load_npy(npy_filepath):
    with open(npy_filepath, 'rb') as f:
        npy = np.load(f)
    return npy

def create_point_clouds(file_path, start_count, finish_count):
    pcd_list = []
    camera_x_mm = 0.320
    camera_z_mm = 1.000
    camera_translation = np.array([[camera_x_mm],
                                       [0],
                                       [camera_z_mm],
                                       [1]])
    
    for i in range(start_count, finish_count):
            
        depth_npy = load_npy(file_path+'depth/rs_'+ str(i) + '.npy')

        pose = extract_numbers_from_file(file_path+'/pose/robot_pose_'+str((i+1))+'.txt')

        
        rot_euler = quaternion_rotation_matrix(pose[3:])

        #print(pose[3:])

        translation_vector_base = np.array([[pose[0]],
                                       [pose[1]],
                                       [pose[2]],
                                       [1]])
        
        #print(translation_vector_base)
        #print(rot_euler)
        base_transform = np.r_[rot_euler, np.array([[0, 0, 0]]) ]

        base_transform= np.c_[base_transform, translation_vector_base]

        #print(f'Base Transform:')
        #print(base_transform)

        camera_transform = np.array([ [1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, -1],
                                      [0, 0, 0]

        ])
        camera_transform = np.c_[ camera_transform, camera_translation ]  
        #print(camera_transform)

        base_camera = base_transform @ camera_transform

        print('Base Camera: ')
        print(base_camera)


                        
        depth = o3d.geometry.Image(depth_npy.astype(np.uint16))   
        color = cv.cvtColor(cv.imread(file_path+'rgb/rs_'+str(i)+'.png'), cv.COLOR_RGB2BGR)
        color = o3d.geometry.Image(color)
        # create rgbd
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False, depth_trunc=11.0)
        # segd = o3d.geometry.RGBDImage.create_from_color_and_depth(seg, depth, convert_rgb_to_intensity=False)
        
        # show rgb and depth images
        #show_rgbg_o3d(rgbd)

        # 4. Create point cloud for each image
        
        # create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # flip the orientation, so it looks upright, not upside-down

        pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        pcd.transform(base_camera)
        
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


def load_point_clouds(voxel_clouds_size=0.0):
    pcds = []
    #emo_icp_pcds = o3d.data.DemoICPPointClouds()
    for path in range(130, 141):
        pcd = o3d.io.read_point_cloud("/workspaces/ROB-8/docker/src/content/spatial_map_data/pcd/5LpN3gDmAk7_"+ str(path) + ".pcd")
        pcd.estimate_normals()
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds
start_im = 20
last_im =23
pcds_l = create_point_clouds('/home/christian/Github/ROB-8/docker/src/content/spatial_map_data/', start_im, last_im)
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
#pcds_down = load_point_clouds(voxel_size)
#pcds_down = o3d.io.read_point_cloud("/home/gayath/project/ROB-8/docker/src/content/demo_data/pcd/5LpN3gDmAk7_130.pcd")
#o3d.visualization.draw_geometries(pcds_down)

def pairwise_registration(source, target, max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
        #o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
        #o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            print('pcd: ', target_id)
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

print("Full registration ...")
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds_donw_l,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)


print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
    
print("Transform points and display")
for point_id in range(len(pcds_donw_l)):
    print(pose_graph.nodes[point_id].pose)
    pcds_l[point_id].transform(pose_graph.nodes[point_id].pose)
o3d.visualization.draw_geometries(pcds_donw_l)

pcds = load_point_clouds(voxel_size)
pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(pcds_donw_l)):
    pcds_l[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined += pcds[point_id]
pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
o3d.visualization.draw_geometries([pcd_combined_down])








