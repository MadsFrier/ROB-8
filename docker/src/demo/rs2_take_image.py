
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

stream = True

def load_npy(npy_filepath):
    with open(npy_filepath, 'rb') as f:
        npy = np.load(f)
    return npy

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)

try:
  for i in range(0, 50):
    frames = pipeline.wait_for_frames()

    print('image ', i, ' taken, ', 50-i, ' images left')

    # create align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    depth_image = np.array(aligned_frames.get_depth_frame().get_data())
    color_image = np.array(aligned_frames.get_color_frame().get_data())

    np.save('/home/gayath/project/ROB-8/docker/src/content/rs_data/depth/rs_' + str(i) + '.npy', depth_image)
    plt.imsave('/home/gayath/project/ROB-8/docker/src/content/rs_data/rgb/rs_' + str(i) + '.png', color_image)

    plt.figure()
    plt.imshow(color_image)
    plt.figure()
    plt.imshow(depth_image)
    #plt.figure()
    #plt.imshow(load_npy('/home/mads/github/ROB-8/docker/src/content/rs_data/depth/rs_' + str(iteration) + '.npy'))

    #plt.show()
    if not stream:
        print('images taken')
        pipeline.stop()
        break
    
    time.sleep(1)
    
finally:
    pipeline.stop()