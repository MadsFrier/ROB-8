
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import json

stream = False

def load_npy(npy_filepath):
    with open(npy_filepath, 'rb') as f:
        npy = np.load(f)
    return npy

# Create a pipeline
pipeline = rs.pipeline()

with open("/home/gayath/project/ROB-8/docker/settings of d455.json", "r") as jsonFile: 
    jsonObj = json.load(jsonFile)
#changing parameters in the json file
# add
    jsonObj["viewer"]["stream-height"] = 840
    jsonObj["viewer"]["stream-width"] = 480
    jsonObj["parameters"]["controls-autoexposure-manual"] = 33000
    jsonObj["parameters"]["controls-depth-gain"] = 16
    jsonObj["parameters"]["controls-laserpower"] = 300

    newData = json.dumps(jsonObj, indent=4)

# open
with open("/home/gayath/project/ROB-8/docker/modified settings of d455.json", "w") as jsonFile:

# write
    jsonFile.write(newData)
    json_string= newData
    print("W: ",json_string)
    #jsonFile.close()


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
dev = profile.get_device()
advnc_mode = rs.rs400_advanced_mode(dev)
advnc_mode.load_json(json_string)

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
    count_im = 72

    np.save('/home/christian/Github/ROB-8/docker/src/content/spatial_map_data/depth/rs_' + str(count_im) + '.npy', depth_image)
    plt.imsave('/home/christian/Github/ROB-8/docker/src/content/spatial_map_data/rgb/rs_' + str(count_im) + '.jpg', color_image)

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