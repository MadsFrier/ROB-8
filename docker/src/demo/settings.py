import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time

#jsonObj = json.load(open("/home/gayath/changed settings of d455.json"))
#json_string= str(jsonObj).replace("'", '\"')\

with open("/home/gayath/project/ROB-8/docker/settings of d455.json", "r") as jsonFile:
    jsonObj = json.load(jsonFile)
    
    #json_string= jsonObj
    #print("W: ",json_string)
    
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



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

freq=30

print("Data frequency",freq)

print("Width: ", 840)
print("Height: ", 480)
print("FPS: ", 30)
config.enable_stream(rs.stream.depth, 840, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 840, 480, rs.format.bgr8, 30)
cfg = pipeline.start(config)
dev = cfg.get_device()
advnc_mode = rs.rs400_advanced_mode(dev)
advnc_mode.load_json(json_string)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        #Initialize colorizer class
        colorizer = rs.colorizer()
        # Convert images to numpy arrays, using colorizer to generate appropriate colors
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Stack both images horizontally
        images = np.hstack((color_image, depth_image))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()