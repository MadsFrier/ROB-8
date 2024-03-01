#!/usr/bin/env python3

from demo1 import *

from demo2 import *

if not os.path.isfile("/workspaces/docker/src/content/rgb.mp4"):
    print("Creating video")
    create_video("/workspaces/docker/src/content/5LpN3gDmAk7_1", "/workspaces/docker/src/content", fps=30)
else:
    print("File already exists, skipping video creation")

# show_videos(["/workspaces/docker_ubuntu/src/content/rgb.mp4", "/workspaces/docker_ubuntu/src/content/depth.mp4"])

# setup parameters
# @markdown meters per cell size
cs = 0.05 # @param {type: "number"}
# @markdown map resolution (gs x gs)
gs = 1000 # @param {type: "integer"}
# @markdown camera height (used for filtering out points on the floor)
camera_height = 1.5 # @param {type: "number"}
# @markdown depth pixels subsample rate
depth_sample_rate = 100 # @param {type: "integer"}
# @markdown data where rgb, depth, pose are loaded and map are saved
data_dir = "/workspaces/docker/src/content/5LpN3gDmAk7_1/" # @param {type: "string"}

create_lseg_map_batch(load_depth, data_dir, camera_height=camera_height, cs=cs, gs=gs, depth_sample_rate=depth_sample_rate)