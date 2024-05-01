import matplotlib.pyplot as plt
import numpy as np

# Directory of data
data_directory = "/workspaces/ROB-8/docker/src/content/" # This is for users inside of docker only

# choose database
dataset = 'meetwb_db/'
data_directory = data_directory + dataset

# choose prompt
prompt = 'other, floor, ceiling, window, table, chair, plant, package, wall, door, trash_can, curtain, ceiling_lamp, cabinet, painting'

# colors associated with prompt
colors = [
[0, 0, 0],       # Black    : 1
[255, 0, 0],     # Red      : 2  
[0, 255, 0],     # Lime     : 3
[0, 0, 255],     # Blue     : 4
[255, 255, 0],   # Yellow   : 5
[0, 255, 255],   # Cyan     : 6
[255, 0, 255],   # Magenta  : 7
[192, 192, 192], # Silver   : 8
[128, 128, 128], # Gray     : 9
[128, 0, 0],     # Maroon   : 10
[128, 128, 0],   # Olive    : 11
[0, 128, 0],     # Green    : 12
[128, 0, 128],   # Purple   : 13
[0, 128, 128],   # Teal     : 14
[0, 0, 128],     # Navy     : 15
[255, 165, 0],   # Orange   : 16
[255, 215, 0],   # Gold     : 17
[255, 255, 255], # White    : 18
[255, 105, 180], # Hot Pink : 19
[75, 0, 130],    # Indigo   : 20
[255, 192, 203], # Pink     : 21
[0, 255, 127],   # Spring Green
[0, 206, 209],   # Dark Turquoise
[148, 0, 211],   # Dark Violet
[244, 164, 96]   # Sandy Brown
]

# Convert colors to numpy array for matplotlib compatibility
colors = np.array(colors) / 255.0

# Creating an empty plot
fig, ax = plt.subplots()

# Creating legend without plotting anything
for color, label in zip(colors, prompt.split(', ')):
    ax.plot([], [], marker='o', markersize=10, color=color, label=label)

# Showing only the legend
ax.legend()

plt.savefig(data_directory + 'legend.png')