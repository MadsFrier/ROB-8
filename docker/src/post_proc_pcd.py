import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import json

def remove_sparse_points(pcd_points, pcd_colours, min_neighbor_points, radius):
    print("Removing sparse points...")
    mask = np.zeros(len(pcd_points), dtype=bool)
    for i, point in enumerate(pcd_points):
        #print('Checking point',i, 'out of',len(pcd_points))
        num_neighbors = 0
        dists = np.linalg.norm(pcd_points - point, axis=1)
        dists[i] = np.inf
        num_neighbors = np.sum(dists <=radius)
        mask[i] = num_neighbors >= min_neighbor_points

    flt_pcd_points = pcd_points[mask]
    flt_pcd_colours = pcd_colours[mask]
    return flt_pcd_points, flt_pcd_colours
def cluster_and_find_centre(pcd_points):
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(pcd_points)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    landmarks = []
    
    for i in unique_labels:
        indices = np.where(labels == i)[0]
        clustered_pcd_points = pcd_points[indices]
        landmark = np.median(clustered_pcd_points, axis=0)
        landmarks.append(list(landmark))
    
    return landmarks

def find_closest_colour(pcd_colours, colours):
    print('Fitting colours to predefined colours...')

    for i, colour in enumerate(pcd_colours):
        #print('Fitting color', i, 'out of', len(pcd_colours))
        dists = np.linalg.norm(colours - colour, axis=1)
        closest_index = np.argmin(dists)
        pcd_colours[i] = colours[closest_index]
        
    return pcd_colours
    
prompt = 'other, floor, ceiling, window, table, chair, plant, package, wall, door, trash_can, curtain, ceiling_lamp, cabinet, painting'
prompt = prompt.replace(" ", "")
labels = prompt.split(',')
# colours associated with prompt
colours = [
[0,     0,      0],         # Black             : 1
[255,   0,      0],         # Red               : 2  
[0,     255,    0],         # Lime              : 3
[0,     0,      255],       # Blue              : 4
[255,   255,    0],         # Yellow            : 5
[0,     255,    255],       # Cyan              : 6
[255,   0,      255],       # Magenta           : 7
[192,   192,    192],       # Silver            : 8
[128,   128,    128],       # Gray              : 9
[128,   0,      0],         # Maroon            : 10
[128,   128,    0],         # Olive             : 11
[0,     128,    0],         # Green             : 12
[128,   0,      128],       # Purple            : 13
[0,     128,    128],       # Teal              : 14
[0,     0,      128],       # Navy              : 15
[255,   165,    0],         # Orange            : 16
[255,   215,    0],         # Gold              : 17
[255,   255,    255],       # White             : 18
[255,   105,    180],       # Hot Pink          : 19
[75,    0,      130],       # Indigo            : 20
[255,   192,    203],       # Pink              : 21
[0,     255,    127],       # Spring Green      : 22
[0,     206,    209],       # Dark Turquoise    : 23
[148,   0,      211],       # Dark Violet       : 24
[244,   164,    96]         # Sandy Brown       : 25
]

pcd_path = "/workspaces/ROB-8/docker/src/content/meeti_db/seg_pcd/seg_map.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)

pcd_points, pcd_colours = np.asarray(pcd.points), (np.asarray(pcd.colors) * 255).astype(np.uint8)

# fit pcd_colours to align with colours
all_indices = [i for i in range(len(colours))]

chosen_indices = [5, 6, 7] # chair, plant, package

del_indices = [x for x in all_indices if x not in chosen_indices]
del_colours = []
for del_index in del_indices:
    del_colours.append(colours[del_index])

fitted_pcd_colours = find_closest_colour(pcd_colours, np.array(colours))

# remove unwanted classes
proc_pcd_points = pcd_points.copy()
proc_pcd_colours = fitted_pcd_colours.copy()
for chosen_colour in del_colours:
    indices = np.where(np.all(proc_pcd_colours == chosen_colour, axis=1))[0]

    proc_pcd_points = np.delete(proc_pcd_points, indices, axis=0)
    proc_pcd_colours = np.delete(proc_pcd_colours, indices, axis=0)

# remove sparse points and cluster
chosen_colours = []
for chosen_index in chosen_indices:
    chosen_colours.append(colours[chosen_index])

flt_pcd_points = np.empty([1,3])
flt_pcd_colours = np.empty([1,3])

landmarks_dict = {}
landmarks = []

for i, chosen_colour in enumerate(chosen_colours):
    chosen_prompt = labels[chosen_indices[i]]

    indices = np.where(np.all(proc_pcd_colours == chosen_colour, axis=1))[0]
    
    flt_points, flt_colours = remove_sparse_points(proc_pcd_points[indices], proc_pcd_colours[indices], 50, 0.25) # [m]
    # cluster
    res = cluster_and_find_centre(flt_points)
    landmarks.append(res)
    
    for y, x in enumerate(res):
        landmarks_dict[chosen_prompt +'_'+ str(y)] = x
    
    flt_pcd_points = np.vstack((flt_pcd_points, flt_points))
    flt_pcd_colours = np.vstack((flt_pcd_colours, flt_colours))

#for lm in landmarks:
#    for i in lm:
#        i = [i[0], i[1], i[2]]
#        flt_pcd_points = np.vstack((flt_pcd_points, i))
#        flt_pcd_colours = np.vstack((flt_pcd_colours, [0, 0, 0]))

flt_pcd_points = flt_pcd_points[1:]
flt_pcd_colours = flt_pcd_colours[1:]

# convert np array to pcd 
flt_pcd = o3d.geometry.PointCloud()

flt_pcd.points = o3d.utility.Vector3dVector(flt_pcd_points)
flt_pcd.colors = o3d.utility.Vector3dVector(flt_pcd_colours/255.0)

# save landmarks as json
with open('/workspaces/ROB-8/docker/src/content/meeti_db/landmarks/lms.json', 'w') as file:
    json.dump(landmarks_dict, file)

#o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([flt_pcd])

#save pcd
o3d.io.write_point_cloud('/workspaces/ROB-8/docker/src/content/meeti_db/seg_pcd/seg_post_map.pcd', flt_pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)