#imports

import os
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2
import sqlite3 as sql
from PIL import Image
import io
import shutil

from demo.lseg_demo import load_lseg, run_lseg, torch_clear_cache

# functions

def flatten(lst):
    return [x for xs in lst for x in xs]


if __name__ == "__main__":
    # Directory of data
    data_directory = "/workspaces/ROB-8/docker/src/content/" # This is for users inside of docker only
    
    # choose database
    dataset = 'meeti_db/'
    data_directory = data_directory + dataset
    
    rgb_pcd_folder = 'rgb_pcd/'
    seg_pcd_folder = 'seg_pcd/'
    rgb_folder = 'rgb/'
    seg_folder = 'seg/'
    db_folder = 'db/'


    # Choose database
    db_name = "meeti.db"
    
    #dublicate db to prepare for updates to the db
    shutil.copy(data_directory + db_name, data_directory + db_name[:-3] + '_seg.db')
    
    # choose lseg png file name
    file_name = "meeti_"
    
    # allow for lseg to run
    allow_lseg = True
    
    # Choose to create rgb or seg pcd or both
    pcds = [rgb_pcd_folder] # [rgb_pcd_folder, seg_pcd_folder]
    
    # choose prompt
    prompt = 'other, floor, ceiling, window, table, chair, plant, package, wall, door, trash_can, curtain, ceiling_lamp, cabinet, painting'
    
    # colors associated with prompt
    colors = [
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

    # check if there is enough colors for prompt
    if len(colors) < len(prompt.split(',')):
        print('Not enough colors for the amount of prompts, add more colors to continue')
        exit()
    
    # sql
    conn = sql.connect(data_directory + db_folder + db_name[:-3] + '_seg.db')

    data = conn.execute("SELECT image FROM Data")
    images = data.fetchall()

    for i, image in enumerate(images):
        
        print('Processing image', i+1, 'of', len(images))
        
        image_blob = image[0]
                
        # Create a PIL Image object from the image blob
        rgb_img = np.array(Image.open(io.BytesIO(image_blob)))
        
        if allow_lseg:
             
            plt.imsave(data_directory + rgb_folder + file_name + str(i) + ".jpg", rgb_img)
          
            # load lseg model and segment rgb image            
            model, labels = load_lseg(prompt)
            
            palette = flatten(colors[0:len(labels)])
            
            lseg_img, mask_img, patches = run_lseg(rgb_img, model, labels, palette, show=False)
            lseg_img = np.array(lseg_img, dtype=np.uint8)
            mask_img = np.array(mask_img)

            #clear cache to clear GPU memory for next iteration
            torch_clear_cache(print_cache=True)
            
            # resize segmented image to match rgb and convert to rgb to add color corresponding to labels
            lseg_img = cv2.resize(lseg_img, dsize=(rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            mask_img = cv2.resize(mask_img, dsize=(rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            lseg_img = cv2.cvtColor(lseg_img, cv2.COLOR_GRAY2RGB)
            
            # save segmentation
            plt.imsave(data_directory + seg_folder + file_name + str(i) + ".jpg", mask_img)
        
        with open(data_directory + seg_folder + file_name + str(i) + '.jpg', "rb") as image:
            f = image.read()
            b = bytearray(f)
        print('saving changes to db')     
        conn.execute(f"UPDATE Data SET image = ? WHERE id = ?", (b, i+1))
        conn.commit()
        
    conn.close()