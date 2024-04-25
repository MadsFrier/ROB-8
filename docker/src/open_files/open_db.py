import sqlite3 as sql
import cv2
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

conn = sql.connect("/workspaces/ROB-8/docker/src/content/group_data/test.db")

black_image = np.zeros((720, 1280, 3), dtype=np.uint8)

with open("/workspaces/ROB-8/docker/src/content/group_data/black_image.jpg", "rb") as image:
  f = image.read()
  b = bytearray(f)

#conn.execute(f"UPDATE Data SET image = ? WHERE id = ?", (b, 3))
#conn.commit()
data = conn.execute("SELECT image FROM Data")
images = data.fetchall()

for image in images:
        
    image_blob = image[0]
    print("----------------")
    print(type(image_blob))
    print(image_blob[0:10])
    print(len(image_blob))
            
    # Create a PIL Image object from the image blob
    image = np.array(Image.open(io.BytesIO(image_blob)))
    
    plt.imshow(image)
    plt.show()
    
# replace all image with black images

conn.close()