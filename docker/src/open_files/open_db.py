import sqlite3 as sql
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

conn = sql.connect("/workspaces/ROB-8/docker/src/content/meeti_db/dbs/meeti.db")

data = conn.execute("SELECT image FROM Data")
images = data.fetchall()

for image in images:
        
    image_blob = image[0]
            
    # Create a PIL Image object from the image blob
    image = np.array(Image.open(io.BytesIO(image_blob)))
    plt.imshow(image)
    plt.show()
    
# replace all image with black images

conn.close()