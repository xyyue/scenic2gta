import numpy as np
import cv2
import os

# image converting part
root = "C:\Program Files (x86)\Steam\steamapps\common\Grand Theft Auto V\\kk"
strideWidth = int((1920 * 3 + 3) / 4) * 4
bins = [f.path for f in os.scandir(root) if f.path.endswith(".bin")]
for bin_file in bins:
    with open(bin_file, "rb") as fr:
        frame = fr.read()
        l = len(frame)
        if len(frame) == 6912000:
            buff = np.fromstring(frame, dtype='uint8')
            img = as_strided(buff, strides=(strideWidth, 3, 1), shape=(1200, 1920, 3))
            cv2.imwrite(bin_file.replace("bin", "png"), img)
