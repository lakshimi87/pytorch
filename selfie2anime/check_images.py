import cv2
import os

PathFormat = "dataset/%s/"

setName = input("set name: ")
path = PathFormat%setName
files = os.listdir(path)
files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]

images = []
for i in range(len(files)):
	image = cv2.imread(path+files[i])
	hist = cv2.calcHist([image], [0, 1, 2], None, [128, 128, 128], [0, 256, 0, 256, 0, 256])
	hist[127, 127, 127] = 0 #ignore all white pixels
	cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
	images.append(hist)

for i in range(len(files)-1):
	image1 = images[i]
	for j in range(i+1, len(files)):
		image2 = images[j]
		# Find the metric value
		metric_val = cv2.compareHist(image1, image2, cv2.HISTCMP_CORREL)
		if round(metric_val, 2) > 0.8: print(files[i], files[j], metric_val)
