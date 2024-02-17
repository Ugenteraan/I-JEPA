import os 
import glob 
import cv2 
from PIL import Image 



for x in glob.glob('./debug_dataset/**', recursive=True):

	if x.endswith(('png', 'jpg', 'jpeg', 'bmp')):

		try:
			image = cv2.imread(x)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = Image.fromarray(image)
		except Exception as err:
			print("Got problem with ", x)
			print(err)
			os.remove(x)
			print("Removed.")
			
