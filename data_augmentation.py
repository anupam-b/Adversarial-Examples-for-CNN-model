import cv2
import random
import os

from data_augmentation_util.flip import *
from data_augmentation_util.brightness import *
from data_augmentation_util.rotate_image import *
from data_augmentation_util.projection_transform import *
from data_augmentation_util.noise import *

def add_data(root, files):
	a = root.replace("gtsrb-german-traffic-sign/Train", '')
	os.makedirs("train_augmented"+"/"+str(a[1:]))
	print("Adding samples to class "+ str(a[1:]))
	n = len(files)
	add_size = 2500-n
	l=[]
	for i in range(len(files)):
		if(files[i][:2]!="GT"):
			img = cv2.imread(root+"/"+files[i])
			l.append(img)

	l2=[]
	while(len(l2)<=add_size):
		rnd_img = l[random.randint(0,len(l)-1)]
		rnd_num = random.randint(1,5)
		img = None
		if(rnd_num == 1):
			img = brightness(rnd_img)
		elif(rnd_num == 2):
			img = rotate_image(rnd_img)
		elif(rnd_num == 3):
			img = projection_transform(rnd_img)
		elif(rnd_num == 4):
			img = noise(rnd_img)
		else:
			img = flip(rnd_img, int(a[1:]))
		l2.append(rnd_img)
	l_tot = l + l2
	for j in range(len(l_tot)):
		cv2.imwrite("train_augmented"+"/"+str(a[1:])+"/"+str(j)+".png", l_tot[j])

print("Starting augmentation\n")

m=0
for root, dirs, files in os.walk("gtsrb-german-traffic-sign/Train"):
	if(root != "gtsrb-german-traffic-sign/Train"):
		add_data(root, files)

print("Completed")


'''
# TEST DATA AUGMENTATION
im = cv2.imread("aug_examples/01_speed_limit_30.jpg")
img = flip(im, 1)
cv2.imwrite("aug_examples/1.png", img)

img = brightness(im, 0.5)
cv2.imwrite("aug_examples/2.png", img)

img = rotate_image(im, 0.5)
cv2.imwrite("aug_examples/3.png", img*255)

img = projection_transform(im, 0.5)
cv2.imwrite("aug_examples/4.png", img*255)

img = noise(im)
cv2.imwrite("aug_examples/5.png", img*255)
'''