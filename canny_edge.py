import tensorflow as tf
import numpy as np

MAX = 1	#Max value of a pixel
GAUS_KERNEL = 3
GAUS_SIGMA  = 1.2
def Gaussian_Filter(kernel_size=GAUS_KERNEL, sigma=GAUS_SIGMA): #Default: Filter_shape = [5,5]
# 	--> Reference: https://en.wikipedia.org/wiki/Canny_edge_detector#Gaussian_filter
	k = (kernel_size-1)//2 
	filter = []
	sigma_2 = sigma**2
	for i in range(kernel_size):
		filter_row = []
		for j in range(kernel_size):
			Hij = np.exp(-((i+1-(k+1))**2 + (j+1-(k+1))**2)/(2*sigma_2))/(2*np.pi*sigma_2)
			filter_row.append(Hij)
		filter.append(filter_row)
	
	return np.asarray(filter).reshape(kernel_size,kernel_size,1,1)

"""
 NOTE: 	All variables are initialized first for reducing proccessing time.
"""
gaussian_filter = tf.constant(Gaussian_Filter(), tf.float32) 				#STEP-1
h_filter = tf.reshape(tf.constant([[-1,0,1],[-2,0,2],[-1,0,1]], tf.float32), [3,3,1,1])	#STEP-2
v_filter = tf.reshape(tf.constant([[1,2,1],[0,0,0],[-1,-2,-1]], tf.float32), [3,3,1,1])	#STEP-2

np_filter_0 = np.zeros((3,3,1,2))
np_filter_0[1,0,0,0], np_filter_0[1,2,0,1] = 1,1 ### Left & Right
# print(np_filter_0)
filter_0 = tf.constant(np_filter_0, tf.float32)
np_filter_90 = np.zeros((3,3,1,2))
np_filter_90[0,1,0,0], np_filter_90[2,1,0,1] = 1,1 ### Top & Bottom
filter_90 = tf.constant(np_filter_90, tf.float32)
np_filter_45 = np.zeros((3,3,1,2))
np_filter_45[0,2,0,0], np_filter_45[2,0,0,1] = 1,1 ### Top-Right & Bottom-Left
filter_45 = tf.constant(np_filter_45, tf.float32)
np_filter_135 = np.zeros((3,3,1,2))
np_filter_135[0,0,0,0], np_filter_135[2,2,0,1] = 1,1 ### Top-Left & Bottom-Right
filter_135 = tf.constant(np_filter_135, tf.float32)
	
np_filter_sure = np.ones([3,3,1,1]); np_filter_sure[1,1,0,0] = 0
filter_sure = tf.constant(np_filter_sure, tf.float32)
border_paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])

def Border_Padding(x, pad_width):
	for _ in range(pad_width): x = tf.pad(x, border_paddings, 'SYMMETRIC')
	return x
def FourAngles(d):
	d0   = tf.cast(tf.greater_equal(d,157.5), tf.float32)+tf.cast(tf.less(d,22.5), tf.float32)
	d45  = tf.cast(tf.greater_equal(d,22.5), tf.float32)*tf.cast(tf.less(d,67.5), tf.float32)
	d90  = tf.cast(tf.greater_equal(d,67.5), tf.float32)*tf.cast(tf.less(d,112.5), tf.float32)
	d135 = tf.cast(tf.greater_equal(d,112.5), tf.float32)*tf.cast(tf.less(d,157.5), tf.float32)
	# return {'d0':d0, 'd45':d45, 'd90':d90, 'd135':d135}
	return (d0,d45,d90,d135)

"""
	NOTES: 
	- Input ('img_tensor'): shape = [batch_size, height, width, 1] (This version supports only 1 channel)
	- Output: a batch of images with pixels are either 1 (edge) or 0 (non-edge)
"""
def TF_Canny(img_tensor, minRate=0.10, maxRate=0.40, 
			 preserve_size=True, remove_high_val=False, return_raw_edges=False):
	""" STEP-0 (Preprocessing): 
		1. Scale the tensor values to the expected range ([0,1]) 
		2. If 'preserve_size': As TensorFlow will pad by 0s for padding='SAME',
							   it is better to pad by the same values of the borders.
							   (This is to avoid considering the borders as edges)
	"""
	img_tensor = (img_tensor/tf.reduce_max(img_tensor))*MAX
	if preserve_size: img_tensor = Border_Padding(img_tensor, (GAUS_KERNEL-1)//2)

	""" STEP-1: Noise reduction with Gaussian filter """
	x_gaussian = tf.nn.convolution(img_tensor, gaussian_filter, padding='VALID')
	### Below is a heuristic to remove the intensity gradient inside a cloud ###
	if remove_high_val: x_gaussian = tf.clip_by_value(x_gaussian, 0, MAX/2)
	
	
	""" STEP-2: Calculation of Horizontal and Vertical derivatives  with Sobel operator 
		--> Reference: https://en.wikipedia.org/wiki/Sobel_operator	
	"""
	if preserve_size: x_gaussian = Border_Padding(x_gaussian, 1)
	Gx = tf.nn.convolution(x_gaussian, h_filter, padding='VALID')
	Gy = tf.nn.convolution(x_gaussian, v_filter, padding='VALID')
	G 		= tf.sqrt(tf.square(Gx) + tf.square(Gy))
	BIG_PHI = tf.atan2(Gy,Gx)
	BIG_PHI	= (BIG_PHI*180/np.pi)%180 		### Convert from Radian to Degree
	D_0,D_45,D_90,D_135 = FourAngles(BIG_PHI)	### Round the directions to 0, 45, 90, 135 (only take the masks)
	
	
	""" STEP-3: NON-Maximum Suppression
		--> Reference: https://stackoverflow.com/questions/46553662/conditional-value-on-tensor-relative-to-element-neighbors
	"""
	
	""" 3.1-Selecting Edge-Pixels on the Horizontal direction """	
	targetPixels_0 = tf.nn.convolution(G, filter_0, padding='SAME')
	isGreater_0 = tf.cast(tf.greater(G*D_0, targetPixels_0), tf.float32)
	isMax_0 = isGreater_0[:,:,:,0:1]*isGreater_0[:,:,:,1:2]
	### Note: Need to keep 4 dimensions (index [:,:,:,0] is 3 dimensions) ###
	
	""" 3.2-Selecting Edge-Pixels on the Vertical direction """
	targetPixels_90 = tf.nn.convolution(G, filter_90, padding='SAME')
	isGreater_90 = tf.cast(tf.greater(G*D_90, targetPixels_90), tf.float32)
	isMax_90 = isGreater_90[:,:,:,0:1]*isGreater_90[:,:,:,1:2]
	
	""" 3.3-Selecting Edge-Pixels on the Diag-45 direction """
	targetPixels_45 = tf.nn.convolution(G, filter_45, padding='SAME')
	isGreater_45 = tf.cast(tf.greater(G*D_45, targetPixels_45), tf.float32)
	isMax_45 = isGreater_45[:,:,:,0:1]*isGreater_45[:,:,:,1:2]
	
	""" 3.4-Selecting Edge-Pixels on the Diag-135 direction """
	targetPixels_135 = tf.nn.convolution(G, filter_135, padding='SAME')
	isGreater_135 = tf.cast(tf.greater(G*D_135, targetPixels_135), tf.float32)
	isMax_135 = isGreater_135[:,:,:,0:1]*isGreater_135[:,:,:,1:2]
	
	""" 3.5-Merging Edges on Horizontal-Vertical and Diagonal directions """
	edges_raw = G*(isMax_0 + isMax_90 + isMax_45 + isMax_135)
	edges_raw = tf.clip_by_value(edges_raw, 0, MAX)
	
	### If only the raw edges are needed ###
	if return_raw_edges: return tf.squeeze(edges_raw)
	
	
	""" STEP-4: Hysteresis Thresholding """
	edges_sure = tf.cast(tf.greater_equal(edges_raw, maxRate), tf.float32)
	edges_weak = tf.cast(tf.less(edges_raw, maxRate), tf.float32)*tf.cast(tf.greater_equal(edges_raw, minRate), tf.float32)
	
	edges_connected = tf.nn.convolution(edges_sure, filter_sure, padding='SAME')*edges_weak
	for _ in range(10): edges_connected = tf.nn.convolution(edges_connected, filter_sure, padding='SAME')*edges_weak
	
	edges_final = edges_sure + tf.clip_by_value(edges_connected,0,MAX)
	return tf.squeeze(edges_final)

#import matplotlib.pyplot as plt

#img = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/raw_aug_rgb_img/rgb_00023.png")
#img = tf.image.decode_png(img, 1)
#img = tf.image.resize(img, [224, 224]) / 255.
#img = TF_Canny(img[tf.newaxis, :, :, :], return_raw_edges=True)
#plt.imshow(img, cmap="gray")
#plt.show()