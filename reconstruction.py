#
# 3d reconstruction
# Author: Feliipe Costa
#


# Package importation
import cv2
import numpy as np 
from matplotlib import pyplot as plt 

#=====================================
# Function declarations
#=====================================

#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')


def remove_invalid(disp_arr, points, colors):
    mask = (
        (disp_arr > disp_arr.min()) &
        np.all(~np.isnan(points), axis=1) &
        np.all(~np.isinf(points), axis=1)
    )
    return points[mask], colors[mask]


#=========================================================
# Stereo 3D reconstruction 
#=========================================================

# 1 = LEFT
# 2 = RIGHT

retval = np.load("./camera_params/retval.npy")
cameraMatrix1 = np.load("./camera_params/cameraMatrix1.npy")
distCoeffs1 = np.load("./camera_params/distCoeffs1.npy")
cameraMatrix2 = np.load("./camera_params/cameraMatrix2.npy")
distCoeffs2 = np.load("./camera_params/distCoeffs2.npy")
R = np.load("./camera_params/R.npy")
T = np.load("./camera_params/T.npy")
E = np.load("./camera_params/E.npy")
F = np.load("./camera_params/F.npy")

# recfify
RL = np.load("./camera_params/RL.npy")
RR = np.load("./camera_params/RR.npy")
PL = np.load("./camera_params/PL.npy")
PR = np.load("./camera_params/PR.npy")
Q = np.load("./camera_params/Q.npy")
roiL = np.load("./camera_params/roiL.npy")
roiR = np.load("./camera_params/roiR.npy")

#***************************************
#**************LOAD FRAMES**************
#***************************************


frameR = cv2.imread('rframe190.jpg')   # Wenn 0 then Right Cam and wenn 2 Left Cam
frameL = cv2.imread('lframe190.jpg')

h,w = frameR.shape[:2]

Left_Stereo_Map = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, RL, PL, (w,h), cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, RR, PR, (w,h), cv2.CV_16SC2)


#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

#Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)


# win_size = 5
# min_disp = -1
# max_disp = 63 #min_disp * 9
# num_disp = max_disp - min_disp # Needs to be divisible by 16

# #Create Block matching object. 
# stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
# 	numDisparities = num_disp,
# 	blockSize = 5,
# 	uniquenessRatio = 5,
# 	speckleWindowSize = 5,
# 	speckleRange = 5,
# 	disp12MaxDiff = 2,
# 	P1 = 8*3*win_size**2,#8*3*win_size**2,
# 	P2 =32*3*win_size**2) #32*3*win_size**2)


#*************************************
#***** Starting the StereoVision *****
#*************************************

# Call the two cameras


# print(frameR.shape)
# print(map1x.shape)
# print(map1x[0])
# print(map1x[1])
# print(map1x)

# Rectify the images on rotation and alignement
Left_nice= None
Right_nice= None

Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, Left_nice, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, Right_nice, cv2.BORDER_CONSTANT, 0)


# Convert from color(BGR) to gray
grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

# Compute the 2 images for the Depth_image
disparity_map = stereo.compute(grayL,grayR) #.astype(np.float32)/ 16
# plt.imshow(disp,'gray')
# plt.show()

print(disparity_map.shape)
#Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q) #.reshape(-1, 3)
print(points_3D.shape)
#Get color points
colors = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2RGB) #.reshape(-1, 3)


#output_points, output_colors = remove_invalid(disparity_map.reshape(-1), points_3D, colors)

# #Get rid of points with value 0 (i.e no depth)
mask_map = disparity_map > disparity_map.min()

# #Mask colors and points. 
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

#Define name for output file
output_file = 'reconstructed.ply'

#Generate point cloud 
print ("\n Creating the output file... \n")
create_output(output_points, output_colors, output_file)
