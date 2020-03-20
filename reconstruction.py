#
# 3d reconstruction
# Author: Feliipe Costa
#


# Package importation
import cv2
import numpy as np 

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
R1 = np.load("./camera_params/R1.npy")
R2 = np.load("./camera_params/R2.npy")
P1 = np.load("./camera_params/P1.npy")
P2 = np.load("./camera_params/P2.npy")
Q = np.load("./camera_params/Q.npy")
roi1 = np.load("./camera_params/roi1.npy")
roi2 = np.load("./camera_params/roi2.npy")

# focal_length = 652.070950915


# Q = np.float32([[1,0,0,0],
#                 [0,-1,0,0],
#                 [0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally. 
#                 [0,0,0,1]])

# Q = np.float32([[1,0,0,-640/2.0],
#                 [0,-1,0,480/2.0],
#                 [0,0,0,-focal_length],
#                 [0,0,1,0]])

#***************************************
#**************LOAD FRAMES**************
#***************************************


imgR = cv2.imread('direita.jpg')   # Wenn 0 then Right Cam and wenn 2 Left Cam
imgL = cv2.imread('esquerda.jpg')

h,w = imgR.shape[:2]
left_maps = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w,h), cv2.CV_16SC2)
right_maps = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w,h), cv2.CV_16SC2)

r_imgL = cv2.remap(imgL, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
r_imgR = cv2.remap(imgR, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)

# cv2.imshow('origr', imgR)
# cv2.imshow('origil', imgL)

# cv2.imshow('rName.jpg', r_imgL)
# cv2.imshow('lName.jpg', r_imgR)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
# window_size = 3
# min_disp = 2
# num_disp = 130-min_disp
# stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
#     numDisparities = num_disp,
#     blockSize = window_size,
#     uniquenessRatio = 10,
#     speckleWindowSize = 100,
#     speckleRange = 32,
#     disp12MaxDiff = 5,
#     P1 = 8*3*window_size**2,
#     P2 = 32*3*window_size**2)


win_size = 5
min_disp = 2
max_disp = min_disp * 9 #63
num_disp = max_disp - min_disp # Needs to be divisible by 16

#Create Block matching object. 
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
	numDisparities = num_disp,
	blockSize = win_size,
	uniquenessRatio = 5,
	speckleWindowSize = 5,
	speckleRange = 5,
	disp12MaxDiff = 2,
	P1 = 8*3*win_size**2,#8*3*win_size**2,
	P2 =32*3*win_size**2) #32*3*win_size**2)


# #*************************************
# #***** Starting the StereoVision *****
# #*************************************

# # Call the two cameras


# # print(frameR.shape)
# # print(map1x.shape)
# # print(map1x[0])
# # print(map1x[1])
# # print(map1x)

# # Rectify the images on rotation and alignement
# Left_nice= None
# Right_nice= None

# Left_nice = cv2.remap(imgL, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4) #cv2.remap(imgL, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4, Left_nice, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
# Right_nice = cv2.remap(imgR, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)  #cv2.remap(imgR, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4, Right_nice, cv2.BORDER_CONSTANT, 0)


####
# Left_nice_rl = Left_nice
# Right_nice_rl = Right_nice
# frameR_rl = frameR
# frameL_rl = frameL

##    # Draw Red lines
# for line in range(0, int(Right_nice_rl.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
#     Left_nice_rl[line*20,:]= (0,0,255)
#     Right_nice_rl[line*20,:]= (0,0,255)

# for line in range(0, int(frameR_rl.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
#     frameL_rl[line*20,:]= (0,255,0)
#     frameR_rl[line*20,:]= (0,255,0)    
    
# Show the Undistorted images
#cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
#cv2.imshow('Normal', np.hstack([frameL, frameR]))

#RL
# cv2.imshow('Both Images REDLINE', np.hstack([Left_nice_rl, Right_nice_rl]))
# cv2.imshow('Normal REDLINE', np.hstack([frameL_rl, frameR_rl]))

# Convert from color(BGR) to gray
grayR= cv2.cvtColor(r_imgL,cv2.COLOR_BGR2GRAY)
grayL= cv2.cvtColor(r_imgR,cv2.COLOR_BGR2GRAY)

# Compute the 2 images for the Depth_image
disparity_map = stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
# plt.imshow(disp,'gray')
# plt.show()

print(disparity_map.shape)
#Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q) #.reshape(-1, 3)
print(points_3D.shape)
#Get color points
colors = cv2.cvtColor(r_imgR, cv2.COLOR_BGR2RGB) #.reshape(-1, 3)


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
