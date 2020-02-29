#
# STEREO CALIBRATION
# Author: Feliipe Costa
#
# Package importation
import numpy as np
import cv2
from matplotlib import pyplot as plt 

#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************
# Termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(0,69):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    t= str(i)
    ChessImaR= cv2.imread('./calibImages/chessboard-R'+t+'.png',0)    # Right side
    ChessImaL= cv2.imread('./calibImages/chessboard-L'+t+'.png',0)    # Left side
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               (9,6),None)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               (9,6),None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1],None,None)
hR,wR= ChessImaR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1],None,None)
hL,wL= ChessImaL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

print('Cameras Ready to use')

#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
#flags |= cv2.CALIB_USE_INTRINSIC_GUESS
#flags |= cv2.CALIB_FIX_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_ASPECT_RATIO
#flags |= cv2.CALIB_ZERO_TANGENT_DIST
#flags |= cv2.CALIB_RATIONAL_MODEL
#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_K3
#flags |= cv2.CALIB_FIX_K4
#flags |= cv2.CALIB_FIX_K5

cameraMatrix1 = None
distCoeffs1 = None
cameraMatrix2 = None
distCoeffs2 = None
R = None
T = None
E = None
F = None

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, ChessImaR.shape[::-1], flags = cv2.CALIB_FIX_INTRINSIC)

#StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image not croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped


#Save parameters into numpy file
np.save("./camera_params/retval", retval)
np.save("./camera_params/cameraMatrix1", cameraMatrix1)
np.save("./camera_params/distCoeffs1", distCoeffs1)
np.save("./camera_params/cameraMatrix2", cameraMatrix2)
np.save("./camera_params/distCoeffs2", distCoeffs2)
np.save("./camera_params/R", R)
np.save("./camera_params/T", T)
np.save("./camera_params/E", E)
np.save("./camera_params/F", F)

# recfify
np.save("./camera_params/RL", RL)
np.save("./camera_params/RR", RR)
np.save("./camera_params/PL", PL)
np.save("./camera_params/PR", PR)
np.save("./camera_params/Q", Q)
np.save("./camera_params/roiL", roiL)
np.save("./camera_params/roiR", roiR)


