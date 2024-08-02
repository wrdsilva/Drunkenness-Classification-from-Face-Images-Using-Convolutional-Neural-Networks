import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
from scipy.stats import mannwhitneyu


def tiff_to_ndarray(path):
    """
    Retrieves individual frames from a multipage .tiff file and returns them in n-dimensional array format.

    Parameters
    ----------
    path : str
        The path of the multipaged .tiff file to be decomposed.

    Returns
    -------
    ndarray
        A n-dimensional array containing the .tiff file retrieved frames.
    """
    
    # Instantiate an image object
    img = Image.open(path)

    # Instantiate an empty list of frames
    frames = []

    # Iterate over the frames sequence
    try:
        for i, page in enumerate(ImageSequence.Iterator(img)):
            # Appending the found frames to the frame list
            frames.append(np.array(page))
            
    except Exception as ex:
        print(ex)

    # Returning the frame list as numpy ndarray
    return np.asarray(frames)


def normalize_img(img_arr):
    """
    Receives a 16-bit frame and normalizes it to the 8-bit format with pixel values ranging from 0 to 255.

    Parameters
    ----------
    img_array : ndarray
        The frame to be normalized.

    Returns
    -------
    ndarray
        A 8-bit n-dimensional array with values ranging from 0 to 255.
    """

    normalized_img = np.zeros((128, 160))
    normalized_img = cv2.normalize(img_arr, normalized_img, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    normalized_img = cv2.convertScaleAbs(normalized_img)
    
    return normalized_img


def display_and_save(roi, right_side, left_side, save_plot=False, filename=None):
    """
    Displays the face regions of interest considered for pixel behavior analysis.

    Parameters
    ----------
    roi : ndarray
        The image obtained after extracting the face region from the background.
    right_side : ndarray
        A sliced ndarray representing the roi pixels from right to center.
    left_side : ndarray
        A sliced ndarray representing the roi pixels from center to left.
    save_plot : bool
        A flag indicating whether to save the plot.
    filename : str
        A string indicating the filename to save the plot.
    """

    # Displaying and saving the face region of interest
    plt.figure()
    plt.title("Region of Interest")
    plt.imshow(roi, cmap='gray')
    #plt.axis('off')

    if save_plot == True:
        plt.savefig("{0}_roi.pdf".format(filename), dpi=600, bbox_inches='tight', pad_inches=0.0)

    # Displaying and saving the face's right side
    plt.figure()
    plt.title("Face's right side")
    plt.imshow(right_side, cmap='gray')
    #plt.axis('off')

    if save_plot == True:
        plt.savefig("{0}_right-side.pdf".format(filename), dpi=600, bbox_inches='tight', pad_inches=0.0)

    # Displaying and saving the face's left side
    plt.figure()
    plt.title("Face's left side")
    plt.imshow(left_side, cmap='gray')
    #plt.axis('off')

    if save_plot == True:
        plt.savefig("{0}_left-side.pdf".format(filename), dpi=600, bbox_inches='tight', pad_inches=0.0)

    plt.show()


def find_centroid(image):
    """
    Calculates the image moments and returns its central points along the face region of interest.

    Parameters
    ----------
    image : ndarray
        A keyframe in grayscale.

    Returns
    -------
    cX : int
        The image horizontal axis central point index.
    cY : int
        The image vertical axis central point index.
    roi : ndarray
        A n-dimensional array representing the face region of interest extracted by an ellipse mask.
    """

    # Defining an image copy for the binarization process
    roi = image.copy()

    # Defining the region of interest n-dimensional array
    mask = np.zeros_like(roi)

    # Performing the image binarization
    ret, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_OTSU)
    
    # Calculating the binary image moments
    M = cv2.moments(thresh)
    
    # Calculating the centroid coordinates
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"]) # horizontal axis
        cY = int(M["m01"] / M["m00"]) # vertical axis
    else:
        cX, cY = 0, 0
    
    # Defining the ellipse mask central points
    center_coordinates = (cX+2, cY+2)

    # Defining the ellipse mask width and height 
    axesLength = (35, 55)

    # Defining the ellipse mask angle
    angle = 0
    startAngle = 0
    endAngle = 360


    # Ellipse mask parameters for each subject in sober and drunk (4th acquisition period) states:
 
    # subjects 37 to 41          subjects 32 to 36          subjects 27 to 31           subjects 22 to 26           subjects 16 to 21            subjects 11 to 15           subjects 6 to 10            subjects 1 to 5                                                                                                                   
    
    # [0] cx+2, cy, (36,60)      [0] cx+5, cy+5, (40,60)    [0] cx-4, cy+4, (35,55)     [0] cx, cy+1, (35,55)       [0] cx, cy+4, (36,50)        [0] cx, cy, (40,60)         [0] cx-2, cy, (40,60)       [0] cx, cy, (40,60)           
    # [3] cx, cy, (38, 60)       [3] cx+3, cy, (46, 66)     [3] cx-4, cy, (38,58)       [3] cx+2, cy+1, (35,55)     [3] cx, cy+8, (40,57)        [3] cx, cy, (40,60)         [3] cx-2, cy-2, (35,50)     [3] cx+3, cy-3, (34,54)                                                                               
    # [4] cx-1, cy+2, (40,60)    [4] cx+2, cy, (40,60)      [4] cx+1, cy+8, (43,60)     [4] cx-2, cy+10, (34,50)    [4] cx-5, cy+5, (42,60)      [4] cx+1, cy+4, (45,60)     [4] cx-2, cy-2, (35,50)     [4] cx-2, cy-4, (34,54)         
    # [7] cx-1, cy-2, (37, 57)   [7] cx+4, cy, (35, 55)     [7] cx+2, cy+6, (40,60)     [7] cx-2, cy+6, (40,60)     [7] cx-1, cy+5, (40,60)      [7] cx+2, cy+4, (42,60)     [7] cx-2, cy+6, (38,58)     [7] cx+3, cy, (36,60)                                                                             
    # [8] cx+4, cy+4, (40,60)    [8] cx, cy, (37,57)        [8] cx-2, cy+8, (45,60)     [8] cx, cy+4, (37,57)       [8] cx-1, cy+1, (32,52)      [8] cx-2, cy+2, (38,58)     [8] cx, cy+4, (42,60)       [8] cx+1, cy-1, (35,55)       
    # [11] cx+2, cy+4 (40,60)    [11] cx, cy, (35, 55)      [11] cx-1, cy+4, (43,60)    [11] cx+3, cy, (39,57)      [11] cx-4, cy-3, (32,50)     [11] cx-2, cy+2, (38,58)    [11] cx+2, cy+6, (40,57)    [11] cx+1, cy-1, (33,53)                                                                            
    # [12] cx+2, cy-2, (33,53)   [12] cx, cy, (40,60)       [12] cx-2, cy+4, (40,60)    [12] cx, cy+6, (37,57)      [12] cx-1, cy+4, (37,57)     [12] cx-2, cy, (40,60)      [12] cx+1, cy+6, (37,60)    [12] cx-1, cy, (40,60)  
    # [15] cx, cy-10 (30,45)     [15] cx+1, cy-2 (37,56)    [15] cx+1, cy+5 (40,60)     [15] cx+2, cy+6 (40,60)     [15] cx, cy, (40,60)         [15] cx, cy+4, (45,65)      [15] cx-1, cy+6, (38,60)    [15] cx+2, cy+2, (35,55)                                                                            
    # [16] cx+2, cy-8, (34,48)   [16] cx+3, cy+3, (35,55)   [16] cx+1, cy, (37,57)      [16] cx-1, cy+1, (34,48)    [16] cx-3, cy+2, (37,57)     [16] cx-2, cy, (37,57)      [16] cx-5, cy+5, (40,60)    [16] cx-4, cy+4, (40,60)      
    # [19] cx, cy+4, (36,52)     [19] cx-1, cy+1, (34,50)   [19] cx+1, cy, (37,57)      [19] cx, cy+4, (42,58)      [19] cx-1, cy+1, (35,50)     [19] cx-2, cy, (33,50)      [19] cx+2, cy+5, (42,60)    [19] cx-2, cy, (40,60)   


    # Defining the ellipse color and line thickness
    color = (255, 255, 255)
    thickness = -1
     
    # Drawing a white ellipse in a black background
    mask = cv2.ellipse(mask, center_coordinates, axesLength, angle, 
                             startAngle, endAngle, color, thickness)
    
    # Performing a bit-wise AND operation between the ellipse mask and the given keyframe 
    roi = np.bitwise_and(roi, mask)
    
    # Returning the keyframe centroid points and its region of interest
    return cX, cY, roi


def calculate_statistics(image, filename, subject_state):
    """
    Calculates the mean, standard deviation and variance of pixels on the right and left sides of a subject's face.

    Parameters
    ----------
    image : ndarray
        A subject keyframe obtained in the 1st or 4th acquisition period.
    filename : str
        A string indicating the subject filename for saving the image plots.
    subject_state : {'sober', 'drunk'}
        A string indicating the subject state for printing.
    """

    # Finding the keyframe centroid coordinates and its region of interest
    cx, cy, roi = find_centroid(image)

    # Normalizing the pixels scale [0-1]
    roi = roi / 255.0

    # Slicing the image array to find the face's left side
    left_side = roi[:, cx:]

    # Selecting the valid pixels for statistics calculation
    left_side_valid_pixels = left_side[left_side > 0.0] # only the pixels inside the ellipse mask

    # Calculating the mean, standard deviation and variance of pixels on the left side of a subject's face
    left_mean = np.mean(left_side_valid_pixels)
    left_std = np.std(left_side_valid_pixels)
    left_var = np.var(left_side_valid_pixels)

    # Slicing the image array to find the face's right side
    right_side = roi[:, :cx]

    # Selecting the valid pixels for statistics calculation
    right_side_valid_pixels = right_side[right_side > 0.0] # only the pixels inside the ellipse mask

    # Calculating the mean, standard deviation and variance of pixels on the right side of a subject's face
    right_mean = np.mean(right_side_valid_pixels)
    right_std = np.std(right_side_valid_pixels)
    right_var = np.var(right_side_valid_pixels)

    print(os.path.basename(filename))
    print(roi.shape)

    # Extracting the filename basename
    file_basename = os.path.basename(filename)

    # Printing the mean, standard deviation and variance of pixels on the right and left sides of a subject's face.
    print("\nFace from a subject in {0} state".format(subject_state))

    print("\nPixel average from face's right side: {0:0.2f}".format(right_mean))
    print("Pixels standard deviation from face's right side: {0:0.2f}".format(right_std))
    print("Pixel variance from face's right side: {0:0.2f}".format(right_var))

    print("\nPixel average from face's left side: {0:0.2f}".format(left_mean))
    print("Pixels standard deviation from face's left side: {0:0.2f}".format(left_std))
    print("Pixel variance from face's left side: {0:0.2f}".format(left_var))

    # Displaying the given subject's keyframe region of interest and its face's right and left sides
    display_and_save(roi=roi, right_side=right_side, left_side=left_side, filename=os.path.splitext(file_basename)[0])


def hypothesis_test():
    """
    Performs a Mann-Whitney hypothesis test to verify if the face's right and left sides pixel averages 
    vary accordingly to sober and drunk states.

    Notes
    -----
    Check the Scipy API reference (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html) 
    for more details regarding the Mann-Whitney hypothesis test.
    """

    # Defining the face's right side pixel average distribution regarding the sober state
    sober_right_avg = [
        0.83, 0.83, 0.78, 0.82, 0.78,
        0.90, 0.86, 0.80, 0.81, 0.81,
        0.78, 0.78, 0.78, 0.80, 0.76,
        0.80, 0.77, 0.85, 0.78, 0.79,
        0.82, 0.77, 0.71, 0.82, 0.85,
        0.74, 0.75, 0.83, 0.86, 0.84,
        0.88, 0.82, 0.82, 0.81, 0.80,
        0.80, 0.85, 0.86, 0.78, 0.84
        ]
    
    # Defining the face's left side pixel average distribution regarding the sober state
    sober_left_avg = [
        0.84, 0.84, 0.82, 0.84, 0.83,
        0.89, 0.88, 0.85, 0.84, 0.86,
        0.82, 0.83, 0.82, 0.85, 0.85,
        0.85, 0.86, 0.87, 0.88, 0.89,
        0.84, 0.82, 0.74, 0.80, 0.89,
        0.87, 0.86, 0.89, 0.91, 0.84,
        0.88, 0.82, 0.80, 0.82, 0.78,
        0.78, 0.85, 0.83, 0.75, 0.80
        ]
    
    # Defining the face's right side pixel average distribution regarding the drunk state (4th acquisition period)
    drunk_right_avg = [
        0.89, 0.86, 0.85, 0.87, 0.87,
        0.92, 0.84, 0.83, 0.78, 0.86,
        0.84, 0.83, 0.84, 0.84, 0.86,
        0.80, 0.82, 0.88, 0.80, 0.78,
        0.83, 0.80, 0.79, 0.83, 0.83,
        0.84, 0.86, 0.87, 0.86, 0.84,
        0.87, 0.85, 0.89, 0.85, 0.86,
        0.83, 0.86, 0.85, 0.86, 0.89
        ]
    
    # Defining the face's left side pixel average distribution regarding the drunk state (4th acquisition period)
    drunk_left_avg = [
        0.88, 0.85, 0.82, 0.86, 0.88,
        0.90, 0.85, 0.82, 0.80, 0.84,
        0.91, 0.85, 0.87, 0.89, 0.90,
        0.86, 0.85, 0.86, 0.90, 0.81,
        0.85, 0.81, 0.77, 0.77, 0.83,
        0.85, 0.82, 0.86, 0.86, 0.84,
        0.83, 0.85, 0.87, 0.83, 0.85,
        0.80, 0.86, 0.84, 0.85, 0.87
        ]
    
    # Running the Mann-Whitney hypothesis test
    print("=======================================================================================")

    print("\nHypothesis test regarding the face's right side thermal symmetry between the sober and drunk states")
    # A p-value less than alpha (0.05) rejects the Mann-Whitney test null hypothesis (that the distribution underlying 
    # sample x is the same as the distribution underlying sample y), indicating that the face's right side pixel average
    # distributions vary between the 1st and 4th acquisition periods.

    stat, p = mannwhitneyu(sober_right_avg, drunk_right_avg, alternative='two-sided')
    print("\nMann-Whitney p-value: {0}".format(p))

    print("\n=======================================================================================")

    print("\nHypothesis test regarding the face's left side thermal symmetry between the sober and drunk states")
    # A p-value less than alpha (0.05) rejects the Mann-Whitney test null hypothesis (that the distribution underlying 
    # sample x is the same as the distribution underlying sample y), indicating that the face's left side pixel average
    # distributions vary between the 1st and 4th acquisition periods.

    stat, p = mannwhitneyu(sober_left_avg, drunk_left_avg, alternative='two-sided')
    print("\nMann-Whitney p-value: {0}".format(p))

    print("\n=======================================================================================")


# Defining the database directory path
database_path = r"Insert the database directory path here"

# Defining the facial subset pattern
dataset_pattern = r"_f_"

# Defining the subject n 20 pattern
# These subjects samples were discarded due to the data unavailability 
# regarding the last acquisition period
remotion_pattern = r"_nefeli_"

# Instantiating an empty list for storing filenames
file_paths = []

# Iterating over the database directory folders
for dirpath, dirs, files in os.walk(database_path):
    for dir in dirs:
        subdir = os.path.join(database_path, dir)
        # Iterating over the database subdirectories
        for subdir_path, dirs, files in os.walk(subdir):
            for file in files:
                # Verifying if the filename corresponds to the facial subset
                if re.search(pattern=dataset_pattern, string=file):
                    # Verifying if the filename do not correspond to subject n 20
                    if not re.search(pattern=remotion_pattern, string=file):
                        # Defining a filename for the subject sample
                        subject = os.path.join(subdir, file)
                        # Appending the filename to the filenames list
                        file_paths.append(subject)

# Defining a set of samples for analysis
samples = file_paths[:20]

# Converting a multipage .tiff file to a n-dimensional array
sober_frames = tiff_to_ndarray(samples[12]) # 1st acquisition period indexes: 0, 4, 8, 12, 16
drunk_frames = tiff_to_ndarray(samples[15]) # 4th acquisition period indexes: 3, 7, 11, 15, 19

# Normalizing the sample data type and pixel scale 
sample = normalize_img(drunk_frames[0])

# Defining the sample file path
sample_path = samples[15]

# Calculating statics from a subject facial image
calculate_statistics(image=sample, filename=sample_path, subject_state='drunk')
