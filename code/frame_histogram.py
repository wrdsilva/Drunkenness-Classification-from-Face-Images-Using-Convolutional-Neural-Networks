import os
import re
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sewar.full_ref import psnr
from PIL import Image, ImageSequence


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


def ecc_frame_alignment(reference_frame, frame_to_align):
    """
    Receives two frames and align them according to the Enhanced Correlation Coefficient (ECC) criterion. 

    Parameters
    ----------
    reference_frame : ndarray
        A frame that will be the reference for the alignment algorithm.
    frame_to_align : ndarray
        A frame that needs to be aligned.
    
    Returns
    -------
    ndarray
        A frame aligned in accordance with the reference image.
    """

    # Find the reference image size
    image_size = reference_frame.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the identity matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations
    number_of_iterations = 50

    # Specify the correlation coefficient increment threshold between two iterations
    termination_eps = 1e-3

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix
    cc, warp_matrix = cv2.findTransformECC(reference_frame, frame_to_align, warp_matrix, 
                                           warp_mode, criteria, inputMask=None, gaussFiltSize=1)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        aligned_frame = cv2.warpPerspective(frame_to_align, warp_matrix, (image_size[1], image_size[0]), 
                                            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        aligned_frame = cv2.warpAffine(frame_to_align, warp_matrix, (image_size[1], image_size[0]), 
                                       flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

    # return the aligned frame
    return aligned_frame


def display_and_save(first_frame, last_frame, aligned_frame, frame_difference, save_plot=False, filename=None):
    """
    Displays the element-wise subtraction between the final and initial frames of a sequence, \\
    and these frames histograms comparison.

    Parameters
    ----------
    first_frame : ndarray
        The initial frame from a frame sequence.
    last_frame : ndarray
        The final frame from a frame sequence.
    aligned_frame : ndarray
        The frame obtained after running the ecc alignment algorithm.
    frame_difference : ndarray
        The element-wise subtraction between the final and initial frames.
    save_plot : bool
        A flag indicating whether to save the plot.
    filename : str
        A string indicating the filename to save the plot.
    """
    
    # Displaying and saving the initial frame
    plt.figure()
    plt.title("First frame")
    plt.imshow(first_frame, cmap="gray")
    plt.axis('off')

    if save_plot == True:
        plt.savefig("{0}_first-frame_test.pdf".format(filename), dpi=600, bbox_inches='tight', pad_inches=0)

    # Displaying and saving the final frame
    plt.figure()
    plt.title("Last frame")
    plt.imshow(last_frame, cmap="gray")
    plt.axis('off')

    if save_plot == True:
        plt.savefig("{0}_last-frame_test.pdf".format(filename), dpi=600, bbox_inches='tight', pad_inches=0)

    # Displaying and saving the aligned frame
    plt.figure()
    plt.title("Aligned frame")
    plt.imshow(aligned_frame, cmap="gray")
    plt.axis('off')

    if save_plot == True:
        plt.savefig("{0}_aligned-frame_test.pdf".format(filename), dpi=600, bbox_inches='tight', pad_inches=0)

    # Displaying and saving the element-wise subtraction between the initial and final frames
    plt.figure()
    plt.title("Last frame - First frame")
    plt.imshow(frame_difference, cmap="gray")
    plt.axis('off')

    if save_plot == True:
        plt.savefig("{0}_frame-difference_test.pdf".format(filename), dpi=600, bbox_inches='tight', pad_inches=0)

    # Displaying and saving the histogram comparison
    plt.figure()
    plt.title("Histogram comparison")
    plt.hist((first_frame.ravel(), last_frame.ravel()), bins='auto', histtype="bar", label=("First frame", "Last frame"), color=('gold', 'red'), alpha=0.8)
    plt.xlabel("Grayscales")
    plt.ylabel("Pixels")
    plt.legend()

    if save_plot == True:
        plt.savefig("{0}_histogram-comparison_test.pdf".format(filename), dpi=600, bbox_inches='tight', pad_inches=0.01)

    # Showing all plots
    plt.show()


def display_and_save_psnr(frames, save_plot=False, filename=None):
    """
    Calaculates the Peak Signal-to-Noise Ratio between the frames of a sequence and displays it.

    Parameters
    ----------
    frames : ndarray
        A frame sequence.
    save_plot : bool
        A flag indicating whether to save the plot.
    filename : str
        A string indicating the filename to save the plot.
    """

    # Instantiating an empty list for storing the psnr between the fisrt frame and remaining frames of the sequence
    psnr_list = []

    # Iterating over the frame sequence
    for i in range(len(frames)):
        # Calculating the psnr between the first frame and i-th frame of the sequence
        # Appending the obtained value to the psnr list
        psnr_list.append(psnr(frames[0], frames[i], 255))
        print("PSNR between frame 0 and frame {0}: {1}".format(i, psnr_list[i]))

    # Since the psnr between a frame to itself tends to infinite,
    # we defined a maximum value to be showed on the y axis before 
    # the infinity symbol. This value was defined after observing
    # all psnr values in a prior test
    psnr_list[0] = 51

    # Aranging the plot y axis values
    labels = np.arange(np.min(psnr_list), 55, 5)

    # Converting the y axis values to string
    labels = [float(i) for i in labels]
    labels = ["{0:.2f}".format(i) for i in labels]

    # Changing the y axis highest value label to the infinity symbol
    labels[-1] = 'inf'  # r"$\infty$"

    # Defining the plot colormap
    tab20c_colormap = matplotlib.cm.get_cmap("tab20c")

    # Displaying and saving the PSNR comparison
    plt.figure()
    plt.plot(psnr_list, '.-', linewidth=1.75, color=tab20c_colormap(2), markerfacecolor=tab20c_colormap(0), markeredgecolor=tab20c_colormap(0))  # color="lightsalmon", markerfacecolor="tomato", markeredgecolor="tomato"
    plt.xlabel("Sequence of frames")
    plt.ylabel("PSNR (dB)")
    plt.yticks(np.arange(np.min(psnr_list), 55, 5), labels)
    
    if save_plot == True:
        plt.savefig("{0}_psnr-comparison_test.pdf".format(filename), dpi=600, bbox_inches='tight', pad_inches=0.01)
    
    # Showing the psnr comparison
    plt.show()


def dataset_histogram_analysis():
    """
    Iterates over each subject frame sequence in order to verify thermal fluctuations occurrences.
    """

    # Defining the database directory path
    database_path = r"Insert the database directory path here"

    # Defining the facial subset pattern
    pattern = r"_f_"

    # Defining an iteration counter
    count = 0

    # Iterating over the database directory folders
    for dirpath, dirs, files in os.walk(database_path):
        for dir in dirs:
            subdir = os.path.join(database_path, dir)
            # Iterating over the database subdirectories
            for subdir_path, dirs, files in os.walk(subdir):
                for file in files:
                    # Verifying if the filename corresponds to the facial subset
                    if re.search(pattern=pattern, string=file):
                        subject = os.path.join(subdir, file)
                        print(file)
                        
                        # Converting the multipage .tiff file to a n-dimensional array
                        frames = tiff_to_ndarray(subject)
                        print("{0}\n".format(frames.shape))

                        # Selecting the frame sequence initial frame
                        first_frame = normalize_img(frames[0])

                        # Selecting the frame sequence final frame
                        last_frame = normalize_img(frames[-1])

                        # Aligning the final and initial frames according to the ecc criteria
                        last_frame = ecc_frame_alignment(first_frame, last_frame)

                        # Performing the element-wise subtraction between the final and initial frames
                        diff = last_frame - first_frame

                        # Displaying plots for analysis
                        display_and_save(first_frame=frames[0], last_frame=frames[49], 
                                         aligned_frame=last_frame, frame_difference=diff)

                        display_and_save_psnr(frames=frames)

                        '''
                        Example of how to use the display function to save the plots:

                        display_and_save(first_frame=frames[0], last_frame=frames[49], 
                                         aligned_frame=last_frame, frame_difference=diff,
                                         save_plot=True, filename=os.path.splitext(file)[0])
                        '''

                        input("\nPress any key to continue...\n")

                        # Uncomment line below to break the loop over a subject samples
                        #break

            # Increment the iterations counter
            #count = count+1
           
           # Uncomment the line below to break the database subdirectory iterations  
           # if count == 1:
           #     break
        # Uncomment the line below to break the database directory iterations
        #break


def sample_histogram_analysis():
    """
    Displays the subject n° 37 initial and final frames subtraction, histogram comparison and psnr comparison
    to exemplify the database thermal fluctuations analysis.
    """

    # Defining the subject n 37 directory path
    subject = r"Insert the file path relative to the 2nd acquisition period"

    # Converting the multipage .tiff file to a n-dimensional array
    frames = tiff_to_ndarray(subject)
    print("{0}\n".format(frames.shape))

    # Selecting the frame sequence initial frame
    first_frame = normalize_img(frames[0])

    # Selecting the frame sequence final frame
    last_frame = normalize_img(frames[-1])

    # Aligning the final and initial frames according to the ecc criteria
    last_frame = ecc_frame_alignment(first_frame, last_frame)

    # Performing the element-wise subtraction between the final and initial frames
    diff = last_frame - first_frame

    # Displaying plots for analysis
    display_and_save(first_frame=frames[0], last_frame=frames[49], 
                        aligned_frame=last_frame, frame_difference=diff)

    display_and_save_psnr(frames=frames)


sample_histogram_analysis()
