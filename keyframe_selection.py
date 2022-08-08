import os
import re
import cv2
import numpy as np
from PIL import Image, ImageSequence
import pickle  


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


def frame_averaging(files, dataset_type):
    """
    Receives a list of filenames and averages each subject samples into a single frame.

    Parameters
    ----------
    files : list
        A list of multipage .tiff filenames.
    dataset_type : {'training', 'test'}
        A string indicating the subset name for saving the dataset file.
    """

    # Instantiating empty lists for storing the average frames and its respective labels
    dataset = []
    labels = []

    # Iterating over filenames 
    for file in files:
        # Perform a regular expression on filenames to find the text pattern indicating the 
        # sober state acquisition period
        if re.search(pattern=r"_1_", string=file):
            print(file)
            
            # Converting the multipage .tiff file matching the pattern to a n-dimensional array
            frames = tiff_to_ndarray(file)
            
            # Define the reference frame
            reference_frame = normalize_img(frames[0])

            # Define the list of frames to be aligned
            frames_to_align = frames[1:]

            # Define an empty list for storing the aligned frames
            aligned_frames = []

            # Appending the reference frame to the aligned frames list
            aligned_frames.append(reference_frame)
            
            # Iterating over the list of frames to be aligned
            for frame in frames_to_align:
                # Run the alignment algorithm
                aligned_frame = ecc_frame_alignment(reference_frame, normalize_img(frame))

                # Appending the aligned frame to the aligned frames list
                aligned_frames.append(aligned_frame)

            # Calculating the average frame from the aligned frames list
            # the average frame data type was defined as int16 since it is the samples original type
            frame_average = np.mean(aligned_frames, axis=0, keepdims=True, dtype=np.uint16)

            # Converting the average frame color space to RGB since the VGG16 pre-trained model expects
            # images with 3 color channels
            x = cv2.cvtColor(frame_average[0], cv2.COLOR_GRAY2RGB)

            # Appending the averaged frame to the average frames list
            dataset.append(x)

            # Definig the sample label according to the acquisition period
            y = 0

            # Appending the defined label to the labels list
            labels.append(y)

            print("Class: {0}\n".format(y))
        
        # Perform a regular expression on filenames to find the text pattern indicating the 
        # drunk state acquisition period (90 minutes after alcohol ingestion)
        if re.search(pattern=r"_4_", string=file):
            print(file)

            # Converting the multipage .tiff file matching the pattern to a n-dimensional array
            frames = tiff_to_ndarray(file)
            
            # Define the reference frame
            reference_frame = normalize_img(frames[0])

            # Define the list of frames to be aligned
            frames_to_align = frames[1:]
            
            # Define an empty list for storing the aligned frames
            aligned_frames = []

            # Appending the reference frame to the aligned frames list
            aligned_frames.append(reference_frame)

            # Iterating over the list of frames to be aligned
            for frame in frames_to_align:
                # Run the alignment algorithm
                aligned_frame = ecc_frame_alignment(reference_frame, normalize_img(frame))
                
                # Appending the aligned frame to the aligned frames list
                aligned_frames.append(aligned_frame)
            
            # Calculating the average frame from the aligned frames list
            # the average frame data type was defined as int16 since it is the samples original type
            frame_average = np.mean(aligned_frames, axis=0, keepdims=True, dtype=np.uint16)

            # Converting the average frame color space to RGB since the VGG16 pre-trained model expects
            # images with 3 color channels
            x = cv2.cvtColor(frame_average[0], cv2.COLOR_GRAY2RGB)

            # Appending the averaged frame to the average frames list
            dataset.append(x)
            
            # Definig the sample label according to the acquisition period
            y = 1

            # Appending the defined label to the labels list
            labels.append(y)

            print("Class: {0}\n".format(y))

    # Printing the dataset length
    print("Dataset length: {0}".format(len(dataset)))
    print("Label vector length: {0}\n".format(len(labels)))

    print("\nDataset encoding...")

    # Defining the sample and label sets filenames
    sets = {
        "x": (r"Datasets\Balanced\frame_average\x_{0}-set.pkl".format(dataset_type), dataset),
        "y": (r"Datasets\Balanced\frame_average\y_{0}-set.pkl".format(dataset_type), labels)
        }
    
    # Encoding the datasets into pickle files
    for key in sets.keys():
        with open(sets[key][0], "wb") as file:
            pickle.dump(sets[key][1], file)


def balanced_dataset_keyframe_selection(files, keyframe_index, dataset_type):
    """
    Receives a list of filenames and saves a balanced subset of sober and drunk samples regarding a given keyframe.

    This function encodes the samples obtained prior to alcohol consumption and 90 minutes after the last dose ingestion.

    Parameters
    ----------
    files : list
        A list of multipage .tiff filenames.
    keyframe_index : int
        An index to a specific frame from the frame sequence.
    dataset_type : {'training', 'test'}
        A string indicating the subset name for saving the dataset file.
    """

    # Instantiating empty lists for storing the keyframes and its respective labels
    dataset = []
    labels = []

    # Iterating over filenames 
    for file in files:
        # Perform a regular expression on filenames to find the text pattern indicating the 
        # sober state acquisition period
        if re.search(pattern=r"_1_", string=file):
            print(file)

            # Converting the multipage .tiff file matching the pattern to a n-dimensional array
            frames = tiff_to_ndarray(file)

            # Converting the keyframe color space to RGB since the VGG16 pre-trained model expects
            # images with 3 color channels
            x = cv2.cvtColor(frames[keyframe_index], cv2.COLOR_GRAY2RGB)

            # Appending the selected keyframe to the keyframes list
            dataset.append(x)

            # Definig the sample label according to the acquisition period
            y = 0

            # Appending the defined label to the labels list
            labels.append(y)

            print("Class: {0}\n".format(y))
        
        # Perform a regular expression on filenames to find the text pattern indicating the 
        # drunk state acquisition period (90 minutes after alcohol ingestion)
        if re.search(pattern=r"_4_", string=file):
            print(file)

            # Converting the multipage .tiff file matching the pattern to a n-dimensional array
            frames = tiff_to_ndarray(file)

            # Converting the keyframe color space to RGB since the VGG16 pre-trained model expects
            # images with 3 color channels
            x = cv2.cvtColor(frames[keyframe_index], cv2.COLOR_GRAY2RGB)

            # Appending the selected keyframe to the keyframes list
            dataset.append(x)

            # Definig the sample label according to the acquisition period
            y = 1

            # Appending the defined label to the labels list
            labels.append(y)

            print("Class: {0}\n".format(y))
    
    # Printing the dataset length
    print("Dataset length: {0}".format(len(dataset)))
    print("Label vector length: {0}\n".format(len(labels)))

    # Defining the keyframe ordinal suffix
    if keyframe_index == 0:
        order = 'st'
    else:
        order = 'th'

    # Defining the keyframe folder name
    keyframe_folder = '{0}'.format(keyframe_index + 1) + order

    print("Keyframe: {0}".format(keyframe_folder))

    print("\nDataset encoding...")

    # Defining the sample and label sets filenames
    sets = {
        "x": (r"Datasets\Balanced\{0}_frame\x_{1}-set.pkl".format(keyframe_folder, dataset_type), dataset),
        "y": (r"Datasets\Balanced\{0}_frame\y_{1}-set.pkl".format(keyframe_folder, dataset_type), labels)
        }

    # Encoding the datasets into pickle files
    for key in sets.keys():
        with open(sets[key][0], "wb") as file:
            pickle.dump(sets[key][1], file)


def unbalanced_dataset_keyframe_selection(files, keyframe_index, dataset_type):
    """
    Receives a list of filenames and saves a subset of sober and drunk samples regarding a given keyframe.

    This function encodes the samples obtained in all acquisition periods.

    Parameters
    ----------
    files : list
        A list of multipage .tiff filenames.
    keyframe_index : int
        An index to a specific frame from the frame sequence.
    dataset_type : {'training', 'test'}
        A string indicating the subset name for saving the dataset file.
    """

    # Instantiating empty lists for storing the keyframes and its respective labels
    dataset = []
    labels = []

    # Iterating over filenames
    for file in files:
        print(file)

        # Converting the multipage .tiff file to a n-dimensional array
        frames = tiff_to_ndarray(file)

        # Converting the keyframe color space to RGB since the VGG16 pre-trained model expects
        # images with 3 color channels
        x = cv2.cvtColor(frames[keyframe_index], cv2.COLOR_GRAY2RGB)

        # Appending the selected keyframe to the keyframes list
        dataset.append(x)

        # Performing a regular expression on filenames to find the text pattern indicating the 
        # sober state acquisition period
        if re.search(pattern=r"_1_", string=file):
            # Definig the sample label according to the acquisition period
            y = 0

            # Appending the defined label to the labels list
            labels.append(y)

            print("Class: {0}\n".format(y))
        else:
            # Definig the sample label to 1 (true) if the text pattern was not found
            y = 1

            # Appending the defined label to the labels list
            labels.append(y)

            print("Class: {0}\n".format(y))

    # Printing the dataset length
    print("Dataset length: {0}".format(len(dataset)))
    print("Label vector length: {0}\n".format(len(labels)))

    # Defining the keyframe ordinal suffix
    if keyframe_index == 0:
        order = 'st'
    else:
        order = 'th'

    # Defining the keyframe folder name
    keyframe_folder = '{0}'.format(keyframe_index + 1) + order

    print("Keyframe: {0}".format(keyframe_folder))

    print("\nDataset encoding...")

    # Defining the sample and label sets filenames
    sets = {
        "x": (r"Datasets\Unbalanced\{0}_frame\x_{1}-set.pkl".format(keyframe_folder, dataset_type), dataset),
        "y": (r"Datasets\Unbalanced\{0}_frame\y_{1}-set.pkl".format(keyframe_folder, dataset_type), labels)
        }

    # Encoding the datasets into pickle files
    for key in sets.keys():
        with open(sets[key][0], "wb") as file:
            pickle.dump(sets[key][1], file)


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

# Defining the training subset 
# This set will be subdivided again into training and validation subsets in later stages
training_set = file_paths[:-20]

# Defining the test subset
# This set will only be used for the model performance evaluation
test_set = file_paths[-20:]

# Defining the keyframe index
keyframe_index = 0

# Encoding the keyframe-based balanced training set
balanced_dataset_keyframe_selection(training_set, keyframe_index, 'training')

print("\n====================================================================================\n")

# Encoding the keyframe-based balanced test set
balanced_dataset_keyframe_selection(training_set, keyframe_index, 'test')
    