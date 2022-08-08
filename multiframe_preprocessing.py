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


# Defining the test subset
# This set will only be used for the model performance evaluation
test_set = file_paths[-20:]

# Instantiating empty lists for storing each subject frame sequence and its respective labels
dataset = []
labels = []

# Iterating over filenames
for file in test_set:
    print(file)

    # Converting the multipage .tiff file to a n-dimensional array
    frames = tiff_to_ndarray(file)
    
    # Iterating over retrieved frames
    for x in frames:
        # Converting the frame color space to RGB since the VGG16 pre-trained model expects
        # images with 3 color channels
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        # Appending the preprocessed frame to the frame sequence list
        dataset.append(x)

    # Performing a regular expression on filenames to find the text pattern indicating the 
    # sober state acquisition period
    if re.search(pattern=r"_1_", string=file):
        # Definig the frame sequence label according to the acquisition period
        y = np.zeros(50, dtype=int)

        # Appending the defined label to the labels list
        labels.append(y)

        # Printing the label type, values and array size for sanity check
        print("\nzeros\n")
        print("{0}\n".format(y))
        print("{0}\n".format(len(y)))
    else:
        # Definig the frame sequence label to 1 (true) if the text pattern was not found
        y = np.ones(50, dtype=int)

        # Appending the defined label to the labels list
        labels.append(y)

        # Printing the label type, values and array size for sanity check
        print("\nones\n")
        print("{0}\n".format(y))
        print("{0}\n".format(len(y)))

# Performing a list comprehension operation to unfold the label subarrays into a list of labels
labels = [item for sublist in labels for item in sublist]

# Printing the dataset length
print("Dataset length: {0}".format(len(dataset)))
print("Label vector length: {0}\n".format(len(labels)))

# Printing the samples shape and data type
print("\n{0}".format(dataset[0].shape))
print(dataset[0].dtype)

print("\nDataset encoding...")

# Defining the sample and label sets filenames
sets = {
    "x": (r"Datasets\Raw\x_test-set.pkl", dataset),
    "y": (r"Datasets\Raw\y_test-set.pkl", labels)
    }

# Encoding the datasets into pickle files
for key in sets.keys():
    with open(sets[key][0], "wb") as file:
        pickle.dump(sets[key][1], file)
       