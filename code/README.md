# Source code for producing the results and figures

The code is divided between Python scripts and Jupyter notebooks. The scripts implement functions for processing the database and analyzing the frame sequence of each subject. In this sense, the first step for reproducing our work is running the `multiframe_preprocessing.py` script. This script will retrieve individual frames from each multi-page .tiff file and convert them to the RGB color space, since the VGG16 pre-trained model expects images with 3 color channels. In addition, this script will return the balanced and imbalanced datasets used along the experiments described in the paper.

The `frame_histogram.py` script implement functions for performing the thermogram subtraction analysis (showed in the *Dataset description* subsection), the histogram comparison and the PSNR analysis (showed in the *Inference interpretation* subsection).

The notebooks perform the data analysis and processing and generate the figures for the paper.


## Analysis set up

> You can use this section to explain how to set up the experiment. In the Reed
> group we often submit 'jobs' (i.e., sets of model runs), which should be
> explained here.


## Generating results and figures

> You can use this area to explain how to process the outputs to generate the
> results and figures used in the paper. A makefile that automates the process
> would be ideal here. You can use text like the following:
