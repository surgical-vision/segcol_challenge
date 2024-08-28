# SegCol Task 2

You can create your list of filenames and their corresponding segmentation maps with **generate_ann.py**.

The DeepLab models can be trained by running train.py. 
Check the **argument_parser.py** to ensure a good configuration of running. 

### Setup in argument parser
*root-dir*         : location of your data folder

*train-img-file*   : the csv file with the names of the training images

*train-segm-file*  : the csv file with the names of the training annotations

*valid-img-file*   : the csv file with the names of the validation images

*valid-segm-file*  : the csv file with the names of the validation annotations


For saving the results, you can make use of **save_results.py**. Set up the **SAVE_PREDICTED_RESULTS = True**.

### Active Learning Loop

Active Learning is performed in the **train_active.py** script. You may use that script and develop your own active selection method similar to the baselines in the */active_selection/* folder.

train_active.py performs the active learning loop by partitioning the current training set as labelled and unlabelled. Therefore, at every cycle, it selects a new batch from the unlabelled partition of the training set and at it is added to the labelled set. The testing is performed on the validation set. The sample code is using DeepLab as a model, however we advise to use both DeepLab and Dexined (from task1) to simultate this cycle. The main reason is that the final evaluation it will be performed on the ensemble of Dexined for colon folds and DeepLab for tools.

### Saving the list of selected 

Once you have concluded to your active selection methodology, you can run **create_unlabelled_selection_list.py**. Please make the changes needed to the script. This will generate the txt file with the image names from the unlabelled set as: *candidate_unlabelled_imgs.txt* (check the example). This file will be needed for the evaluation where the combination of your selected unlabelled images and the current training set will test the performance of Dexined + DeepLab.

# Good Luck!
