# EndoVis SegCol Challenge
## Generating dataloader txt files
You can also create your list of filenames and their corresponding segmentation maps with generate_ann.py.
## Training the model
The models can be trained by running train.py. Check the argument_parser.py to ensure a good configuration of running. Also, set up your root directory in train.py.
## Saving the results
For saving the results, you can make use of save_results.py 
## Model
A DeepLab-based pre-trained model with DRN backbone can be found [here](https://drive.google.com/drive/folders/1MUWG3oRP4jV7eRL5lE-qEqvzZv0_tqcp?usp=drive_link). This can be used to get some preliminary results of the tools and colon folds detection. We recommend using the Dexined for the colon folds prediction instead of this model. However, for tools segmentation you may prefer a model similar to this one.
