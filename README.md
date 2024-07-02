# SegCol Challenge 2024
### Semantic Segmentation for Tools and Fold Edges in Colonoscopy data

#### Challenge Website [https://www.synapse.org/#!Synapse:syn54124209](https://www.synapse.org/#!Synapse:syn54124209)

#### Contact: [endovis-weiss-vision@ucl.ac.uk](endovis-weiss-vision@ucl.ac.uk)


## Baseline
We have included [Dexined](Folds_Baseline_Dexined) and [DeepLab](Tools_Baseline_DeepLab) as baselines for folds edge detection and tool segmentation, respectively. After generating the `.npy` predictions, the output from both methods can be merged using [merge.py](merge.py) and then evaluated as done in [Evaluation](#evaluation)

We will use this baseline for Task2 evaluation. For each model, please refer to the instructions under the corresponding folders.

## Evaluation
Task 1 and 2 will be evaluated according to the eval.py script. 

### Task 1

The evaluation for Task 1 requires firstly generating predictions according to 'docker_templates'. 

#### 1. Generate predictions: 

```bash
cd docker_templates/Task1_dummy_docker
docker build -t <image_name> . 
docker run --gpus all -it --rm -v "$(pwd)/../../data:/data" <image_name> /data/input /data/output
```


#### 2. Evaluate:

```bash
cd ../..
python eval.py

```
### Task 2

Task 2 requires firstly generating a text file according to 'docker_templates', then checking if the text file content format is correct:

#### 1. Generate sample list: 

```bash
cd docker_templates/Task2_dummy_docker
docker build -t <image_name> . 
docker run --gpus all -it --rm -v "$(pwd)/../../data:/data" <image_name> /data/input /data/output
```


#### 2. Format check:


```bash
cd ../..
python Task2_file_format_check.py
```

The generated list of images with the training data will be used to retrain the baseline model and after training we will evaluate using [eval.py](eval.py).
