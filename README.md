# 🩺 SegCol Challenge 2024
### 🌟 Semantic Segmentation for Tools and Fold Edges in Colonoscopy data

#### 🌐 Challenge Website [https://www.synapse.org/#!Synapse:syn54124209](https://www.synapse.org/#!Synapse:syn54124209)

#### 📧 Contact: [endovis-weiss-vision@ucl.ac.uk](endovis-weiss-vision@ucl.ac.uk)


## 📈 Baseline
We have included [Dexined](Folds_Baseline_Dexined) and [DeepLab](Tools_Baseline_DeepLab) as baselines for folds edge detection and tool segmentation, respectively. After generating the `.npy` predictions, the output from both methods can be merged using [merge.py](merge.py) and then evaluated as done in [Evaluation](#evaluation)

We will use this two baseline models for Task2 evaluation. For each model, please refer to the instructions under the corresponding folders.

## 📋 Model Performance
Task 1 baseline scores are shown below. 

|      **Classes**    | **ODS** | **OIS** | **Dice** | **AP**  | **CLDice** |**Threshold**|
|---------------------|---------|---------|----------|---------|---------|---------|
| **Folds (Dexined)** | 0.2461  | 0.2525  | 0.2684   | 0.0894  | 0.2736  |  0.82   |
| **Tool 1 (Deeplab)** | 0.0546  | 0.0721 | 0.3636   | 0.1330  | 0.2440  |  0.65   |
| **Tool 2 (Deeplab)** | 0.0489  | 0.0698 | 0.0930   | 0.0331  | 0.0414  |  0.37   |
| **Tool 3 (Deeplab)** | 0.1214  | 0.1255 | 0.8041   | 0.6515  | 0.2837  |  0.64   |


## 📊 Evaluation
Task 1 and 2 will be evaluated according to the eval.py script. 

### 📝 Task 1

The evaluation for Task 1 requires firstly generating predictions according to 'docker_templates'. 

#### 1️⃣ Generate predictions: 

```bash
cd docker_templates/Task1_dummy_docker
docker build -t <image_name> . 
docker run --gpus all -it --rm -v "$(pwd)/../../data:/data" <image_name> /data/input /data/output
```


#### 2️⃣ Evaluate:

```bash
cd ../..
python eval.py

```
### 📝 Task 2

Task 2 requires firstly generating a text file according to 'docker_templates', then checking if the text file content format is correct:

#### 1️⃣ Generate sample list: 

```bash
cd docker_templates/Task2_dummy_docker
docker build -t <image_name> . 
docker run --gpus all -it --rm -v "$(pwd)/../../data:/data" <image_name> /data/input /data/output
```


#### 2️⃣ Format check:


```bash
cd ../..
python Task2_file_format_check.py
```

The generated list of images with the training data will be used to retrain the baseline model and after training we will evaluate using [eval.py](eval.py).


#### 3️⃣ Active learning strategy:

We provide an example of active selection for Task 2 in folder [Active_Learning](Active_Learning). Even though this example is based on DeepLab, in Task2 we will evaluate on both DexiNed and DeepLab for Folds and Tools respectively by training on the selected data list.