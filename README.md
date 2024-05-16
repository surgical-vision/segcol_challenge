# SegCol Challenge 2024
## Semantic Segmentation for Tools and Fold Edges in Colonoscopy data

[Challenge Website](https://www.synapse.org/#!Synapse:syn54124209/wiki/626563)

Contact: [endovis-weiss-vision@ucl.ac.uk](endovis-weiss-vision@ucl.ac.uk)



## Evaluation
Task 1 and 2 will be evaluated according to the eval.py script. 

The evaluation for Task 1 requires fistly generating predictions according to 'docker_templates'. 

1. Generate predictions: 

```bash
cd docker_templates/Task1_dummy_docker
docker build -t <image_name> . 
docker run --gpus all -it --rm -v "$(pwd)/../../data:/data" <image_name> /data/input /data/output
```


2. Evaluate:

```bash
cd ../..
python eval.py

```

Task 2 requires fistly generating a text file according to 'docker_templates', then checking if the text file content format is correct:

1. Generate sample list: 

```bash
cd docker_templates/Task2_dummy_docker
docker build -t <image_name> . 
docker run --gpus all -it --rm -v "$(pwd)/../../data:/data" <image_name> /data/input /data/output
```


2. Format check:


```bash
cd ../..
python Task2_file_format_check.py
```

The generated list of images with the training data will be used to retrain the baseline model and after training we will evaluate using [eval.py](eval.py).