## Project experimentations
This folder contains all the code used to perform all the experimentations in this projet. 

The script argument "-i" or "--input_folder" should point to the RVL-CDIP root folder. In this root folder, there should have the "images" and "labels" folders.

**classes/datasets.py** : contains the DataModule and Dataset implementation for RLV-CDIP to work with PyTorch and PyTorch Lightning. It can use textual mode, visual mode or an hybrid approch.

**classes/load_models.py** : it contains all my models implementations for PyTorch Lightning. Here are defined the initial configuration of the model, the modification added to the strcture and méthodes used in train/test/val steps. 

**classes/utils.py** : useful methods like the classes list to indexes or a custom resize method that I don't acutally use.

**iit_cdip_matching/main.py** : This script is used to match the textual content of RVL-CDIP's dataset from ITT-CDIP dataset by copying the text file into RVL-CDIP folder.

```
python main.py  --input_folder "../path/to/iit-cdip" -output_folder "../path/to/rvl-cdip"
```

**Model training files** :
 - **mobilenetv3.py** : used to train a mobilenetv3-large model. This scripts accepts an additional argument *-p/--pretrained* to determine if the ImageNet's weight must be used. 
 - **resnet50.py** : used to train a resnet50 model.
 - **roberta.py** : used to train a RoBERTa for textual classification.
 - **hybrid.py** : used to train the first hybrid model composed of a RoBERTa and a ResNet50. 
 - **multicnn.py** : used to train the second hybrid model composed of a MobileNetV3-Large and ResNet50. This script trains a final classification bloc and don't retrain the whole two model
  
Example command for one model train :

```
python resnet50.py  -i "../data" -b 16 -e 30 -l 0.001 -o "adam" -g 4
```

**infer_model.py** : this script is used to test the *Donut* and *DiT* model as they are already fine-tuned on RVL-CDIP for classification. A *PretrainedModel* class is used to infer one of these models. A text file containing a list of predicted class and target class is created.

Example command :

```
python infer_model.py  -i "../data" -m "donut" -o "tests"
```


**test_model.py** : this script is used to test a model that I just train in order to get the test accuracy and create a confusion matrice later. Currently, it works on ResNet50, MobileNetV3-Large (base and pre-trained version). I encoutered a problem loading my trained RoBERTa model and the same problem would happen with the hybrid model.

Example command :

```
python test_model.py  -i "../data" -m "resnet50" -b 16
```