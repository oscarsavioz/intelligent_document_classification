# Intelligent document classification system
## Oscar Savioz
This project is about experimenting deep learning algorithms for the document image classification
task. The main goal is to have trained a multimodal classifier that use **visual** and **textual** data. The dataset used in the exeperiments is the famous **RVL-CDIP** and the
OCR files contained in it's parent dataset, **IIT-CDIP**.

> Models trained and their weights are not provided in this project. You should train them  on your own and store the models in a repository like Weights and Biases.

The **/src** folder contain all to source code and files used in this project. 

 - **/ocr** : contains all scripts about OCR analysis and spell checking tools. I experimented with 4 OCR-related tools :
    - Azure Cognitive Services 
    - EasyOCR
    - Tesseract 
    - Spell checking using NLTK 

 - **/experiments** : the main folder of this projet. It contains all the files used to run experiments. An experiment is a model (simple or hybrid) training and performance evaluation
 
    - Models training
    - Inference of DiT and Donut models
    - Tests of previously trained models 
    - Script to prepare textual data for RVL-CDIP
