# Intelligent document classification system
## Oscar Savioz
This project is about experimenting deep learning algorithms for the document image classification
task. The dataset used in the exeperiments is the famous **[RVL-CDIP](https://adamharley.com/rvl-cdip/)** and the
OCR files contained in it's parent dataset, **IIT-CDIP**. Several models were trained on this dataset with different configurations. 

The types of neural networks architectures experimented are mainly Convolutionnal Neural Networks (CNNs), which are specialzed architectures for processing digital images, and Transformers, which are really good at learning patterns of large sequential data as text, but also images. The global idea of this project is to train CNNs on the document images provided in the dataset with some pre-processing. In addition, the text extracted from the images with OCR tools are used to train Transformers-based architectures. The goal is to train the models with different combinations of hyper-parameters and achieve the best accuracy. Some experiments were done on multi-modal and hybrid architectures. 

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
