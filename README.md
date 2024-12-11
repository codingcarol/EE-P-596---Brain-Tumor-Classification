# EE-P-596---Brain-Tumor-Classification
# Project Overview: A brief description of the project’s purpose and goals.
This project uses the Brain Tumor MRI Dataset from Kaggle (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data) to classify four types of brain tumors (Glioma, Meningioma, Pituitary, No Tumor).   
The goal is to use a CNN to classify the brain tumors with > 90% accuracy. 

# Setup Instructions
Install without requirements.txt
```
conda create --name env_name  
conda activate env_name  
conda install jupyter  
conda install matplotlib numpy scikit-learn seaborn tqdm  
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia  
jupyter notebook
```

Install with requirements.txt 
```
conda create --name env_name --file requirements.txt
conda activate env_name
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
jupyter notebook
```

Create a 'data' folder. Download the dataset from Kaggle and put it in your data folder.  
Download the model from Google Drive (link below) and put it in the directory

The file structure should look like 
```
├── README.md
├── requirements.txt
├── model.pth
├── src/
│   ├── main.py      # Entry point of the program
│   ├── model.py     # Model definition
├── data/
│   ├── Training/
│      ├── glioma    
│      ├── meningioma
│      ├── notumor
│      ├── pituitary
│   ├── Testing/
│      ├── glioma    
│      ├── meningioma
│      ├── notumor
│      ├── pituitary       
├── demo/            
│      ├── brain_tumor_classification_pretrained_128x128
└── results/         # Placeholder for generated results
```

# How to Run 
Once jupyter notebook is opened, run the ipynb file. Everything should run correctly if the folder structure is followed. 
The expected output should be two types of results - (1) the overall testing data accuracy, and (2) the testing data accuracy for each class. 

# Pre-trained Model Link
https://drive.google.com/file/d/1rYeeD69CvXJp448U0I4UiwJp78Ilof7m/view?usp=drive_link

# Acknowledgments
Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data  
Codebase: partially based off Lab 4 homework assignment from EE P 596
