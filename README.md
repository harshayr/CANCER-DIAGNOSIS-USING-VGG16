 <h1>CANCER DIAGNOSIS</h1>

 ## ABOUT PROJECT

 Detecting and classifying lung cancer involves several challenges and onre of the chanllenge is

 Imaging Challenges:

* Variability in Images: Different types of lung cancers may present with variable appearances on medical    imaging (such as chest X-rays or CT scans). The variability in size, shape, and location of tumors can complicate accurate diagnosis.
* Overlapping Features: Radiological features of lung cancer may overlap with benign lesions or other lung diseases, making it challenging to differentiate between them.

This model will help doctors to overcome these challenges. It helps to diffrentiate between three most comman types of lung cancer 
* lung adenocarcinoma
* Lung Squamous Cell Carcinoma:
* Neuroendocrine Lung Cancer

Challenges:
* Image preprocessing and labelling was the most difficult as this is my 1st project so to overcome this problem i had explored various methods in the tensorflow library and finaly i came up with Imagegenrator and flow_from_dict methods

* Required more computational power and time consuming task. I have solved this problem by using callbacks method in tensorflow and also able to store losses and visual representation of validation losses and accuracy

Drawbacks:
* Only three types of lung cancer can be predicted
* Required tumors cell photo
* If you upload any picture still it will show any three of them type 

Features i hope to implement in the future:
* Model should detect all types of cancer type
* Diffrentiate Carcinogenic and Noncarcinogenic cell pictures

## How to Install and Run the Project

Step1: First clone the repository using
```sh
git clone https://github.com/harshayr/CANCER-TYPE-DETECTOR.git
```

Step2: go into current working directory 
```sh
cd CANCER-DIAGNOSIS
```

Step3: Install prerequisites by pasting below command to your terminal
```sh
pip install -r requirment.txt
```

Step4: Run streamlit file using terminal or command promt
```sh
streamlit run main.py
```
 If you want to create new enviroment with tensorflow-gpu you can follow below steps

Step1: Move yml file in your main directory

Step2: Deactivate base enviroment 
```sh
conda deactivate
```

Step3: Create new enviroment make sure yml file in your main directory and it will create new enviroment of name tensorflow as well as install all packages in yml file
```sh
conda env create -f tensorflow-apple-metal.yml -n tensorflow
```

Step4: Activate tensorflow enviroment 
```sh
conda acticate tensorflow
```


