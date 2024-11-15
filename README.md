# GS-DTA
Intergrating Graph and Sequence Models for Drug-traget binding affinity.

Requirements

+ python 3.7.12, pytorch 1.13.0 , numpy 1.21.6, pandas 1.3.5,  , torch_genometrci 2.3.1 , scipy 1.7.3

1.Build your virtual environment
 ```
# create 
conda create -n  GSDTA python=3.7
# activate 
conda activate GSDTA
```
 2.Install packages
+ After activating the environment , you need to install the required packages
```
# install
conda install numpy, scipy, pandas, torch, torch_genometric
```
3.Clone GSDTA
+ After installing the required packages , you need to download GS-DTA from github:
```
git clone https://github.com/zhuziguang/GSDTA.git
```
Tested data(Davis and KIBA)

The example data can be downloaded form   https://github.com/thinng/GraphDTA/tree/master/data()

Usage
Train Model

1.Create Dataset  in pytorch format.
```
python create_data.py
```

2.Train model
Run the following command to train the model.
```
conda activate GSDTA
cd GSDTA
python training.py 0 0
```
Where the first argument is for the  index of the datasets,0/1 for 'Davis' or 'KIBA'.Respectively,the second argument is for the index of the cuda.Be careful about the cuda you choose.
This will return the model and results file that achieves the best MSE on the test data throughout the training process.

3.Train a prediction model with validation
Run the following conmmand to test the model.
```
python training_validation.py 0 0
```
Where the first argument is for the  index of the datasets,0/1 for 'Davis' or 'KIBA'.Respectively,the second argument is for the index of the cuda.Be careful about the cuda you choose.
This will returns the model that achieved the best MSE on the validation data throughout the training process,as well as the model's performance on the test data.

The model weights are obtained from https://www.dropbox.com/scl/fo/73l2jobpp2h8mmhb3ejan/AJGmt7H3ErbOpphHfhabbg8?rlkey=icbnrnf6nf1thpl97d59y5zsu&st=97acgxng&dl=0

