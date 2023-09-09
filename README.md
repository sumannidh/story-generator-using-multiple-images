# VisualStorytelling

### Dependencies
Python 3.6 or 2.7<br>
[Pytorch](https://pytorch.org) >= 1.0.0
<br>
### Prerequisites
##### 1. Clone the repository
```
git clone https://github.com/ShobhitDubey0729/multi-image-cued-storytelling.git
```
##### 2. Download requirements
```
pip3 install -r requirements.txt
```{.python}
python3
>>> import nltk
>>> nltk.download('punkt')
>>> exit()
```

<br>

### Preprocessing

##### 1. Download the dataset
[VIST homepage](http://visionandlanguage.net/VIST/dataset.html)

##### 2. Resize images and build vocabulary
All the images should be resized to 256x256.
```
python3 resize.py --image_dir [train_image_dir] --output_dir [output_train_dir]
python3 resize.py --image_dir [val_image_dir] --output_dir [output_val_dir]
python3 resize.py --image_dir [test_image_dir] --output_dir [output_test_dir]
python3 preprocessing.py
```

<br>

### Training & Validation for Seq2seq

```
python3 train_seq2seq.py
```
### Training & Validation for Global attention
```
python3 train.py
```


<br>

### Evaluation

##### 1. Download the [evaluation tool (METEOR score)](https://github.com/windx0303/VIST-Challenge-NAACL-2018) for the VIST Challenge
```
git clone https://github.com/windx0303/VIST-Challenge-NAACL-2018 ../VIST-Challenge-NAACL-2018
```

##### 2. Install Java
```
sudo apt install default-jdk
```

##### 3. Run eval.py script
```
python3 eval.py --model_num [my_model_num]
```
The result.json file will be found in the root directory.


Reference-
The script is based on https://github.com/tkim-snu/GLACNet
