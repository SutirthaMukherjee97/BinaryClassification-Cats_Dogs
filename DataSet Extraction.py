## Download From Kaggle

!pip install -q kaggle

from google.colab import files
files.upload()

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

#! kaggle datasets list

!kaggle competitions download -c dogs-vs-cats
#! mkdir dogs cat classifier

import zipfile
with zipfile.ZipFile("/content/dogs-vs-cats.zip","r") as zip_ref:
   zip_ref.extractall("")

with zipfile.ZipFile("/content/train.zip","r") as zip_ref:
  zip_ref.extractall("")
with zipfile.ZipFile("/content/test1.zip","r") as zip_ref:
  zip_ref.extractall("")

print(len(os.listdir('/content/test1'))) #test dataset
# print(len(os.listdir('/content/test_set/test_set/cats'))) #cats
# print(len(os.listdir('/content/test_set/test_set/dogs'))) #dogs
print(len(os.listdir('/content/train'))) #train dataset
# print(len(os.listdir('/content/training_set/training_set/cats'))) #cats
# print(len(os.listdir('/content/training_set/training_set/dogs'))) #dogs
