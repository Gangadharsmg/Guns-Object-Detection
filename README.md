# Guns-Object-Detection
Model to detect the gun objects in an image



### Clone the Mask R-CNN GitHub Repository
git clone https://github.com/matterport/Mask_RCNN.git



### Install the Mask R-CNN Library
cd Mask_RCNN

python setup.py install



### Download the dataset
https://www.kaggle.com/issaisasank/guns-object-detection

Create directory named **gun**.

Once the dataset is downloaded, keep both folders (Images, Labels) under **gun** directory



### Install all the requirements
pip install -r requirements.txt

Just make sure all the relative paths are correct in your system.



### Run the python file to train and evaluate the model
python gun_object_detection.py
