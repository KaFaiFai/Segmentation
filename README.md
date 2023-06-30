# Setup
1. go to https://www.cityscapes-dataset.com/downloads/ and download the `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip`
2. unzip them to `./training_data`
2. run the following commands to generate panoptic images
```bash
cd dataset
git clone https://github.com/mcordts/cityscapesScripts.git
cd ..
python dataset/cityscapesScripts/cityscapesscripts/preparation/createPanopticImgs.py --dataset-folder ./training_data/gtFine --output-folder ./training_data/panoptic
python dataset/cityscapesScripts/cityscapesscripts/preparation/createTrainIdInstanceImgs.py
python dataset/cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py
```