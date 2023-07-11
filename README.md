# Setup
1. go to https://www.cityscapes-dataset.com/downloads/ and download the `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip`
2. unzip them to `./training_data`
3. run the following commands to generate panoptic images
```bash
python -m pip install cityscapesscripts
cp sample.env .env
# modify .env to your environment
python dataset/CityscapesPrepare.py
```