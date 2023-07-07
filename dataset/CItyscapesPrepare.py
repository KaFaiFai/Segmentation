from cityscapesscripts.preparation import createTrainIdLabelImgs, createTrainIdInstanceImgs, createPanopticImgs
import os
from pathlib import Path
from dotenv import load_dotenv

def prepare():
    load_dotenv()
    print(os.environ['CITYSCAPES_DATASET'])
    createTrainIdLabelImgs.main()
    createTrainIdInstanceImgs.main()
    createPanopticImgs.convert2panoptic()

if __name__ == "__main__":
    prepare()