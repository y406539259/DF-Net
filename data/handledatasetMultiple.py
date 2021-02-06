import os

DRIVE_train_txt = "/home/yinpengshuai/FrangiNetNew/data/DRIVE/DRIVEtraining.txt"
DRIVE_test_txt = "/home/yinpengshuai/FrangiNetNew/data/DRIVE/DRIVEtesting.txt"
CHASEDB1_train_txt = "/home/yinpengshuai/FrangiNetNew/data/CHASEDB1/CHASEDB1training.txt"
CHASEDB1_test_txt = "/home/yinpengshuai/FrangiNetNew/data/CHASEDB1/CHASEDB1testing.txt"
STARE_txt = "/home/yinpengshuai/FrangiNetNew/data/Stare/StaretrainingAll.txt"

DRIVE_IMAGE = "/home/yinpengshuai/FrangiNetNew/data/DRIVE/images"
DRIVE_MASK = "/home/yinpengshuai/FrangiNetNew/data/DRIVE/masks"
DRIVE_LABEL = "/home/yinpengshuai/FrangiNetNew/data/DRIVE/label"

CHASEDB1_IMAGE = "/home/yinpengshuai/FrangiNetNew/data/CHASEDB1/images"
CHASEDB1_MASK = "/home/yinpengshuai/FrangiNetNew/data/CHASEDB1/Masks"
CHASEDB1_LABEL = "/home/yinpengshuai/FrangiNetNew/data/CHASEDB1/label"

STARE_IMAGE = "/home/yinpengshuai/FrangiNetNew/data/Stare/images"
STARE_MASK = "/home/yinpengshuai/FrangiNetNew/data/Stare/Masks"
STARE_LABEL = "/home/yinpengshuai/FrangiNetNew/data/Stare/labels"

def GenerateFullPath(train_test ,dataset):
    ImageList = []
    LabelList = []
    MaskList = []

    if dataset=="DRIVE":

        if train_test == "train":
            DRIVE_train_file = open(DRIVE_train_txt,'r')
            alltrainfile = DRIVE_train_file.readlines()

            for img_name in alltrainfile:
                label_name = img_name.rstrip('\n')[:-12] + 'manual1.gif'
                mask_name = img_name.rstrip('\n')[:-12] + 'training_mask.gif'
                img_name = img_name.rstrip('\n')

                Full_img_name = os.path.join(DRIVE_IMAGE, img_name)
                Full_label_name = os.path.join(DRIVE_LABEL, label_name)
                Full_mask_name = os.path.join(DRIVE_MASK, mask_name)

                ImageList.append(Full_img_name)
                LabelList.append(Full_label_name)
                MaskList.append(Full_mask_name)

        if train_test == "test":
            DRIVE_test_file = open(DRIVE_test_txt, 'r')
            alltestfile = DRIVE_test_file.readlines()

            for img_name in alltestfile:
                label_name = img_name.rstrip('\n')[:-8] + 'manual1.gif'
                mask_name = img_name.rstrip('\n')[:-8] + 'test_mask.gif'
                img_name = img_name.rstrip('\n')

                Full_img_name = os.path.join(DRIVE_IMAGE, img_name)
                Full_label_name = os.path.join(DRIVE_LABEL, label_name)
                Full_mask_name = os.path.join(DRIVE_MASK, mask_name)

                ImageList.append(Full_img_name)
                LabelList.append(Full_label_name)
                MaskList.append(Full_mask_name)

    if dataset=="CHASEDB1":

        if train_test == "train":
            CHASEDB1_train_file = open(CHASEDB1_train_txt,'r')
            alltrainfile = CHASEDB1_train_file.readlines()

            for img_name in alltrainfile:
                label_name = img_name.rstrip('\n')[:-4] + '_1stHO.png'
                mask_name = img_name.rstrip('\n')
                img_name = img_name.rstrip('\n')

                Full_img_name = os.path.join(CHASEDB1_IMAGE, img_name)
                Full_label_name = os.path.join(CHASEDB1_LABEL, label_name)
                Full_mask_name = os.path.join(CHASEDB1_MASK, mask_name)

                ImageList.append(Full_img_name)
                LabelList.append(Full_label_name)
                MaskList.append(Full_mask_name)

        if train_test == "test":
            CHASEDB1_test_file = open(CHASEDB1_test_txt, 'r')
            alltestfile = CHASEDB1_test_file.readlines()

            for img_name in alltestfile:
                label_name = img_name.rstrip('\n')[:-4] + '_1stHO.png'
                mask_name = img_name.rstrip('\n')
                img_name = img_name.rstrip('\n')

                Full_img_name = os.path.join(CHASEDB1_IMAGE, img_name)
                Full_label_name = os.path.join(CHASEDB1_LABEL, label_name)
                Full_mask_name = os.path.join(CHASEDB1_MASK, mask_name)

                ImageList.append(Full_img_name)
                LabelList.append(Full_label_name)
                MaskList.append(Full_mask_name)

    if dataset=="STARE":

        DRIVE_file = open(STARE_txt,'r')
        allfile = DRIVE_file.readlines()

        for img_name in allfile:
            label_name = img_name.rstrip('\n')[:-4] + '.ah.ppm'
            mask_name = img_name.rstrip('\n')[:-4] + '.jpg'
            img_name = img_name.rstrip('\n')

            Full_img_name = os.path.join(STARE_IMAGE, img_name)
            Full_label_name = os.path.join(STARE_LABEL, label_name)
            Full_mask_name = os.path.join(STARE_MASK, mask_name)

            ImageList.append(Full_img_name)
            LabelList.append(Full_label_name)
            MaskList.append(Full_mask_name)

    return ImageList, LabelList, MaskList

ImageListDriveTrain, LabelListDriveTrain, MaskListDriveTrain = GenerateFullPath("train", "DRIVE")
ImageListDriveTest, LabelListDriveTest, MaskListDriveTest = GenerateFullPath("test", "DRIVE")
ImageListChaseDB1Train, LabelListChaseDB1Train, MaskListChaseDB1Train = GenerateFullPath("train", "CHASEDB1")
ImageListChaseDB1Test, LabelListChaseDB1Test, MaskListChaseDB1Test = GenerateFullPath("test", "CHASEDB1")
ImageListStare, LabelListStare, MaskListStare = GenerateFullPath("train", "STARE")

TrainImageList = []
TrainLabelList = []
TrainMaskList = []

TestImageList = []
TestLabelList = []
TestMaskList = []

TrainImageList += ImageListDriveTrain
TrainImageList += ImageListChaseDB1Train
TrainImageList += ImageListChaseDB1Test
TrainImageList += ImageListStare

TrainLabelList += LabelListDriveTrain
TrainLabelList += LabelListChaseDB1Train
TrainLabelList += LabelListChaseDB1Test
TrainLabelList += LabelListStare

TrainMaskList += MaskListDriveTrain
TrainMaskList += MaskListChaseDB1Train
TrainMaskList += MaskListChaseDB1Test
TrainMaskList += MaskListStare

TestImageList += ImageListDriveTest
TestLabelList += LabelListDriveTest
TestMaskList += MaskListDriveTest

RootDir = "/home/yinpengshuai/FrangiNetNew/data/"
Dataset = "CHASEDB1"

TrainImageTxT_ = "Train_" + Dataset + "_Image.txt"
TrainLabelTxT_ = "Train_" + Dataset + "_Label.txt"
TrainMaskTxT_ = "Train_" + Dataset + "_Mask.txt"

TestImageTxT_ = "Test_" + Dataset + "_Image.txt"
TestLabelTxT_ = "Test_" + Dataset + "_Label.txt"
TestMaskTxT_ = "Test_" + Dataset + "_Mask.txt"

TrainImageTxTDir = os.path.join(RootDir, TrainImageTxT_)
TrainLabelTxTDir = os.path.join(RootDir, TrainLabelTxT_)
TrainMaskTxTDir = os.path.join(RootDir, TrainMaskTxT_)

TestImageTxTDir = os.path.join(RootDir, TestImageTxT_)
TestLabelTxTDir = os.path.join(RootDir, TestLabelTxT_)
TestMaskTxTDir = os.path.join(RootDir, TestMaskTxT_)

def WriteList(List, filename):
    with open(filename, "w") as f:
        for EachImage in List:
            f.write(EachImage)
            f.write('\n')

WriteList(TrainImageList, TrainImageTxTDir)
WriteList(TrainLabelList, TrainLabelTxTDir)
WriteList(TrainMaskList, TrainMaskTxTDir)

WriteList(TestImageList, TestImageTxTDir)
WriteList(TestLabelList, TestLabelTxTDir)
WriteList(TestMaskList, TestMaskTxTDir)
