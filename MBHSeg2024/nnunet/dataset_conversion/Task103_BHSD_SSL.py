from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import os

if __name__ == '__main__':
    base = "/home/cansu/3D-CPS-master/data/nnUNet_raw_data_base/nnUNet_raw_data"
    task_name = "Task103_BHSDlabeledunalabeled"

    target_base = join("/home/cansu/3D-CPS-master/data/nnUNet_raw_data_base/nnUNet_raw_data", task_name)
    target_imagesTr = join(base, "Task103_BHSDlabeledunalabeled/imagesTr")
    target_labelsTr = join(base, "Task103_BHSDlabeledunalabeled/labelsTr")
    target_imagesTs = join(base, "Task103_BHSDlabeledunalabeled/imagesTs")
    target_uimagesTr = join(base, "Task103_BHSDlabeledunalabeled/unlabelTr")

    image_cases = subfiles(target_imagesTr, suffix='.nii.gz', join=False)
    label_cases = subfiles(target_labelsTr, suffix='.nii.gz', join=False)
    uimage_cases = subfiles(target_uimagesTr, suffix='.nii.gz', join=False)
    test_cases = subfiles(target_imagesTs, suffix='.nii.gz', join=False)

    json_dict = {}
    json_dict['name'] = "Task103_BHSDlabeledunalabeled"
    json_dict['description'] = "MBH-Seg: Multi-class Brain Hemorrhage Segmentation in Non-contrast CT - Multiclass"
    json_dict['tensorImageSize'] = "3D"
    json_dict['file_ending'] = ".nii.gz"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0" : "background",
        "1": "EDH",
        "2": "IPH",
        "3": "IVH",
        "4": "SAH",
        "5": "SDH",
    }

    # 没有把数据都存在nnUNet_raw_data里面, 把数据集的base写到dataset.json里面
    json_dict['imagesTrBase'] = target_imagesTr
    json_dict['labelsTrBase'] = target_labelsTr
    json_dict['imagesTsBase'] = target_imagesTs
    json_dict['uimagesTrBase'] = target_uimagesTr

    json_dict['numTraining'] = len(image_cases)
    json_dict['numTest'] = len(test_cases)
    json_dict['numUnlabeled'] = len(uimage_cases)

    json_dict['training'] = [{'image':join(target_imagesTr, image.split("_0000")[0] + ".nii.gz"),
                              "label":join(target_labelsTr, label)}
                             for image, label in zip(image_cases, label_cases)]
    json_dict['test'] = [join(target_imagesTs, c.split("_0000")[0] + ".nii.gz") for c in test_cases]
    json_dict['unlabeled'] = [join(target_uimagesTr, c.split("_0000.")[0] + ".nii.gz") for c in uimage_cases]

    save_json(json_dict, os.path.join(target_base, "dataset.json"))