import os
import shutil
import json
import SimpleITK as sitk

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rescale_intensity(image, window_level=50, window_width=100):
    min_intensity = window_level - window_width / 2 # 50 - 100/2 = 0
    max_intensity = window_level + window_width / 2 # 50 + 100/2 = 100
    rescaled_image = sitk.IntensityWindowing(image, min_intensity, max_intensity)
    return rescaled_image

def postprocess(images_path, segmentations_path, post_processed_folder):
    for file in os.listdir(images_path):
        if file.endswith('.nii.gz'):
            id = file.split('.')[0].split('_')[1]
            name = "mbh_" + id + ".nii.gz"
            print(name)
            itk_img_segmentation = sitk.ReadImage(segmentations_path + "/" + "mbh_" + id + ".nii.gz")

            itk_img = sitk.ReadImage(images_path + "/" + "mbh_" + id + "_0000.nii.gz")
            rescaled_image = rescale_intensity(itk_img, window_level=50, window_width=100)
            seg_array_org = sitk.GetArrayFromImage(itk_img_segmentation)
            seg_array = sitk.GetArrayFromImage(itk_img_segmentation)
            img_array = sitk.GetArrayFromImage(rescaled_image)
            
            # Check where img_array is 0 and seg_array is not 0
            mask = (img_array == 0) & (seg_array != 0)
            
            # Set those positions in seg_array to 0
            seg_array[mask] = 0

            # save the new segmentation as nii.gz with eth same configurations as the original one. 
            new_seg = sitk.GetImageFromArray(seg_array)
            new_seg.CopyInformation(itk_img_segmentation)
            sitk.WriteImage(new_seg, post_processed_folder + "/" + "mbh_" + id + ".nii.gz")


def rename_to_orig(names_ids_dict_test, out_folder):
    for file in os.listdir(post_processed_folder):
        if file.endswith('.nii.gz'):
            print(file)
            id_number = file.split('.')[0].split('_')[1]
            id_number = int(id_number)
            print(id_number)
            for key, value in names_ids_dict_test.items():
                if value == id_number:
                    name = key
            new_name = f"{name}"
            shutil.copy(post_processed_folder + "/" + file, out_folder + "/" + new_name)

if __name__ == "__main__":
    repo_path = os.getcwd()

    task_name = "103"
    model = "3d_fullres"
    trainer_class_name = "nnUNetTrainerV2_SSL"
    plans_identifier = "nnUNetPlansv2.1"
    folds = [0]
    chk = "model_best"

    # Postprocessing
    Task_name_full = f"Task{task_name}_BHSDlabeledunalabeled"
    images_path = repo_path + "/data/SecondStage_test/named"
    segmentations_path = repo_path + f"/mid_results/{Task_name_full}_{plans_identifier}_{trainer_class_name}_fold{folds}_{model}_normal"
    post_processed_folder = segmentations_path + "/post_processed"
    path_second_phase= repo_path + "/data/SecondStage_test"
    
    create_folder(post_processed_folder)
    
    postprocess(images_path, segmentations_path, post_processed_folder)

    with open(path_second_phase + "/dict_test_names.json", 'r') as f:
         names_ids_dict_test = json.load(f)

    results_folder = repo_path + "/inference_results"
    create_folder(results_folder)

    rename_to_orig(names_ids_dict_test, results_folder)