# libraries
import os 
import sys
import shutil
import json
import nibabel as nib
import SimpleITK as sitk

repo_path = os.getcwd()  
nnunet_path = repo_path + '/nnunet'
os.chdir(repo_path) 
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None
sys.path.insert(0,str(nnunet_path)) if str(nnunet_path) not in sys.path else None
# add repo_path/nnunet to PYTHONPATH
os.environ['nnUNet_raw_data_base'] = repo_path + '/data/nnUNet_raw_data_base'
os.environ['nnUNet_preprocessed'] = repo_path + '/data/nnUNet_preprocessed'
os.environ['RESULTS_FOLDER'] = repo_path + '/data/RESULTS_FOLDER'

print("Environment variables:")
print("nnUNet_raw_data_base:", os.environ.get("nnUNet_raw_data_base"))
print("nnUNet_preprocessed:", os.environ.get("nnUNet_preprocessed"))
print("RESULTS_FOLDER:", os.environ.get("RESULTS_FOLDER"))
print("python path:", os.environ.get("PYTHONPATH"))

from nnunet.inference.run_prediction import run_prediction

# functions

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rename_files_nnunet(path_second_phase):
    # Rename Secondphase test cases to nnunet format.  
    id = 100
    for file in os.listdir(path_second_phase):
        if file.endswith(".nii.gz"):
            print(file)
            id = id + 1
            new_name = "mbh_" + str(id) + "_0000.nii.gz"
            dict_test_names[file] = id
            shutil.copy(os.path.join(path_second_phase, file), os.path.join(new_path, new_name))
            print(new_name)

    with open(path_second_phase + "/dict_test_names.json", "w") as f:
        json.dump(dict_test_names, f)


def rescale_intensity(image, window_level=50, window_width=100):
    min_intensity = window_level - window_width / 2 # 50 - 100/2 = 0
    max_intensity = window_level + window_width / 2 # 50 + 100/2 = 100
    rescaled_image = sitk.IntensityWindowing(image, min_intensity, max_intensity)
    return rescaled_image


if __name__ == "__main__":
    repo_path = os.getcwd()
    # Renaming the files. 
    path_second_phase= repo_path + "/data/SecondStage_test"
    new_path = repo_path + "/data/SecondStage_test/named"
    dict_test_names = {}
    create_folder(new_path)
    rename_files_nnunet(path_second_phase)

    # Inference 
    input_folder=repo_path + "/data/SecondStage_test/named"
    output_folder = repo_path + "/mid_results"
    create_folder(output_folder)
    task_name = "103"
    model = "3d_fullres"
    trainer_class_name = "nnUNetTrainerV2_SSL"
    plans_identifier = "nnUNetPlansv2.1"
    folds = [0]
    chk = "model_best"

    # Call the function
    run_prediction(input_folder=input_folder, output_folder=output_folder, task_name=task_name, model=model, 
                trainer_class_name=trainer_class_name, plans_identifier=plans_identifier, folds=folds, chk=chk)


    print("PREDICTION DONE")
