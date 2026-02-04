import torch
import os
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

def run_prediction(input_folder, output_folder, task_name, model="3d_fullres", trainer_class_name=default_trainer, 
                   cascade_trainer_class_name=default_cascade_trainer, folds='None', save_npz=False, 
                   lowres_segmentations=None, num_threads_preprocessing=6, num_threads_nifti_save=2, disable_tta=False, 
                   step_size=0.5, overwrite_existing=False, mode="normal", all_in_gpu=None, plans_identifier=default_plans_identifier, 
                   chk='model_final_checkpoint', fixed_spacing=None, part_id=0, num_parts=1, 
                   disable_pp=False, gpu=1):

    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres"

    if lowres_segmentations == "None":
        lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    assert all_in_gpu in [None, False, True]

    # Handle cascade scenario
    if model == "3d_cascade_fullres" and lowres_segmentations is None:
        print("lowres_segmentations is None. Attempting to predict 3d_lowres first...")
        assert part_id == 0 and num_parts == 1, "if you don't specify a --lowres_segmentations folder for the cascade inference"
        model_folder_name = join(network_training_output_dir, "3d_lowres", task_name, trainer_class_name + "__" + plans_identifier)
        assert isdir(model_folder_name), f"model output folder not found: {model_folder_name}"
        lowres_output_folder = join(output_folder, "3d_lowres_predictions")
        predict_from_folder(model_folder_name, input_folder, lowres_output_folder, folds, False, num_threads_preprocessing, 
                            num_threads_nifti_save, None, part_id, num_parts, not disable_tta, overwrite_existing, mode, 
                            all_in_gpu, step_size=step_size, disable_postprocessing=disable_pp)
        lowres_segmentations = lowres_output_folder
        torch.cuda.empty_cache()
        print("3d_lowres done")

    if model == "3d_cascade_fullres":
        trainer = cascade_trainer_class_name
    else:
        trainer = trainer_class_name


    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" + plans_identifier)
    print(f"using model stored in {model_folder_name}")
    assert isdir(model_folder_name), f"model output folder not found: {model_folder_name}"

    inference_suffix = f"{task_name}_{plans_identifier}_{trainer_class_name}_fold{folds}_{model}_{mode}"
    output_folder = join(output_folder, inference_suffix)

    predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing, 
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta, overwrite_existing, 
                        all_in_gpu, step_size=step_size, checkpoint_name=chk)


# Example call to the function:
# run_prediction(input_folder="/path/to/input", output_folder="/path/to/output", task_name="Task001")

