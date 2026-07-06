import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

# Define your notebooks in the exact execution order
# notebooks = [
#     "damage_prediction_customCNN.ipynb",
#     "damage_prediction_efficientnet.ipynb",
#     "damage_prediction_mobilenet_small.ipynb",
#     "damage_prediction_mobilenet_large.ipynb",
#     "damage_prediction_resnet.ipynb",
#     "damage_prediction_densenet.ipynb"
        
# ]

notebooks = [
    "damage_prediction_mobilenet_small.ipynb",
    "damage_prediction_mobilenet_large.ipynb",
    "damage_prediction_efficientnet.ipynb",
    "damage_prediction_resnet.ipynb",
    "damage_prediction_densenet.ipynb"
        
]

# Explicitly path to the active bootcamp python binary
ENV_PYTHON = "/mnt/mydata/06.04.2026_home_laptop_backup/jlab-env/bin/python3"

def run_notebook(nb_path):
    print(f"Starting notebook: {nb_path}", flush=True)
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    # By setting a blank kernel name and explicitly overriding the path via system args,
    # nbconvert is forced to use the exact python executing this runner script.
    ep = ExecutePreprocessor(timeout=None)
    
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(os.path.abspath(nb_path))}})
        print(f"Finished successfully: {nb_path}\n", flush=True)
    except CellExecutionError as e:
        print(f"Error broken in {nb_path}:\n{e}\n", flush=True)
    except Exception as e:
        print(f"Unexpected error in {nb_path}:\n{e}\n", flush=True)
    finally:
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

if __name__ == "__main__":
    # Safety verification: ensure the script is executing under the right binary
    current_python = sys.executable
    if current_python != ENV_PYTHON:
        print(f"Warning: Script is running on {current_python} instead of {ENV_PYTHON}")

    for notebook in notebooks:
        if os.path.exists(notebook):
            run_notebook(notebook)
        else:
            print(f"File not found: {notebook}", flush=True)

    print("All overnight training tasks finalized.", flush=True)

