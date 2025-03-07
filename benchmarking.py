import time
import sys
import os
import pickle
import numpy as np
from pathlib import Path

from TNCO_solver import TNCOsolver
from utils.wandb_utils import presetup_experiment
from utils.main_utils import read_data_file


def benchmark():
    config = presetup_experiment()
    input_dir = "datasets/tt-contr-eval"
    output_dir = "results/tt-contr-eval"
    for filepath in Path(input_dir).rglob('*'):
        if not filepath.is_file():
            continue
        config['network']['test_files'] = filepath
        train_files, eval_file = read_data_file(config)
        with open(eval_file, 'rb') as f:
            eqs, baseline_solutions, _ = pickle.load(f)
        if not isinstance(eqs, list):
            eqs = [eqs]
        operands = []
        # For each equation defined in the pickle file equation list
        for eq in eqs:
            # Create a list of empty tensors with the same shapes as the tensors used in this equation
            operands.append([np.empty(s) for s in eq[1]])
        tnco_solver = TNCOsolver(config)
        relative_path = filepath.relative_to(input_dir)
        filename = os.path.join(output_dir, relative_path)
        #filename = os.path.splitext(filename)[0] + '.result'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        path = tnco_solver.find_path(filename)

if __name__ == '__main__':
    benchmark()
