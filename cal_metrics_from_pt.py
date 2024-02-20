import os
import numpy as np
import torch
from glob import glob


if __name__ == '__main__':
    eval_path = '/home/huangzl/workspace2/IPDiff-gspbapv5comp-n05/eval_results/'
    results_fn_list = glob(os.path.join(eval_path, 'metrics_*.pt'))
    print("num of results.pt: ",  len(results_fn_list))
    docking_mode = 'vina_score'

    qed_all = []
    sa_all = []
    qvina_all = []
    vina_score_all = []
    vina_min_all = []
    vina_dock_all = []

    for rfn in results_fn_list:
        result_i = torch.load(rfn)['all_results']
        qed_all += [r['chem_results']['qed'] for r in result_i]
        sa_all += [r['chem_results']['sa'] for r in result_i]
        if docking_mode == 'qvina':
            qvina_all += [r['vina'][0]['affinity'] for r in result_i]
        elif docking_mode in ['vina_dock', 'vina_score']:
            vina_score_all += [r['vina']['score_only'][0]['affinity'] for r in result_i]
            vina_min_all += [r['vina']['minimize'][0]['affinity'] for r in result_i]
            if docking_mode == 'vina_dock':
                vina_dock_all += [r['vina']['dock'][0]['affinity'] for r in result_i]

    qed_all_mean, qed_all_median = np.mean(qed_all), np.median(qed_all)
    sa_all_mean, sa_all_median = np.mean(sa_all), np.median(sa_all)

    print("qed_all_mean, qed_all_median:", qed_all_mean, qed_all_median)
    print("sa_all_mean, sa_all_median:", sa_all_mean, sa_all_median)

    if len(qvina_all):
        qvina_all_mean, qvina_all_median = np.mean(qvina_all), np.median(qvina_all)
        print("qvina_all_mean, qvina_all_median:", qvina_all_mean, qvina_all_median)

    if len(vina_score_all):
        vina_score_all_mean, vina_score_all_median = np.mean(vina_score_all), np.median(vina_score_all)
        print("vina_score_all_mean, vina_score_all_median:", vina_score_all_mean, vina_score_all_median)

    if len(vina_min_all):
        vina_min_all_mean, vina_min_all_median = np.mean(vina_min_all), np.median(vina_min_all)
        print("vina_min_all_mean, vina_min_all_median:", vina_min_all_mean, vina_min_all_median)

    if len(vina_dock_all):
        vina_dock_all_mean, vina_dock_all_median = np.mean(vina_dock_all), np.median(vina_dock_all)
        print("qvina_all_mean, qvina_all_median:" , vina_dock_all_mean, vina_dock_all_median)
