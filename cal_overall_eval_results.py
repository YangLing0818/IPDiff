import os
import numpy as np
import torch
from glob import glob
from utils import misc
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_result_path', default='./eval_results', type=str)
    parser.add_argument('--docking_mode', type=str, default='vina_dock', choices=['none', 'vina_score', 'vina_dock'])
    args = parser.parse_args()

    eval_path = args.eval_result_path
    docking_mode = args.docking_mode
    results_fn_list = glob(os.path.join(eval_path, 'metrics_*.pt'))

    logger = misc.get_logger('evaluation_overall', log_dir=eval_path)

    logger.info(f'the num of results.pt is {len(results_fn_list)}.')

    qed_all = []
    sa_all = []
    vina_score_all = []
    vina_min_all = []
    vina_dock_all = []

    for rfn in results_fn_list:
        result_i = torch.load(rfn)['all_results']
        qed_all += [r['chem_results']['qed'] for r in result_i]
        sa_all += [r['chem_results']['sa'] for r in result_i]

        if docking_mode in ['vina_dock', 'vina_score']:
            vina_score_all += [r['vina']['score_only'][0]['affinity'] for r in result_i]
            vina_min_all += [r['vina']['minimize'][0]['affinity'] for r in result_i]
            if docking_mode == 'vina_dock':
                vina_dock_all += [r['vina']['dock'][0]['affinity'] for r in result_i]

    qed_all_mean, qed_all_median = np.mean(qed_all), np.median(qed_all)
    sa_all_mean, sa_all_median = np.mean(sa_all), np.median(sa_all)

    logger.info('QED:   Mean: %.3f Median: %.3f' % (qed_all_mean, qed_all_median))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (sa_all_mean, sa_all_median))

    if len(vina_score_all):
        vina_score_all_mean, vina_score_all_median = np.mean(vina_score_all), np.median(vina_score_all)
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (vina_score_all_mean, vina_score_all_median))

    if len(vina_min_all):
        vina_min_all_mean, vina_min_all_median = np.mean(vina_min_all), np.median(vina_min_all)
        logger.info('Vina Min:  Mean: %.3f Median: %.3f' % (vina_min_all_mean, vina_min_all_median))

    if len(vina_dock_all):
        vina_dock_all_mean, vina_dock_all_median = np.mean(vina_dock_all), np.median(vina_dock_all)
        logger.info('Vina Dock:  Mean: %.3f Median: %.3f' % (vina_dock_all_mean, vina_dock_all_median))