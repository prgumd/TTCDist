###############################################################################
#
# File: ttc_depth_calc_error.py
# Available under MIT license
#
# Calculate the error for all sequences released with the TTCDist paper
#
# History:
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import argparse
import csv
import glob
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' # Attempt to disable OpenBLAS multithreading, it makes the script slower
import pickle

import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.io import savemat

from rpg_align_trajectory.rpg_align_trajectory import align_umeyama

def signal_to_array(signal):
    times = [t for t, d in signal]
    data  = [d for t, d in signal]

    times_array = np.array(times)
    data_array = np.vstack(data)

    return np.hstack((np.atleast_2d(times_array).transpose(), data_array))

def load_april(directory):
    results_name = os.path.join(directory, 'results.pickle')
    ttc_data = pickle.load(open(results_name, 'rb'))['results']

    if 'ground_truth_pose' in ttc_data.keys():
        april_signal = ttc_data['ground_truth_pose']
        april_data = signal_to_array(april_signal)
        return april_data
    else:
        return None

def load_ttc(directory):
    results_name = os.path.join(directory, 'results.pickle')
    ttc_data = pickle.load(open(results_name, 'rb'))['results']

    pose_signal = ttc_data['pose_hat']
    pose_data = signal_to_array(pose_signal)

    return pose_data

def load_phi(directory):
    results_name = os.path.join(directory, 'results.pickle')
    ttc_data = pickle.load(open(results_name, 'rb'))['results']

    pose_signal = ttc_data['phi_pose_hat']
    pose_data = signal_to_array(pose_signal)

    return pose_data

def load_vins_mono(directory):
    results_name = os.path.join(directory, 'results.pickle')
    vins_mono_data = pickle.load(open(results_name, 'rb'), encoding='latin1')

    vins_mono_data = vins_mono_data['poses']

    return vins_mono_data

def load_rovio(directory):
    results_name = os.path.join(directory, 'results.pickle')

    if not os.path.exists(results_name):
        return None

    rovio_data = pickle.load(open(results_name, 'rb'), encoding='latin1')

    rovio_data = rovio_data['poses']

    return rovio_data

def load_vicon(directory):
    results_name = os.path.join(directory, 'results.csv')

    # Unfortunately cannot use genfromtxt because vicon leaves out
    # delimiters when data is bad
    #vicon_data = np.genfromtxt(results_name, skip_header=5, delimiter=',', filling_values=np.nan, invalid_raise=False)

    vicon_data_list = []
    with open(results_name) as results_file:
        csv_reader = csv.reader(results_file, delimiter=',')
        for i in range(5):
            csv_reader.__next__()

        for row in csv_reader:
            if len(row) == 9:
                row_float = [float(x) for x in row]
                t = (row_float[0] - 1) * (1.0/200.0)
                vicon_data_list.append([t,] + row_float[6:9])

    vicon_data = np.array(vicon_data_list)
    vicon_data[:, 1:4] /= 1000.0
    return vicon_data

def plot_all_naive(data, abs=False):
    plt.figure()
    data_abs = {}
    for method in data.keys():
        if abs:
            data_abs[method] = np.abs(data[method])
        else:
            data_abs = data

        plt.subplot(3, 1, 1)

        t = data_abs[method][:, 0]

        x = data_abs[method][:, 1]
        plt.plot(t, x)
        plt.legend(data.keys())
        plt.grid(True)

        plt.subplot(3, 1, 2)
        y = data_abs[method][:, 2]
        plt.plot(t, y)
        plt.legend(data.keys())
        plt.grid(True)

        plt.subplot(3, 1, 3)
        z = data_abs[method][:, 3]
        plt.plot(t, z)
        plt.legend(data.keys())
        plt.grid(True)


def select_data(data, min_t, max_t):
    t = data[:, 0]
    data_selected = data[(t >= min_t) & (t <= max_t)]
    return data_selected

def calc_valid_path_length(data, ground_truth='vicon'):
    ground_truth_data = data[ground_truth]

    min_t = None
    max_t = None
    for method in data.keys():
        if method == ground_truth:
            continue
        data_times = data[method][:, 0]
        if min_t is None or data_times[0] > min_t:
            min_t = data_times[0]
        if max_t is None or data_times[-1] < max_t:
            max_t = data_times[-1]

    ground_truth_times = ground_truth_data[:, 0]
    ground_truth_trimmed = ground_truth_data[(ground_truth_times >= min_t) * (ground_truth_times <= max_t), :]

    dist_traveled_samples = np.linalg.norm(ground_truth_trimmed[1:, 1:4] - ground_truth_trimmed[:-1, 1:4], axis=1)
    path_length = np.sum(dist_traveled_samples)
    return path_length


def calc_error(data, ground_truth='april'):
    ground_truth_data = data[ground_truth]

    min_t = None
    max_t = None
    for method in data.keys():
        if method == ground_truth:
            continue
        data_times = data[method][:, 0]
        if min_t is None or data_times[0] > min_t:
            min_t = data_times[0]
        if max_t is None or data_times[-1] < max_t:
            max_t = data_times[-1]

    error_data = {}
    for method in data.keys():
        if method == ground_truth:
            continue

        data_times = data[method][:, 0]
        data_selected = select_data(data[method], min_t, max_t)

        data_interp = interp1d(data_selected[:, 0], data_selected[:, 1:4], axis=0)

        data_times_min = data_selected[:, 0].min()
        data_times_max = data_selected[:, 0].max()

        ground_truth_times = ground_truth_data[:, 0]
        ground_truth_trimmed = ground_truth_data[(ground_truth_times >= data_times_min) * (ground_truth_times <= data_times_max), :]

        data_resampled = data_interp(ground_truth_trimmed[:, 0])

        error = data_resampled - ground_truth_trimmed[:, 1:4]


        name = method + '-' + ground_truth
        error_data[name] = np.hstack((np.atleast_2d(ground_truth_trimmed[:, 0]).transpose(), error))

    return error_data

def calc_error_stats(error_data):
    error_stats = {}
    for method in error_data.keys():
        rmse = 100*np.linalg.norm(error_data[method][:, 1:4], axis=0) / (error_data[method].shape[0]**0.5)

        ate_samples = error_data[method][:, 1:4].flatten()
        ate = 100*np.linalg.norm(ate_samples) / (ate_samples.shape[0]**0.5)

        error_stats[method] = (rmse, ate)
    return error_stats

def print_error_stats(error_stats):
    for method in error_stats.keys():
        rmse = error_stats[method][0]
        ate  = error_stats[method][1]
        print('{:<40}: rmse {:0.3f} {:0.3f}  {:0.3f} (cm) ate: {:0.3f} (cm)'.format(method, rmse[0], rmse[1], rmse[2], ate))

def find_transform_time_sync_to_ground_truth(data, ground_truth='april'):
    ground_truth_data = data['vicon']

    ground_truth_data = data[ground_truth]

    min_t = None
    max_t = None
    for method in data.keys():
        if method == ground_truth:
            continue
        data_times = data[method][:, 0]
        if min_t is None or data_times[0] > min_t:
            min_t = data_times[0]
        if max_t is None or data_times[-1] < max_t:
            max_t = data_times[-1]

    transformation = {}
    min_ate = {}
    for method in data.keys():
        if method == ground_truth:
            continue

        data_selected = select_data(data[method], min_t, max_t)
        data_interp = interp1d(data_selected[:, 0], data_selected[:, 1:4], axis=0, bounds_error=False)

        ground_truth_length = ground_truth_data[-1, 0] - ground_truth_data[0, 0]
        if (ground_truth_length < (max_t - min_t)):
            raise Exception('ground truth must be longer than data for now')

        # TODO handle ground truth sample rate correctly
        ground_truth_samples = int(np.floor((max_t - min_t) * 200.0))

        for i in range(0, ground_truth_data.shape[0] - ground_truth_samples):
            if  ('record_000004' in recording
                and (method == 'ttc_wy_bias_2'
                     or method =='rovio')
                and i < 2000): # Hack to fix incorrect correlations that cause insane ATE errors for these two cases
                continue

            ground_truth_selected = ground_truth_data[i:i+ground_truth_samples, :]
            ground_truth_times_shifted = (ground_truth_selected[:, 0] - ground_truth_selected[0, 0]) + min_t
            valid = (ground_truth_times_shifted >= data_selected[0, 0]) & (ground_truth_times_shifted <= data_selected[-1, 0])
            ground_truth_times_shifted = ground_truth_times_shifted[valid]
            ground_truth_selected = ground_truth_selected[valid]

            data_resampled = data_interp(ground_truth_times_shifted)

            try:
                s, R, t = align_umeyama(ground_truth_selected[:, 1:4], data_resampled, known_scale=True)
                assert s == 1
            except np.linalg.LinAlgError:
                #print('error, should not happen')
                continue

            transformed = ((s * R) @ data_resampled.transpose()).transpose() + t

            error = (transformed - ground_truth_selected[:, 1:4]).flatten()
            ate = 100*np.linalg.norm(error) / (error.shape[0]**0.5)

            if method not in min_ate.keys() or ate < min_ate[method]:
                min_ate[method] = ate
                transformation[method] = (s, R, t, i / 200.0 - min_t)
                #print(method, i, i / 200.0, min_ate[method])

    return transformation

def transform_data_to_ground_truth(data, transformations, ground_truth='april'):
    data_transformed = {ground_truth: data[ground_truth]}
    for method in data.keys():
        if method == ground_truth:
            continue

        if method not in transformations:
            continue

        transformation = transformations[method]

        s = transformation[0]
        R = transformation[1]
        t = transformation[2]
        delta_t = transformation[3]

        data_times = data[method][:, 0] + delta_t

        transformed = ((s * R) @ data[method][:, 1:4].transpose()).transpose() + t
        data_transformed[method] = np.hstack((np.atleast_2d(data_times).transpose(), transformed))
    return data_transformed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory containing results files')
    parser.add_argument('--calc', action='store_true', help='Align trajectories with ground truth, save results to file')
    parser.add_argument('--latex', action='store_true', help='Process results file to latex')
    args = parser.parse_args()


    if args.calc:
        # Find all the recordings
        recordings = sorted(glob.glob(os.path.join(args.dir, 'record_*')))

        results = {}

        for recording in recordings:
            print('Processing: {}'.format(recording))
            # Find all the algorithms
            algorithms = sorted(glob.glob(os.path.join(recording, '*')))

            data = {}
            for algorithm in algorithms:
                method = os.path.basename(algorithm)

                #print(method)

                if method[:3] == 'ttc':
                    if method[3:] == '':
                        april_data = load_april(algorithm)
                        if april_data is not None:
                            data['april'] = april_data
                    data[method] = load_ttc(algorithm)
                    data['phi'] = load_phi(algorithm)
                elif method[:9] == 'vins_mono':
                    if 'record_000004' not in recording: # Skip recording where VINS-Mono did not work
                        data[method] = load_vins_mono(algorithm)
                elif method[:5] == 'rovio':
                    data[method] = load_rovio(algorithm)
                elif method[:5] == 'vicon':
                    data[method] = load_vicon(algorithm)
                elif method == 'visualization':
                    pass
                else:
                    raise Exception('unsupported method')

            transformation = find_transform_time_sync_to_ground_truth(data, ground_truth='vicon')
            transformed_data = transform_data_to_ground_truth(data, transformation, ground_truth='vicon')

            error_data = calc_error(transformed_data, ground_truth='vicon')

            #plot_all_naive(transformed_data)
            #plot_all_naive(error_data, abs=True)
            error_stats = calc_error_stats(error_data)
            print_error_stats(error_stats)

            for key in transformation.keys():
                print(transformation[key][3] - transformation[list(transformation.keys())[0]][3], key)

            recording_name = os.path.split(recording)[-1]            
            results[recording_name] = {'transformed_data': transformed_data,
                                       'error_data':       error_data}

        results_save_name = os.path.join(args.dir, 'processed_results.pickle')
        with open(results_save_name, 'wb') as file:
            pickle.dump(results, file)

        results_save_name = os.path.join(args.dir, 'processed_results.mat')
        savemat(results_save_name, mdict=results)

        plt.show()

    if args.latex:
        results_save_name = os.path.join(args.dir, 'processed_results.pickle')
        results_data = pickle.load(open(results_save_name, 'rb'))

        methods_to_error_name = {
            'AprilTag 3': 'april-vicon',
            'VINS-Mono': 'vins_mono-vicon', 
            'ROVIO': 'rovio-vicon', 

            '$\\Phi$-constraint (ours)': 'phi-vicon',
            '$\\tau$-constraint (ours)': 'ttc-vicon',
            #'$\\tau$-constraint (ours) (45 Hz LK output)': 'ttc_affine_skip_1-vicon',
            # '$\\tau$-constraint (ours) (30 Hz LK output)': 'ttc_affine_skip_2-vicon',
            # '$\\tau$-constraint (ours) (15 Hz LK output)': 'ttc_affine_skip_3-vicon',

            #'$\\tau$-constraint (ours) (0.5 deg/s $\\Omega_y$ bias)': 'ttc_wy_bias_0_5-vicon',
            # '$\\tau$-constraint (ours) (1.0 deg/s $\\Omega_y$ bias)': 'ttc_wy_bias_1-vicon',
            # '$\\tau$-constraint (ours) (2.0 deg/s $\\Omega_y$ bias)': 'ttc_wy_bias_2-vicon',
            #'$\\tau$-constraint (ours) (3.0 deg/s $\\Omega_y$ bias)': 'ttc_wy_bias_3-vicon',
            #'$\\tau$-constraint (ours) (4.0 deg/s $\\Omega_y$ bias)': 'ttc_wy_bias_4-vicon',

            #'$\\tau$-constraint (ours) (0.5 deg/s $\\Omega_z$ bias)': 'ttc_wz_bias_0_5-vicon',
            # '$\\tau$-constraint (ours) (1.0 deg/s $\\Omega_z$ bias)': 'ttc_wz_bias_1-vicon',
            # '$\\tau$-constraint (ours) (2.0 deg/s $\\Omega_z$ bias)': 'ttc_wz_bias_2-vicon',
            #'$\\tau$-constraint (ours) (3.0 deg/s $\\Omega_z$ bias)': 'ttc_wz_bias_3-vicon',
            #'$\\tau$-constraint (ours) (4.0 deg/s $\\Omega_z$ bias)': 'ttc_wz_bias_4-vicon',
        }

        latex_table_str  = '\\begin{table*}\n'
        latex_table_str += '  \\centering\n'
        latex_table_str += '  \\begin{tabular}{@{}lcccccccccc@{}}\n'
        latex_table_str += '    \\toprule\n'
        latex_table_str += ' & \\textit{Seq. 1} & \\textit{Seq. 2} & \\textit{Seq. 3} & \\textit{Seq. 4} & \\textit{Seq. 5} & \\textit{Seq. 6} & \\textit{Seq. 7} & \\textit{Seq. 8} & \\textit{Seq. 9} & \\textit{Seq. 10}\\\\\n'
        latex_table_str += '    \\midrule\n'

        line_str = 'Sequence Duration (s)'
        line2_str = 'Distance Traveled (m)'
        for recording in results_data.keys():
            #if recording != 'record_000004':
            #    continue

            error_data = results_data[recording]['error_data']
            dt = error_data['ttc-vicon'][-1, 0] - error_data['ttc-vicon'][0, 0]
            #dt2 = error_data['vins_mono-vicon'][-1, 0] - error_data['vins_mono-vicon'][0, 0]
            #print(dt, dt2)
            line_str += ' & {:1.2f}'.format(dt)

            transformed_data = results_data[recording]['transformed_data']
            path_length = calc_valid_path_length(transformed_data)
            line2_str += ' & {:1.2f}'.format(float(path_length))

        latex_table_str += line_str + '\\\\\n'
        latex_table_str += line2_str + '\\\\\n'

        latex_table_str += '    \\midrule\n'
        latex_table_str += '    Method & &&&& ATE (cm) &&&& \\\\\n'
        latex_table_str += '    \\midrule\n'

        for method in methods_to_error_name.keys():
            results_line = '{} &'.format(method)
            for recording in results_data.keys():
                #if recording != 'record_000004':
                #    continue


                transformed_data = results_data[recording]['transformed_data']
                error_data = results_data[recording]['error_data']

                error_stats = calc_error_stats(error_data)

                if methods_to_error_name[method] in error_stats:
                    rmse, ate = error_stats[methods_to_error_name[method]]
                    results_line += ' {:.2f} &'.format(ate)
                else:
                    results_line += ' - &'

            latex_table_str += results_line[:-1] + '\\\\\n'

        latex_table_str += '    \\bottomrule\n'
        latex_table_str += '  \\end{tabular}\n'
        latex_table_str += '  \\caption{All results in (cm).}\n'
        latex_table_str += '  \\label{tab:atecompare}\n'
        latex_table_str += '\\end{table*}'
        print(latex_table_str)

        error_data_total = {'ttc-vicon': [],
                            'phi-vicon': [],
                            'vins_mono-vicon': [],
                            'april-vicon': [],
                            'rovio-vicon': [],
                            # 'ttc_affine_skip_2-vicon': [],
                            # 'ttc_affine_skip_3-vicon': [],
                            # 'ttc_wy_bias_1-vicon': [],
                            # 'ttc_wy_bias_2-vicon': [],
                            # 'ttc_wz_bias_1-vicon': [],
                            # 'ttc_wz_bias_2-vicon': [],
                            }
        for recording in results_data.keys():
            error_data = results_data[recording]['error_data']
            for method in error_data.keys():
                if recording == 'record_000004' and method == 'vins_mono-vicon':  # Skip recording where VINS-Mono did not work
                    continue
                error_data_total[method].append(error_data[method])

        for key in error_data_total.keys():
            error_data_total[key] = np.vstack(error_data_total[key])

        error_stats = calc_error_stats(error_data_total)

        for key in error_data_total.keys():
            rmse, ate = error_stats[key]
            print(key)
            print(ate)
