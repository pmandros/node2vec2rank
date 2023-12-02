
import multiprocessing
import subprocess
import concurrent.futures


def run_command(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True)


if __name__ == '__main__':
    commands = [
   'nohup python sim_experiment_dims_six.py -num_nodes 50 -num_com 4 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 100 -num_com 4 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 500 -num_com 4 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 1000 -num_com 4 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 50 -num_com 6 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 100 -num_com 6 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 500 -num_com 6 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 1000 -num_com 6 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 50 -num_com 8 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 100 -num_com 8 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 500 -num_com 8 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 1000 -num_com 8 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 50 -num_com 10 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 100 -num_com 10 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 500 -num_com 10 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 1000 -num_com 10 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive True -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 50 -num_com 4 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 100 -num_com 4 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 500 -num_com 4 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 1000 -num_com 4 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 50 -num_com 6 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 100 -num_com 6 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 500 -num_com 6 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 1000 -num_com 6 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 50 -num_com 8 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 100 -num_com 8 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 500 -num_com 8 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 1000 -num_com 8 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 50 -num_com 10 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 100 -num_com 10 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 500 -num_com 10 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &',
'nohup python sim_experiment_dims_six.py -num_nodes 1000 -num_com 10 -num_samples 2000 -sample_uni False -noise_denominator 10 -degree_ground_truth False -degree_naive False -n2v2r_config ../configs/config_DBSM_dims.json &'
]

    max_concurrent = 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(run_command, command) for command in commands]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
