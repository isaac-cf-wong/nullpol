#!/usr/bin/env bash

# scalar_tensor_injection_data0_0_generation
# PARENTS 
# CHILDREN scalar_tensor_injection_data0_0_analysis_H1L1V1
if [[ "scalar_tensor_injection_data0_0_generation" == *"$1"* ]]; then
    echo "Running: /home/chun-fung.wong/.conda/envs/nullpol-3.10/bin/nullpol_pipe_generation outdir_scalar_tensor_injection/scalar_tensor_injection_config_complete.ini --submit --label scalar_tensor_injection_data0_0_generation --idx 0 --trigger-time 0 --injection-file injections.json --outdir outdir_scalar_tensor_injection"
    /home/chun-fung.wong/.conda/envs/nullpol-3.10/bin/nullpol_pipe_generation outdir_scalar_tensor_injection/scalar_tensor_injection_config_complete.ini --submit --label scalar_tensor_injection_data0_0_generation --idx 0 --trigger-time 0 --injection-file injections.json --outdir outdir_scalar_tensor_injection
fi

# scalar_tensor_injection_data0_0_analysis_H1L1V1
# PARENTS scalar_tensor_injection_data0_0_generation
# CHILDREN scalar_tensor_injection_data0_0_analysis_H1L1V1_final_result
if [[ "scalar_tensor_injection_data0_0_analysis_H1L1V1" == *"$1"* ]]; then
    echo "Running: /home/chun-fung.wong/.conda/envs/nullpol-3.10/bin/nullpol_pipe_analysis outdir_scalar_tensor_injection/scalar_tensor_injection_config_complete.ini --submit --outdir outdir_scalar_tensor_injection --detectors H1 --detectors L1 --detectors V1 --label scalar_tensor_injection_data0_0_analysis_H1L1V1 --data-dump-file outdir_scalar_tensor_injection/data/scalar_tensor_injection_data0_0_generation_data_dump.pickle --sampler dynesty"
    /home/chun-fung.wong/.conda/envs/nullpol-3.10/bin/nullpol_pipe_analysis outdir_scalar_tensor_injection/scalar_tensor_injection_config_complete.ini --submit --outdir outdir_scalar_tensor_injection --detectors H1 --detectors L1 --detectors V1 --label scalar_tensor_injection_data0_0_analysis_H1L1V1 --data-dump-file outdir_scalar_tensor_injection/data/scalar_tensor_injection_data0_0_generation_data_dump.pickle --sampler dynesty
fi

# scalar_tensor_injection_data0_0_analysis_H1L1V1_final_result
# PARENTS scalar_tensor_injection_data0_0_analysis_H1L1V1
# CHILDREN 
if [[ "scalar_tensor_injection_data0_0_analysis_H1L1V1_final_result" == *"$1"* ]]; then
    echo "Running: /home/chun-fung.wong/.conda/envs/nullpol-3.10/bin/bilby_result --result outdir_scalar_tensor_injection/result/scalar_tensor_injection_data0_0_analysis_H1L1V1_result.hdf5 --outdir outdir_scalar_tensor_injection/final_result --extension hdf5 --max-samples 20000 --lightweight --save"
    /home/chun-fung.wong/.conda/envs/nullpol-3.10/bin/bilby_result --result outdir_scalar_tensor_injection/result/scalar_tensor_injection_data0_0_analysis_H1L1V1_result.hdf5 --outdir outdir_scalar_tensor_injection/final_result --extension hdf5 --max-samples 20000 --lightweight --save
fi

