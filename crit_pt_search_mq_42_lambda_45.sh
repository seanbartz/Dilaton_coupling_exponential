#!/bin/bash
#PBS -S /bin/bash
#PBS -N cpt_mq_42_lambda_45
#PBS -j oe
#PBS -o ./cpt_mq_42.out
#PBS -l nodes=1:ppn=20
#PBS -l walltime=01:00:00
#PBS -M sean.bartz@indstate.edu
#PBS -m e
cd $PBS_O_WORKDIR

# Assign variables
lambda1=-4.25
mu0=430
mu1=830
mu2=176
ml=42
tmin=150
tmax=450
numtemp=25
minsigma=0
maxsigma=900
mu_initial=0
delta_mu=128
mu_precision=4
source my_env/bin/activate
# Call the function with the variables
python3 critical_point_runner.py $lambda1 $mu0 $mu1 $mu2 $ml $tmin $tmax $numtemp $minsigma $maxsigma $mu_initial $delta_mu $mu_precision
