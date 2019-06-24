source ~/anaconda3/etc/profile.d/conda.sh
conda activate drlnd
python ars_multi.py --env Stoch2-v1 --lr 0.09 --noise 0.03 --steps 30 --episode_length 1000 --energy_weight 0.05 --logdir Stoch2_Jun14_1 --normal 0 
