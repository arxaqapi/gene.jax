"""
Run all benchmarks for the paper

1. Brax encoding benchmarks 
    1. GENE encoding vs direct on multiple problems
    2. W. different HPs 
        1. layers: [128, 128], [32, 32 32, 32]) 
        2. model types: relu + tanh, all tanh
        3. Algorithm change (DES / CMA-ES / Sep CMA-ES)
2. RL algorithms baselines on brax v1
    1. A2C
    2. DDPG
3. meta evolution w. comparable baselines (should be above)
    1. NN based meta-evol
    2. CGP based meta-evol
"""

