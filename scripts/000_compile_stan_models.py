import cmdstanpy
import os
# Compile models to be used 
model_files = ('experiment_models/'+file for file in os.listdir('experiment_models') if file.endswith('.stan'))

for model in model_files:
    print(f'\n\ncompiling {model}\n\n')
    cmdstanpy.CmdStanModel(stan_file = model)