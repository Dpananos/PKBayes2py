import pickle

with open('data/generated_data/param_summary.pkl', 'rb') as file:
    x = pickle.load(file)


for i in range(1, 5):
    mu = round(x[f'mean_beta_a_{i}'],2)
    sigma = round(x[f'sd_beta_a_{i}'],2)
    print(i, mu, sigma)
