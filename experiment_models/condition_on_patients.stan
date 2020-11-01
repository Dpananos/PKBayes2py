functions{
  vector heaviside(vector t){
    
    vector[size(t)] y;
    for(i in 1:size(t)){
      y[i] = t[i]<=0 ? 0 : 1;
    }
    return y;
  }
  
  
  vector conc(real D, vector t, real Cl, real ka, real ke){
    
    return heaviside(t) .* (exp(-ka*t) - exp(-ke*t)) * (0.5 * D * ke * ka ) / (Cl *(ke - ka));
  }
  
}
data{
  int n;
  vector[n] observed_concentrations;
  vector[n] observed_times;

  real sex;
  real age;
  real weight;
  real creatinine;
  
  int nt;
  vector[nt] prediction_times;
  
  int n_doses;
  vector[n_doses] doses;
  vector[n_doses] dose_times;
  
  
  real mean_beta_a_1;
  real mean_beta_a_2;
  real mean_beta_a_3;
  real mean_beta_a_4;
  
  real sd_beta_a_1;
  real sd_beta_a_2;
  real sd_beta_a_3;
  real sd_beta_a_4;
  
  real mean_beta_cl_1;
  real mean_beta_cl_2;
  real mean_beta_cl_3;
  real mean_beta_cl_4;
  
  real sd_beta_cl_1;
  real sd_beta_cl_2;
  real sd_beta_cl_3;
  real sd_beta_cl_4;
    
  real mean_beta_t_1;
  real mean_beta_t_2;
  real mean_beta_t_3;
  real mean_beta_t_4;
  
  real sd_beta_t_1;
  real sd_beta_t_2;
  real sd_beta_t_3;
  real sd_beta_t_4;
  
  real mean_mu_alpha;
  real mean_mu_cl;
  real mean_mu_tmax;
  
  real sd_mu_alpha;
  real sd_mu_cl;
  real sd_mu_tmax;
  
  real rate_sigma;
  real shape_sigma;
  
  real rate_s_alpha;
  real shape_s_alpha;
  
  real rate_s_t;
  real shape_s_t;
  
  real rate_s_cl;
  real shape_s_cl;
  
  
  real weight_mean;
  real age_mean;
  real creatinine_mean;
  real weight_sd;
  real age_sd;
  real creatinine_sd;
}
transformed data{
  real scaled_weight = (weight - weight_mean)/weight_sd;
  real scaled_age = (age - age_mean)/age_sd;
  real scaled_creatinine = (creatinine - creatinine_mean)/creatinine_sd;
  row_vector[4] X = [sex, scaled_weight, scaled_creatinine, scaled_age];
}
parameters{
  real mu_cl;
  vector[4] beta_cl;
  real<lower=0> s_cl;
  real z_cl;
  
  real mu_t;
  vector[4] beta_t;
  real <lower=0> s_t;
  real z_t;
  
  
  real mu_alpha;
  real<lower=0> s_alpha;
  vector[4] beta_alpha;
  real z_a;
  
  real<lower=0> sigma;
}
transformed parameters{
  real<lower=0, upper=1> alpha = inv_logit(mu_alpha + X*beta_alpha + z_a*s_alpha);
  real<lower=0> tmax = exp(mu_t + X*beta_t + z_t*s_t);
  real<lower=0> cl = exp(mu_cl + X*beta_cl + z_cl*s_cl);
  real<lower=0> ka = log(alpha)/(tmax * (alpha-1));
  real<lower=0, upper = ka> ke = alpha*ka;
  vector<lower=0>[n] C = rep_vector(0.0, n);
  for(i in 1:n_doses){
    C += conc(doses[i], observed_times - dose_times[i], cl, ka, ke);
  }
}
model{
  mu_cl ~ normal(mean_mu_cl, sd_mu_cl);
  s_cl ~ gamma(shape_s_cl, rate_s_cl);
  beta_cl[1] ~ normal(mean_beta_cl_1, sd_beta_cl_1);
  beta_cl[2] ~ normal(mean_beta_cl_2, sd_beta_cl_2);
  beta_cl[3] ~ normal(mean_beta_cl_3, sd_beta_cl_3);
  beta_cl[4] ~ normal(mean_beta_cl_4, sd_beta_cl_4);
  z_cl ~ std_normal();
  
  
  mu_t ~ normal(mean_mu_tmax, sd_mu_tmax);
  s_t ~ gamma(shape_s_t, rate_s_t);
  beta_t[1] ~ normal(mean_beta_t_1, sd_beta_t_1);
  beta_t[2] ~ normal(mean_beta_t_2, sd_beta_t_2);
  beta_t[3] ~ normal(mean_beta_t_3, sd_beta_t_3);
  beta_t[4] ~ normal(mean_beta_t_4, sd_beta_t_4);
  z_t ~ std_normal();
  
  
  mu_alpha ~ normal(mean_mu_alpha, sd_mu_alpha);
  s_alpha  ~ gamma(shape_s_alpha, rate_s_alpha);
  beta_alpha[1] ~ normal(mean_beta_a_1, sd_beta_a_1);
  beta_alpha[2] ~ normal(mean_beta_a_2, sd_beta_a_2);
  beta_alpha[3] ~ normal(mean_beta_a_3, sd_beta_a_3);
  beta_alpha[4] ~ normal(mean_beta_a_4, sd_beta_a_4);
  z_a ~ std_normal();
  
  sigma ~  gamma(shape_sigma, rate_sigma);
  observed_concentrations ~ lognormal(log(C), sigma);
  
}
generated quantities{
  vector[nt] ypred = rep_vector(0.0, nt);
  
  for(i in 1:n_doses){
    ypred += conc(doses[i], prediction_times - dose_times[i], cl, ka, ke);
  }
}