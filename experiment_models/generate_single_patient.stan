//This is a vectorized version to generate patients.  If I wanted to generate 1000 patients in one go, I could use this.
//Would need to pass patient covariates as a vector in order for them to be generated.

functions{
  vector pk_ode(real t, vector y, real Cl, real ke, real ka,  int n_doses, vector doses, vector dose_times){
    
    vector[1] dydt;
    dydt[1] = -ke*y[1];
    
    for(i in 1:n_doses){
      if(t>dose_times[i]){
        dydt[1] = dydt[1] + 0.5*doses[i] * ke * ka * exp( -ka * (t-dose_times[i]) ) / Cl;
      }
    }
    
    return dydt;
  }
}

data{
  
  real sex;
  real weight;
  real creatinine;
  real age;
  
  
  int n_doses;
  int n_pred;
  real t_obs[n_pred];
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
  real t0 = 0;
  vector[1] y0 = [0]';
  
  real scaled_weight = (weight - weight_mean)/weight_sd;
  real scaled_age = (age - age_mean)/age_sd;
  real scaled_creatinine = (creatinine - creatinine_mean)/creatinine_sd;
  row_vector[4] X = [sex, scaled_weight, scaled_creatinine, scaled_age];
}
generated quantities{
  
  real mu_cl = normal_rng(mean_mu_cl, sd_mu_cl);
  real mu_tmax = normal_rng(mean_mu_tmax, sd_mu_tmax);
  real mu_alpha = normal_rng(mean_mu_alpha, sd_mu_alpha);
  
  real s_cl = gamma_rng(shape_s_cl, rate_s_cl);
  real s_tmax = gamma_rng(shape_s_t, rate_s_t);
  real s_alpha = gamma_rng(shape_s_alpha, rate_s_alpha);
  
  real b1_cl = normal_rng(mean_beta_cl_1, sd_beta_cl_1);
  real b2_cl = normal_rng(mean_beta_cl_2, sd_beta_cl_2);
  real b3_cl = normal_rng(mean_beta_cl_3, sd_beta_cl_3);
  real b4_cl = normal_rng(mean_beta_cl_4, sd_beta_cl_4);
  vector[4] beta_cl = to_vector([b1_cl, b2_cl, b3_cl, b4_cl]);

  real b1_t = normal_rng(mean_beta_t_1, sd_beta_t_1);
  real b2_t = normal_rng(mean_beta_t_2, sd_beta_t_2);
  real b3_t = normal_rng(mean_beta_t_3, sd_beta_t_3);
  real b4_t = normal_rng(mean_beta_t_4, sd_beta_t_4);
  vector[4] beta_t = to_vector([b1_t, b2_t, b3_t, b4_t]);
    
  real b1_alpha = normal_rng(mean_beta_a_1, sd_beta_a_1);
  real b2_alpha = normal_rng(mean_beta_a_2, sd_beta_a_2);
  real b3_alpha = normal_rng(mean_beta_a_3, sd_beta_a_3);
  real b4_alpha = normal_rng(mean_beta_a_4, sd_beta_a_4);
  vector[4] beta_alpha = to_vector([b1_alpha, b2_alpha, b3_alpha, b4_alpha]);

  real cl = exp(normal_rng(mu_cl + X*beta_cl, s_cl));
  real tmax = exp(normal_rng(mu_tmax + X*beta_t, s_tmax));
  real alpha =  inv_logit(normal_rng(mu_alpha + X*beta_alpha, s_alpha));
  real ka = log(alpha)/(tmax * (alpha-1));
  real ke = alpha * ka;
  
  real sigma = gamma_rng(shape_sigma, rate_sigma);
  
  vector[n_pred] C;
  vector[n_pred] C_obs;
  
  C = to_vector(ode_rk45(pk_ode, y0, t0, t_obs, cl, ke, ka,  n_doses, doses,  dose_times )[,1]);
  C_obs = to_vector(lognormal_rng( log(C) , sigma ));  
}