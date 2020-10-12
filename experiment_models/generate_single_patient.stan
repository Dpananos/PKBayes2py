functions{
  vector heaviside(vector t){
    
    vector[size(t)] y;
    for(i in 1:size(t)){
      y[i] = t[i]<0 ? 0 : 1;
    }
    return y;
  }
  
  
  vector conc(real D, vector t, real Cl, real ka, real ke){
    
    return heaviside(t) .* (exp(-ka*t) - exp(-ke*t)) * (0.5 * D * ke * ka ) / (Cl *(ke - ka));
  }
}

data{
  
  // Patient covariates.  We need these to make predictions prior to seeing data.
  real sex;
  real weight;
  real creatinine;
  real age;
  
  // This will be given at prediction time
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
  
  vector[nt] C = rep_vector(0.0, nt);
  vector[nt] C_obs;
  
  for(i in 1:n_doses){
    C += conc(doses[i], prediction_times - dose_times[i], cl, ka, ke);
  }
  C_obs = to_vector(lognormal_rng( log(C) , sigma ));  
}