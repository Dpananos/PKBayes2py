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
  
  
  // This will be given at prediction time
  int nt;
  vector[nt] prediction_times;
  
  int n_doses;
  vector[n_doses] doses;
  vector[n_doses] dose_times;
}
generated quantities{
  
  real cl  = lognormal_rng(0.69, 0.32);
  real tmax = lognormal_rng(0.98, 0.24);
  real alpha = beta_rng(2,2);
  real ka = log(alpha)/(tmax * (alpha-1));
  real ke = alpha*ka;
  vector[nt]  C = rep_vector(0.0, nt);
  vector[nt] C_noise = rep_vector(0.0, nt);
  real sigma = gamma_rng(250.67, 1458.77);

    for(k in 1:n_doses){
      C += conc(doses[k], prediction_times - dose_times[k], cl, ka, ke);
    }

  for (i in 1:nt){
    C_noise[i] = lognormal_rng(log(C[i]), sigma );
  }
  

    
  
}