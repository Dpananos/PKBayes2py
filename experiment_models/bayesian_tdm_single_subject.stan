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
  
  int nt;
  vector[nt] prediction_times;
  
  int n_doses;
  vector[n_doses] doses;
  vector[n_doses] dose_times;
  vector[1] c0_time;

}
parameters{
  
  real<lower=0> cl;
  real<lower=0> tmax;
  real<lower=0, upper=1> alpha;
  real<lower=0> sigma;
}
transformed parameters{
  real<lower=0> ka = log(alpha)/(tmax * (alpha-1));
  real<lower=0, upper = ka> ke = alpha*ka;
  vector<lower=0>[n] C = rep_vector(0.0, n);
  for(i in 1:n_doses){
    C += conc(doses[i], observed_times - dose_times[i], cl, ka, ke);
  }
}
model{
  cl ~ lognormal(0.69, 0.32);
  tmax ~ lognormal(0.98, 0.24);
  alpha ~ beta(2,2);
  
  
  sigma ~  gamma(250.67, 1458.77);
  observed_concentrations ~ lognormal(log(C), sigma);
  
}
generated quantities{
  vector[1] c0=[0]';
  vector[nt] initial_concentration = rep_vector(0.0, nt);
  vector[nt] ypred = rep_vector(0.0, nt);
  
  for(i in 1:n_doses){
   c0 += conc(doses[i], c0_time - dose_times[i], cl, ka, ke);
  }
  
  initial_concentration = exp(-ke*prediction_times)*c0[1];
  
  for(i in 1:n_doses){
    ypred += conc(doses[i], prediction_times - dose_times[i], cl, ka, ke);
  }
  

  
}