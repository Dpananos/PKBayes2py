data{
  int N; //Total number of observations
  int subjectids[N]; //Subject idendification number as an integer.  Mine go from 1 - 36
  int n_subjects; //How many unique subjects do I have?
  vector[N] times; //Times at which subjects were observed?  Length N
  vector[n_subjects] D; //Doses.
  real yobs[N]; //Observed concentraitons
}
parameters{
  //Mu is on the log scale
  //Because x ~ lognormal -> log(x) ~ normal
  //I model on the log scale and then take exp to give me a lognormal rv
  real<lower=0>  mu_cl;
  real<lower=0> s_cl;
  vector[n_subjects] z_cl;
  
  real<lower=0> mu_tmax;
  real<lower=0> s_t;
  vector[n_subjects] z_t;
  
  vector<lower=0, upper=1>[n_subjects] alpha ;
  
  real<lower=0, upper=1> phi;
  real<lower=0, upper=1> kappa;
  vector<lower=0, upper=1>[n_subjects] delays;
  
  real<lower=0> sigma;
}
transformed parameters{
  vector[n_subjects] Cl = exp(mu_cl + z_cl*s_cl);
  vector[n_subjects] t = exp(mu_tmax + z_t*s_t);
  vector[n_subjects] ka = log(alpha)./(t .* (alpha-1));
  vector[n_subjects] ke = alpha .* ka;
  //Since first non-zero observation is t = 0.5, we know delay is no larger than 0.5
  //since delay is beta, if we multiply by 0.5 then the largest the delay can be is 0.5
  vector[N] delayed_times = times - 0.5*delays[subjectids];
  
  //Vectorized concentration function
  vector[N] C = (0.5 * D[subjectids] ./ Cl[subjectids]) .* (ke[subjectids] .* ka[subjectids]) ./ (ke[subjectids] - ka[subjectids]) .* (exp(-ka[subjectids] .* delayed_times) -exp(-ke[subjectids] .* delayed_times));
}
model{
  mu_tmax ~ normal(log(3.3), 0.25);
  s_t ~ gamma(10, 100);
  z_t ~ normal(0,1);
  
  mu_cl ~ normal(log(3.3),0.15);
  s_cl ~ gamma(15,100);
  z_cl ~ normal(0,1);

  alpha ~ beta(2,2);
  phi ~ beta(20,20);
  kappa ~ beta(20,20);
  delays ~ beta(phi/kappa, (1-phi)/kappa);
  sigma ~ lognormal(log(0.1), 0.2);
  yobs ~ lognormal(log(C), sigma);
}
generated quantities{
  real pop_cl = exp( mu_cl + std_normal_rng()*s_cl);
  real pop_tmax = exp( mu_tmax + std_normal_rng()*s_t);
  real pop_alpha = beta_rng(phi, kappa);
}


