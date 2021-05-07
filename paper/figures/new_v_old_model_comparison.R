library(cmdstanr)
library(tidyverse)
library(tidybayes)
library(here)
library(posterior)

theme_set(theme_classic())

kg_labeler = function(x) glue::glue('{x} kg')

experiment_data = read_csv('../data/generated_data/experiment.csv')

old_model_data = list(
  N = nrow(experiment_data),
  subjectids = experiment_data$subjectids,
  n_subjects = n_distinct(experiment_data$subjectids),
  times = experiment_data$time,
  D = rep(2.5, 36),
  yobs = experiment_data$yobs
)

new_model_data = list(
  n = nrow(experiment_data),
  subjectids = experiment_data$subjectids,
  n_subjectids = n_distinct(experiment_data$subjectids),
  time = experiment_data$time,
  yobs = experiment_data$yobs,
  sex = distinct(experiment_data, subjectids, .keep_all = T)$sex,
  age =  distinct(experiment_data, subjectids, .keep_all = T)$age,
  creatinine =  distinct(experiment_data, subjectids, .keep_all = T)$creatinine,
  weight = distinct(experiment_data, subjectids, .keep_all = T)$weight,
  D = distinct(experiment_data, subjectids, .keep_all = T)$D
)

old_model = cmdstan_model('../experiment_models/tdm_model.stan')
old_fit = old_model$sample(old_model_data, chains=12, parallel_chains=4)

new_model = cmdstan_model('../experiment_models/original_model.stan')
new_fit = new_model$sample(new_model_data, chains = 12, parallel_chains=4, seed=19920908)



z_cl_new = new_fit$draws() %>% 
           as_draws_df %>% 
           spread_draws(z_cl[subjectids]) %>% 
           mean_qi() %>% 
           mutate(which = 'With Covariates')

z_cl_old = old_fit$draws() %>% 
  as_draws_df %>% 
  spread_draws(z_cl[subjectids]) %>% 
  mean_qi() %>% 
  mutate(which = 'Without Covariates')

plot_data = z_cl_old %>% 
  bind_rows(z_cl_new)  %>% 
  inner_join(distinct(experiment_data, subjectids, .keep_all=T)) %>% 
  mutate(which = factor(which, ordered = T, levels = c('Without Covariates', 'With Covariates')),
         sex = if_else(sex>0, 'Male', 'Female'))

base_plot = plot_data %>% 
  ggplot(aes(weight, z_cl, ymin = .lower, ymax = .upper))+
  geom_pointrange(aes(color = factor(sex)))+
  geom_smooth(method = 'lm', se = T, linetype = 'dashed', color = 'black')+
  facet_wrap(~which)

final_plot = base_plot +
  scale_color_brewer(palette = "Set2")+
  scale_x_continuous(labels = kg_labeler)+
  theme(aspect.ratio = 1/1.61, 
        panel.grid.major = element_line(),
        legend.position = 'bottom'
        )+
  labs(x='Weight', 
       y='Estimated Clearance Random Effect',
       color = 'Sex')

final_plot

ggsave('random_effects_change.png', final_plot, height = 9, width = 16, units = 'cm', dpi = 480)


c_new = new_fit$draws() %>% 
  as_draws_df %>% 
  spread_draws(C[i]) %>% 
  mean_qi() %>% 
  mutate(which = 'With Covariates')

c_old = old_fit$draws() %>% 
  as_draws_df %>% 
  spread_draws(C[i]) %>% 
  mean_qi() %>% 
  mutate(which = 'Without Covariates')

experiment_data %>% 
  mutate(i = seq_along(time)) %>% 
  inner_join(bind_rows(c_old, c_new), by = 'i')  %>% 
  select(i, time, subjectids, yobs, C, which) %>% 
  spread(which, C) %>% 
  group_by(subjectids) %>% 
  summarise(error_with_covars = Metrics::rmse(yobs, `With Covariates`),
            error_without_covars = Metrics::rmse(yobs, `Without Covariates`)) %>% 
  ungroup() %>% 
  summarise_all(~1000*mean(.))


r = rstan::read_stan_csv(new_fit$output_files())

bayesplot::neff_ratio(r) %>% min(., na.rm = T)



b_draws = new_fit$draws() %>% 
  posterior::as_draws_df() %>% 
  gather_draws(`beta_.*`[i], regex = T)

convert_vars = tibble(i = 1:4, covar = c('Sex','Weight','Creatinine', 'Age'))

b_draws %>%
  mutate(which_pk = str_remove(.variable, 'beta_')) %>% 
  mutate(
         which_pk = case_when(
           which_pk == 'a' ~ 'alpha',
           which_pk == 'cl' ~ 'Cl',
           which_pk == 't' ~ 't[max]'
         )) %>% 
  inner_join(convert_vars) %>% 
  ggplot(aes(.value, covar))+
  stat_halfeye(.width = 0.95, size = 0.5)+
  geom_vline(aes(xintercept = 0), size = 0.5)+
  facet_wrap(~which_pk, scales = 'free_y', labeller = label_parsed)+
  labs(x=expression(beta), y='')

ggsave('coef_vals.png', height = 9, width = 16, units = 'cm', dpi = 480)

new_fit$draws('beta_a[1]') %>% posterior::as_draws_df() %>% mean_qi()



new_fit$draws('ke') %>% 
  posterior::as_draws_df() %>% 
  spread_draws(ke[i]) %>% 
  mean_qi() %>% 
  left_join(
    experiment_data %>% distinct(subjectids, .keep_all=T) %>% mutate(i = seq_along(subjectids))
  ) %>% 
  ggplot(aes(age, ke))+
  geom_point()
  
     