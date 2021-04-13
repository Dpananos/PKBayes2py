library(tidyverse)
theme_set(theme_classic())

d = read_csv('../data/generated_data/simulation_results.csv')

method_order = c('No Covariate Model','No Covariate Model + 1 Sample', 'Covarite Model', 'Covarite Model + 1 Sample','Optimal Sampling Time', 'Q Learning')
d$Method = factor(d$Method, levels = method_order, ordered = T)

d %>% 
  ggplot(aes(delta, fct_rev(Method)))+
  geom_boxplot(fill = 'Light Gray', notch = T, outlier.size = 1, outlier.alpha = 0.5)+
  labs(x = expression(paste(Delta,'U')),
       y = 'Mode of Personalization')+
  theme(
    panel.grid.major = element_line(),
    panel.grid.minor = element_line(),
  )

  ggsave('models_of_personalization_differences.png', height = 9, width = 16, units = 'cm', dpi = 480)
  