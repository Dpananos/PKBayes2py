library(tidyverse)
theme_set(theme_classic())

d = read_csv('../../data/generated_data/simulation_results.csv')

method_order = c('No Covariate Model','No Covariate Model + 1 Sample', 'Covarite Model', 'Covarite Model + 1 Sample','Optimal Sampling Time', 'Q Learning')
method_rename = c('One Size Fits All','One Size + One Sample', 'Clinical Variables', 'Clinical Variables + One Sample','Optimal Sampling Time', 'Optimal Sequential Dosing')
method_order = factor(method_order, levels = method_order, ordered = T)
method_rename = factor(method_rename, levels = method_rename, ordered = T)

type = c('Static','Dynamic','Static', 'Dynamic','Dynamic','Dynamic')

methods = tibble(Method = method_order, type, method_rename)
d$Method = factor(d$Method, levels = method_order, ordered = T)

d = d %>% 
  left_join(methods)


d %>% 
  ggplot(aes(delta, fct_rev(method_rename)))+
  geom_boxplot(aes(fill = fct_rev(type)), notch = T, outlier.size = 1, outlier.alpha = 0.5)+
  labs(x = expression(paste(Delta,'Reward')),
       y = 'Mode of Personalization',
       fill = 'Personalization Type')+
  theme(
    panel.grid.major = element_line(),
    panel.grid.minor = element_line(),
    legend.position = 'bottom'
  )+
  scale_fill_brewer(palette = 'Set2', direction = -1)

  ggsave('models_of_personalization_differences.png', height = 9, width = 16, units = 'cm', dpi = 480)
  