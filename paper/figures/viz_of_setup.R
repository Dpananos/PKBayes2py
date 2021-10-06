library(tidyverse)
d <- read_csv('../../data/generated_data/process_figure_data.csv')


prior_plot<-ggplot(data=d)+
  geom_ribbon(aes(x=seq(0, 240, length.out = 240), ymin=0.1, ymax = 0.3), fill = 'light gray')+
  geom_ribbon(aes(x=tpre, ymin=prior_low, ymax=prior_high), fill = 'blue', alpha = 0.5)+
  geom_line(aes(x=tpre, y=ytrue_prior))+
  geom_point(aes(tobs, yobs))+
  geom_point(data=d[seq(1, nrow(d), 4),], aes(x=tpre, y=ytrue_prior), shape=4)

full_plot<-prior_plot+
  geom_ribbon(aes(x=tpost, ymin=post_low, ymax=post_high), fill = 'red', alpha = 0.5)+
  geom_line(aes(x=tpost, y=ytrue_post))+
  geom_point(data=d[seq(1, nrow(d), 4),], aes(x=tpost, y=ytrue_post), shape=4)


  


final_plot<-full_plot +
  theme_classic()+
  theme(aspect.ratio = 1/1.61,
        panel.grid.major = element_line(color='light grey'))+
  scale_x_continuous(labels = function(x) x/24, breaks = 24*seq(0, 10, 2))+
  scale_y_continuous(labels = function(x) x*1000, breaks = seq(0, 0.4, 0.1))+
  labs(x='Time (Days)',
       y='Concentration ng/ml')


final_plot
ggsave('viz_of_process.png', height = 4, width = 4*1.61)
