library(tidyverse)
library(patchwork)
d <- read_csv('../../data/generated_data/process_figure_data.csv')

theme_set(theme_light(base_size = 8))

styling<-ggplot(data=d)+
         scale_x_continuous(labels = function(x) x/24, breaks = seq(0, 240, 24*2)) +
         scale_y_continuous(labels = function(x) 1000*x, limits = c(0, .6))+
         labs(x='Time (Days)', y = 'Concentration (ng/ml)')

prior_plot<-styling+
  geom_rect(aes(xmin=0, xmax=240, ymin=0.1, ymax = 0.3), fill = 'light gray', color = 'black')+
  geom_ribbon(aes(x=tpre, ymin=prior_low, ymax=prior_high), fill = 'blue', alpha = 0.5)
  

plot_a<-prior_plot + 
        geom_text(aes(x=7.5*24, y=.290, label = 'Desired Range'), vjust=1, size = 4, color = 'black') + 
        geom_label(aes(x=2*24, y=.600, label = 'Stage 1'), vjust=1, size = 4, color = 'black') + 
        labs(title='Predictions Under Optimal Dose')

plot_b<-prior_plot + 
        geom_point(aes(tobs, yobs)) +
        geom_label(aes(x=2*24, y=.600, label = 'Stage 1'), vjust=1, size = 4, color = 'black') + 
        geom_line(aes(tpre, ytrue_prior+.125))+
        labs(title = 'An Observation is Made')

plot_c<-plot_b + 
        geom_ribbon(aes(x=tpost, ymin=post_low, ymax=post_high), fill = 'red', alpha = 0.5) +
        geom_label(aes(x=2*24, y=.600, label = 'Stage 1'), vjust=1, size = 4, color = 'black') + 
        geom_label(aes(x=8*24, y=.600, label = 'Stage 2'), vjust=1, size = 4, color = 'black') + 
        labs(title = 'Updated Predictions Under Adjusted Dose')


slicer = seq(1, nrow(d), 5)
plot_d<-plot_c + 
        geom_line(aes(tpost, ytrue_post+0.025)) + 
        labs(title='True Concentration Function')


final_plot <- (plot_a | plot_b) / (plot_c + plot_d) + plot_annotation(tag_levels = 'A')

final_plot

ggsave('viz_of_process.png', dpi = 240, height = 5, width = 8)
