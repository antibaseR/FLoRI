library(tidyverse)
library(wesanderson)
library(patchwork)
library(ggpubr)


p = list()
p_num = 1
for (sim_setting in c(3,4)){
  
  for (vul in c(0,1)){
    if (vul==1){
      # load the data
      data_flori = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_flori_hosp_vul', sim_setting,'.csv', sep=''), header=T)
      data_exp = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_exp_hosp_vul', sim_setting,'.csv', sep=''), header=T)
      data_naive = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_naive_hosp_vul', sim_setting,'.csv', sep=''), header=T)
      sub_flori = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_flori_sub_vul', sim_setting,'.csv', sep=''), header=T)
      sub_exp = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_exp_sub_vul', sim_setting,'.csv', sep=''), header=T)
      sub_naive = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_naive_sub_vul', sim_setting,'.csv', sep=''), header=T)

      if (sim_setting %in% c(1,2,6,7)){
        
        data_dritr = read.csv(paste('/Users/chenxinlei/RFL-IDR/DR-ITR/Result/value_df_sim_dritr_hosp_vul', sim_setting,'.csv', sep=''), header=T)
        sub_dritr = read.csv(paste('/Users/chenxinlei/RFL-IDR/DR-ITR/Result/value_df_sim_dritr_sub_vul', sim_setting,'.csv', sep=''), header=T)
        avg_reward = c(mean(sub_flori$X0),mean(sub_exp$X0), mean(sub_naive$X0), mean(sub_dritr$X0))
        
      } else {
        avg_reward = c(mean(sub_flori$X0),mean(sub_exp$X0), mean(sub_naive$X0))
      }
      
      
    } else {
      # load the data
      data_flori = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_flori_hosp', sim_setting,'.csv', sep=''), header=T)
      data_exp = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_exp_hosp', sim_setting,'.csv', sep=''), header=T)
      data_naive = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_naive_hosp', sim_setting,'.csv', sep=''), header=T)
      sub_flori = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_flori_sub', sim_setting,'.csv', sep=''), header=T)
      sub_exp = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_exp_sub', sim_setting,'.csv', sep=''), header=T)
      sub_naive = read.csv(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_naive_sub', sim_setting,'.csv', sep=''), header=T)
      
      if (sim_setting %in% c(1,2,6,7)){
        
        data_dritr = read.csv(paste('/Users/chenxinlei/RFL-IDR/DR-ITR/Result/value_df_sim_dritr_hosp', sim_setting,'.csv', sep=''), header=T)
        sub_dritr = read.csv(paste('/Users/chenxinlei/RFL-IDR/DR-ITR/Result/value_df_sim_dritr_sub', sim_setting,'.csv', sep=''), header=T)
        avg_reward = c(mean(sub_flori$X0),mean(sub_exp$X0), mean(sub_naive$X0), mean(sub_dritr$X0))
        
      } else {
        avg_reward = c(mean(sub_flori$X0),mean(sub_exp$X0), mean(sub_naive$X0))
      }
    }
    
    # reformat the data
    ## The current data dimension is 1000 * 1, that is the results of 10 hospitals in 100 iterations.
    ## Format the result to be 100*10
    data_flori_max = matrix(data_flori$X0, nrow=10, ncol=100) %>%
      t() %>%
      as.data.frame()
    colnames(data_flori_max) = paste0("S", seq(1,10,1))
    
    data_exp_max = matrix(data_exp$X0, nrow=10, ncol=100) %>%
      t() %>%
      as.data.frame()
    colnames(data_exp_max) = paste0("S", seq(1,10,1))
    
    data_naive_max = matrix(data_naive$X0, nrow=10, ncol=100) %>%
      t() %>%
      as.data.frame()
    colnames(data_naive_max) = paste0("S", seq(1,10,1))
    
    if (sim_setting %in% c(1,2,6,7)){
      data_dritr_max = matrix(data_dritr$X0, nrow=10, ncol=100) %>%
        t() %>%
        as.data.frame()
      colnames(data_dritr_max) = paste0("S", seq(1,10,1))
    }
    
    # add a method ID column
    data_flori_max$method = "FLoRI"
    data_exp_max$method = "Mean-optimal"
    data_naive_max$method = "Global"
    if (sim_setting %in% c(1,2,6,7)){
      data_dritr_max$method = "DR-ITR"
    }
    
    # combine the 4 datasets
    if (sim_setting %in% c(1,2,6,7)){
      data = rbind(rbind(rbind(data_flori_max, data_exp_max), data_naive_max), data_dritr_max)
    } else {
      data = rbind(rbind(data_flori_max, data_exp_max), data_naive_max)
    }
    
    # transfer to long format
    data_long = gather(data, hospital, reward, S1:S10, factor_key=TRUE)
    if (sim_setting %in% c(1,2,6,7)){
      data_long$method = factor(data_long$method, levels = c("FLoRI", "Mean-optimal", "Global", "DR-ITR"))
    } else {
      data_long$method = factor(data_long$method, levels = c("FLoRI", "Mean-optimal", "Global"))
    }
    
    if (sim_setting %in% c(1,2,6,7)){
      
      p[[(p_num)]] = data_long %>%
        ggplot(aes(x = hospital, y = reward, fill = method)) +
        #geom_boxplot(outlier.shape = NA) +
        geom_boxplot(lwd = 0.3, outlier.size = 0.3) +
        geom_hline(yintercept=avg_reward[1], color = "#FE2905", alpha = 0.5, size = 0.5) +
        geom_hline(yintercept=avg_reward[2], color = "#03A08A", alpha = 0.5, size = 0.5, linetype="dashed") +
        geom_hline(yintercept=avg_reward[3], color = "#F2AD1F", alpha = 0.5, size = 0.5, linetype="twodash") +
        geom_hline(yintercept=avg_reward[4], color = "#81E2EB", alpha = 0.5, size = 0.5, linetype="dotted") +
        labs(x ="Hospital", y = "Reward", fill="Method", 
             title = c("Binary treatment with a linear decision boundary",
                       "",
                       "Binary treatment with a nonlinear decision boundary", 
                       "",
                       "Violation of the causal assumptions",
                       "",
                       "No site variation of CATE",
                       ""
             )[p_num],
             subtitle = c("(A)", "(B)")[vul+1]) +
        scale_x_discrete(labels = c(paste("Site", seq(1, 10, 1), " "))) +
        scale_fill_manual(values = c("#FE2905", "#03A08A", "#F2AD1F", "#81E2EB")) +
        scale_y_continuous(limits = c(min(data_long$reward), max(data_long$reward)+1)) +
        #scale_y_continuous(limits = c(0.25, 1)) +
        theme(axis.text.x = element_text(size=20),
              axis.text.y = element_text(size=20)) +
        theme_classic()
      
    } else {
      
      p[[(p_num)]] = data_long %>%
        ggplot(aes(x = hospital, y = reward, fill = method)) +
        #geom_boxplot(outlier.shape = NA) +
        geom_boxplot(lwd = 0.3, outlier.size = 0.3) +
        geom_hline(yintercept=avg_reward[1], color = "#FE2905", alpha = 0.5, size = 0.5) +
        geom_hline(yintercept=avg_reward[2], color = "#03A08A", alpha = 0.5, size = 0.5, linetype="dashed") +
        geom_hline(yintercept=avg_reward[3], color = "#F2AD1F", alpha = 0.5, size = 0.5, linetype="twodash") +
        labs(x ="Hospital", y = "Reward", fill="Method", 
             title = c("Multi-category treatment with a linear decision boundary",
                       "",
                       "Multi-category treatment with a nonlinear decision boundary", 
                       ""
             )[p_num],
             subtitle = c("(A)", "(B)")[vul+1]) +
        scale_x_discrete(labels = c(paste("Site", seq(1, 10, 1), " "))) +
        scale_fill_manual(values = c("#FE2905", "#03A08A", "#F2AD1F")) +
        scale_y_continuous(limits = c(min(data_long$reward), max(data_long$reward)+1)) +
        #scale_y_continuous(limits = c(0.25, 1)) +
        theme(axis.text.x = element_text(size=20),
              axis.text.y = element_text(size=20)) +
        theme_classic()
      
    }
    
    p_num = p_num + 1
  }
  
}

plot = ggarrange(p[[1]], p[[2]], 
                 p[[3]], p[[4]],
                 ncol=2, nrow=2,
                 font.label = list(size = 20, color = "black", face = "bold", family = NULL, position = "top"),
                 common.legend = TRUE, legend="bottom")

plot


ggsave(paste('/Users/chenxinlei/RFL-IDR/result/Simulation results/Figures/simulation_multi_category.pdf', sep=''), width = 12, height = 7)

