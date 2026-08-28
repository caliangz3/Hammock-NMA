library(gemtc)
library(BUGSnet)

data(thrombolytic)

varname.t = "treatment"; varname.s = "study"
outcome = "events"; N = "sampleSize"; reference = "SK"
model_type = "random"
seed = 2026

net.model <- data.prep(arm.data = thrombolytic,
                       varname.t = varname.t,
                       varname.s = varname.s)

png("network_plot.png", width = 2000, height = 2000, res = 300)
par(mar = c(0, 0, 0, 0))
net.plot(net.model, 
         node.scale = 5, 
         edge.scale = 1.5,
         study.counts = TRUE, node.colour = "#beaed4",node.lab.cex=1.5, edge.lab.cex=1.3)
dev.off()

set.seed(seed)
effects_model <- nma.model(data = net.model,
                           outcome = outcome,
                           N = N,
                           reference = reference,
                           family = "binomial",
                           link = "logit",
                           type = "consistency",
                           effects = model_type)
effects_results <- nma.run(effects_model,
                           n.adapt = 1000,
                           n.burnin = 1000,
                           n.iter = 20000,
                           thin = 3)

# Obtain DIC
model_fit <- nma.fit(effects_results)
print(paste("DIC =", model_fit$DIC, sep=" "))

# Each result in each MCMC iterations (3 chains * 20000 iterations)
mcmc_samples <- as.matrix(effects_results[["samples"]])
treatment_effects_col <- grep("d\\[", colnames(mcmc_samples))
treament_effect <- mcmc_samples[, treatment_effects_col]

treatment_name_info = effects_results[["trt.key"]]
colnames(treament_effect) = treatment_name_info

write.csv(treament_effect, "Boland2003_treatment_effect.csv", row.names = F, eol = "\n")


png("forest_plot.png", width = 2000, height = 2000, res = 300)
forest_plot = nma.forest(effects_results,
                         central.tdcy = "mean",
                         comparator = reference,
                         log.scale = TRUE)
forest_plot+theme(
  axis.text.x  = element_text(size = 14), 
  axis.text.y  = element_text(size = 14), 
  axis.title.x = element_text(size = 16),
  axis.title.y = element_text(size = 16),
  plot.margin = margin(t=30)
)

dev.off()


treatment_order <- c(
  "AtPA",
  "Ten",
  "UK",
  "Ret",
  "SKtPA",
  "tPA",
  "SK",
  "ASPAC"
)


png("barchart.png", width = 2000, height = 2000, res = 300)
bar_chart <- nma.rank(effects_results, 
                      largerbetter=FALSE)
bar_chart$rankogram+
  scale_x_discrete(limits = treatment_order)+
  theme(axis.text.x = element_text(size = 14),
        axis.title.x = element_text(size = 16),
        axis.text.y = element_text(size = 14),
        axis.title.y = element_text(size = 16),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 16),
        plot.margin = margin(t = 30))
dev.off()


library(ggplot2)
library(tidyr)
library(dplyr)

df <- readxl::read_excel("all_ranks.xlsx")

rank_long <- df %>%
  mutate(Rank = row_number()) %>%
  pivot_longer(cols = c(PBV, Median, EV, SUCRA),
               names_to = "Metric",
               values_to = "Treatment") %>%
  mutate(Metric = factor(Metric,
                         levels = c("PBV", "Median", "EV", "SUCRA")))

png("Ranking.png", width = 1700, height = 1700, res = 300)
ggplot(rank_long, aes(x = Metric, y = Rank, group = Treatment, 
                      color = Treatment)) +
  geom_hline(yintercept = 1:8, color = "grey90",linewidth = 1) +
  geom_line(linewidth = 1) +
  geom_point(size = 8) +
  geom_text(aes(label = Rank), color = "white", size = 6, fontface = "bold") +
  scale_color_brewer(palette = "Set2") +
  scale_y_reverse(limits = c(8.5, 0.5), breaks = 1:8,expand = c(0.01, 0.01)) +
  theme_classic() +
  theme(axis.title = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks = element_blank(),
        axis.line = element_blank(),
        axis.text.x = element_text(size = 14),
        legend.position = "right",
        legend.title = element_blank(),
        legend.text = element_text(size=11),
        plot.margin = margin(l=30))
dev.off()




library(png)
library(cowplot)
library(patchwork)
library(magick)


pic1 <- readPNG("network_plot.png")
pic2 <- readPNG("forest_plot.png")
pic3 <- readPNG("barchart.png")
pic4 <- readPNG("Ranking.png")

pic1Plot <- ggdraw() + draw_image(pic1)
pic2Plot <- ggdraw() + draw_image(pic2)
pic3Plot <- ggdraw() + draw_image(pic3)
pic4Plot <- ggdraw() + draw_image(pic4)

figure <- plot_grid(
  pic1Plot,
  pic2Plot,
  pic3Plot,
  pic4Plot, 
  nrow = 2,
  align = "h",
  labels = c("A", "B", "C", "D"),
  label_y = 0.99,
  label_x = 0.02,
  label_size = 20
)


ggsave(
  "combine.png",
  figure,
  dpi = 600,
  bg = "white",
  units = "in",
  width = 9,
  height = 10
)
