library(netmeta)

data(Franchini2012)

df = Franchini2012[,1:5]
colnames(df) = c("Study", "Treatment", "y","sd","n")

treatment_2 = Franchini2012[,6:9]
treatment_2 = cbind(df$Study, treatment_2)
colnames(treatment_2) = colnames(df)

treatment_3 = Franchini2012[,10:13]
treatment_3 = cbind(df$Study, treatment_3)
colnames(treatment_3) = colnames(df)
treatment_3 = na.omit(treatment_3)

final_data = rbind(df, treatment_2, treatment_3)

net.model <- data.prep(
  arm.data = final_data,
  varname.t = "Treatment",
  varname.s = "Study"
)

#
fixed_effects_model <- nma.model(data = net.model,
                                 outcome = "y",
                                 sd = "sd",
                                 N = "n",
                                 reference = "Placebo",
                                 family = "normal",
                                 link = "identity",
                                 effects = "fixed")

set.seed(2026)
effects_results <- nma.run(fixed_effects_model,
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

write.csv(treament_effect, "Franchini2012_treatment_effect.csv", row.names = F)


