library(BUGSnet)
library(ggplot2)
library(stringr)

# model_type  =c("fixed", "random")
# small_values_good = c(True, False)
get_MCMC_results <- function(data, varname.t, varname.s, outcome, N, reference,
                         model_type, output_name, seed = 2026, n.adapt = 1000, n.burnin = 1000,
                         n.iter = 20000) {
  
  net.model <- data.prep(arm.data = data,
                         varname.t = varname.t,
                         varname.s = varname.s)
  
  effects_model <- nma.model(data = net.model,
                             outcome = outcome,
                             N = N,
                             reference = reference,
                             family = "binomial",
                             link = "logit",
                             type = "consistency",
                             effects = model_type)
  
  set.seed(seed)
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
  
  write.csv(treament_effect, output_name, row.names = F)

}