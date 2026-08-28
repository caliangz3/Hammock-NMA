library(netmeta)
#source("C:/Users/橘白/Desktop/waterloo/Grad/MMath/get_MCMC_results.R")

data(Baker2009)

get_MCMC_results <- function(data, varname.t, varname.s, outcome, N, reference,
                             model_type, seed = 2026, n.adapt = 1000, n.burnin = 1000,
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
                             n.iter = 20000, thin=3)
  
  # Obtain DIC
  model_fit <- nma.fit(effects_results)
  print(paste("DIC =", model_fit$DIC, sep=" "))
  
  # Each result in each MCMC iterations (3 chains * 20000 iterations)
  mcmc_samples <- as.matrix(effects_results[["samples"]])
  treatment_effects_col <- grep("d\\[", colnames(mcmc_samples))
  treament_effect <- mcmc_samples[, treatment_effects_col]
  
  treatment_name_info = effects_results[["trt.key"]]
  colnames(treament_effect) = treatment_name_info
  
  write.csv(treament_effect, "Baker2009_treatment_effect.csv", row.names = F)

}

get_MCMC_results(data = Baker2009, varname.t = "treatment", varname.s = "study", 
                 outcome = "exac", N = "total", reference = "Placebo",
                 model_type = "random")
