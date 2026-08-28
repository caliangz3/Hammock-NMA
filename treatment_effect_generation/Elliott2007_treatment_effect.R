library(BUGSnet)

data(diabetes)

prepped.data <- data.prep(arm.data = diabetes.sim,
                          varname.t = "Treatment",
                          varname.s = "Study")

random_effects_model <- nma.model(data = prepped.data,
                                  outcome = "diabetes",
                                  N = "n",
                                  reference = "Diuretic",
                                  family = "binomial",
                                  link = "cloglog",
                                  time = "followup",
                                  effects = "random")

set.seed(2026)

random_effects_results <- nma.run(random_effects_model,
                                  n.adapt = 1000,
                                  n.burnin = 1000,
                                  n.iter = 20000,
                                  thin = 3)

# Obtain DIC
model_fit <- nma.fit(random_effects_results)
print(paste("DIC =", model_fit$DIC, sep=" "))
  
# Each result in each MCMC iterations (3 chains * 20000 iterations)
mcmc_samples <- as.matrix(random_effects_results[["samples"]])
treatment_effects_col <- grep("d\\[", colnames(mcmc_samples))
treament_effect <- mcmc_samples[, treatment_effects_col]
  
treatment_name_info = random_effects_results[["trt.key"]]
colnames(treament_effect) = treatment_name_info

write.csv(treament_effect, "Elliott2007_treatment_effect.csv", row.names = F, eol = "\n")