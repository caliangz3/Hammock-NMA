library(netmeta)
source("get_MCMC_results.R")

data(Dogliotti2014)


get_MCMC_results(data = Dogliotti2014, varname.t = "treatment", varname.s = "study", 
                 outcome = "stroke", N = "total", reference = "Placebo/Control",
                 model_type = "random", output_name = "Dogliotti2014_treatment_effect.csv")
