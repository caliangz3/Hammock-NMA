library(netmeta)
source("C:/Users/橘白/Desktop/waterloo/Grad/MMath/get_MCMC_results.R")

data(Gurusamy2011)

get_MCMC_results(data = Gurusamy2011, varname.t = "treatment", varname.s = "study", 
                 outcome = "death", N = "n", reference = "Control/Placebo",
                 model_type = "fixed")
