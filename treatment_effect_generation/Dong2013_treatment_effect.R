library(netmeta)

data(Dong2013)


pw <- pairwise(treat = treatment,
               event = death,
               n = randomized,
               studlab = id,
               data = Dong2013,
               sm = "OR", allstudies = T)


MH_NMA <- netmetabin(event1 = event1, n1 = n1, event2 = event2, n2 = n2, 
                     treat1 = treat1, treat2 = treat2,
                     studlab = studlab, data = pw, sm = "OR", method = "MH",
                     reference.group = "Placebo", common = T, random = F)

netrank(MH_NMA, small.values = "good")

TE_mat <- MH_NMA$TE.common
SE_mat <- MH_NMA$seTE.common

ref <- "Placebo"  

TE <- -MH_NMA$TE.common[ref,]
SE <- MH_NMA$seTE.common[ref,]

# remove reference itself
TE <- TE[names(TE) != ref]
SE <- SE[names(SE) != ref]

TE
SE

set.seed(2026)
n <- 60000

theta_samples <- sapply(seq_along(TE), function(j) {
  rnorm(n, mean = TE[j], sd = SE[j])
})

colnames(theta_samples) <- names(TE)

# add reference treatment back as effect 0
theta_samples_all <- cbind(plac = 0, theta_samples)


thin = theta_samples_all[seq(1, nrow(theta_samples_all), by = 3),]
write.csv(thin, "Dong2013_treatment_effect.csv", row.names = F)

#netrank(MH_NMA, small.values = "good")
