library(nmadb)
library(netmeta)

# need package netmeta in version 2.9.0
cardiovascular <- runnetmeta(501267, model = "random")

save.image(file = "501267.RData")

rm(list = ls())


# need package netmeta in new version
load("501267.RData")

# p -value
netrank(cardiovascular, small.values = "good")

cardiovascular_data = cardiovascular$data
name = data.frame(cardiovascular_data$`All cause mortality1`, cardiovascular_data$`All cause mortality2`,
                  cardiovascular_data$...121, cardiovascular_data$...122)[3:7,]
name_correspond = data.frame(num = c(name[,1], name[,2]), treatment = c(name[,3], name[,4]))
name_correspond = na.omit(name_correspond)
name_correspond = name_correspond[order(name_correspond$num),]

set.seed(2026)
n = 20000
thetas <- netmeta:::ranksampling(cardiovascular, 
                                 nsim = n, 
                                 pooled = "common", 
                                 small.values = "good",
                                 keep.samples = TRUE)
sample = data.frame(thetas$sample, check.names = F)
colnames(sample) = name_correspond$treatment

sort(thetas$sucras, decreasing = T)


write.csv(sample, "Fretheim2012_treatment_effect.csv", row.names = F)


