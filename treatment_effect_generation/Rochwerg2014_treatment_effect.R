library(nmadb)
library(netmeta)

# need package netmeta in version 2.9.0
sepsis <- runnetmeta(480931, model = "fixed")

save.image(file = "480931.RData")
rm(list=ls())

# need package netmeta in new version
load("480931.RData")


# p -value
netrank(sepsis, small.values = "good")

set.seed(2026)
n = 20000
thetas <- netmeta:::ranksampling(sepsis, 
                                 nsim = n, 
                                 pooled = "common", 
                                 small.values = "good",
                                 keep.samples = TRUE)
sample = data.frame(thetas$sample, check.names = F)

sort(thetas$sucras, decreasing = T)

colnames(sample)[which(colnames(sample) == "1")] = "Saline"
colnames(sample)[which(colnames(sample) == "2")] = "Albumin"
colnames(sample)[which(colnames(sample) == "3")] = "Heavystarch"
colnames(sample)[which(colnames(sample) == "4")] = "Gelatin"
colnames(sample)[which(colnames(sample) == "5")] = "Balancedcrystalloid"
colnames(sample)[which(colnames(sample) == "6")] = "Lightstarch"

write.csv(sample, "Rochwerg2014_treatment_effect.csv", row.names = F)
