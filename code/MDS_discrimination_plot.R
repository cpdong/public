# Code for making 2-dimension multidimentional scaling(MDS) plot
# https://rpubs.com/crazyhottommy/PCA_MDS
# also someone mention kmean for cluster, but donot know its advantage

setwd("path/to/your/file")

a<- read.csv("TCGA_BRCA_patients_regulon_activity.csv", row.names=1, header=T)

#	TCGA.A7.A0CH.11	TCGA.E9.A1RC.11	TCGA.AC.A5XU.01	TCGA.PE.A5DE.01	TCGA.PE.A5DC.01
# ARID3A	0.04	0.04	0.07	0.07	0.08
# ATF2	0.03	0.08	0.04	0.04	0.03
# ATF1	0.04	0.05	0.06	0.05	0.04
# ATF3	0.17	0.09	0.03	0.13	0.09
# BACH1	0.07	0.11	0.06	0.06	0.04

# define distance function as: euclidean/manhattan/minkowski
d<- dist(t(as.matrix(a)), method = "manhattan")

mds<- cmdscale(d, k=2)
pasub<- as.numeric(as.numeric(substr(rownames(mds), 14, 15)=="11"))

# pdf("MDS_plotting.pdf")
plot(mds[,1], mds[,2], main = "manhattan MDS", col=c("red", "blue")[factor(pasub)], 
     xlab='Dimension-1', ylab='Dimension-2',pch = 20, cex = 1)
abline(h = 0, lty = 2)
abline(v = 0, lty = 2)

dev.off()
