# Code for making 2-dimension PCA plot

setwd("path\to\your\file")

library("factoextra")

a<- read.csv("TCGA_BRCA_patients_regulon_activity.csv", row.names=1, header=T)

#	TCGA.A7.A0CH.11	TCGA.E9.A1RC.11	TCGA.AC.A5XU.01	TCGA.PE.A5DE.01	TCGA.PE.A5DC.01
# ARID3A	0.04	0.04	0.07	0.07	0.08
# ATF2	0.03	0.08	0.04	0.04	0.03
# ATF1	0.04	0.05	0.06	0.05	0.04
# ATF3	0.17	0.09	0.03	0.13	0.09
# BACH1	0.07	0.11	0.06	0.06	0.04

dat<-as.matrix(t(a))
pca<- prcomp(dat,scale = TRUE, center = TRUE)
pcaDATA <- pca$x[,1:2]

pdf("PCA_analysis.pdf")

# get contribution of PCs
fviz_screeplot(pca, addlabels = TRUE, ylim = c(0, 50))
fviz_screeplot(pca, addlabels = TRUE, ylim = c(0, 50)) ->contribution

# add patient labeling
pasub<- as.numeric(as.numeric(substr(rownames(pcaDATA), 14, 15)=="11"))
plot(pcaDATA[,1], pcaDATA[,2], pch=20, col=c("red", "blue")[factor(pasub)], 
     xlab=paste("Dim1(", round(contribution$data[1,2],1), "%)", sep=''), 
	 ylab=paste("Dim2(",round(contribution$data[2,2],1), "%)", sep=''))
abline(h = 0, lty = 2)
abline(v = 0, lty = 2)
#

# make 3-dimension PCA plot
library(rgl)
plot3d(pca$x[,1:3], col=c("red", "blue")[factor(pasub)],  type='s', size=1)
# please rotate the 3-D figure by yourself

dev.off()
