# This code plot and test two gene correlation

setwd("C:/Users/Chuanpeng/Desktop/demo")

# Input geneNames and fileName
geneA<- "TP53"
geneB<- "BRCA1"

fileName<- "TCGA_BRCA_Norm_Tumo_Matched_114Pairs_file.txt"

#####DATA style#######
# sample t1 t2 t3 t4
# geneA  1  1  2  2
# geneB  2  1  2  2
# geneC  2  1  1  1
######################

data<- read.table(paste(fileName) , row.names=1, header=T, stringsAsFactors=FALSE)

x<- as.numeric(data[geneA, ])
y<- as.numeric(data[geneB, ])

#pdf(file = paste(geneA, geneB,"correlation_analysis.pdf", sep='_'))
  
plot(x, y, xlab=paste(geneA, " mRNA expression", sep=''), 
           ylab=paste(geneB, " mRNA expression", sep=''), pch=21,col="blue",bg="#00AAF8")
abline(lm(y ~ x))

pearsontt<- cor.test(x, y, method = "pearson")
cor<- sprintf("%.2f", pearsontt$estimate)
pearson.p<- pearsontt$p.value
spearmantt<- cor.test(x, y, method="spearman", exact=FALSE)
rho<- sprintf("%.2f", spearmantt$estimate)
spearman.p<- spearmantt$p.value

legend("topright",legend=paste(c("Correlation", "Pearson:","Spearman:"), c("",cor,rho)),box.lty=0)
#dev.off()
