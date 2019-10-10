setwd("C:/Users/Chuanpeng/Desktop/Andy")

###################### process the data into shape with Norm(# N)--AD(# N)
df1<- read.csv("merge_data_MAYO_TCX_TCX.csv", header=F, row.names=1, stringsAsFactors=F)
df2<- df1[-1,]
# find the normal sample
df_normal<- df2[, which(df2[7,] =="TCX.CONTROL")]
normN<- dim(df_normal)[2]
# find the disease sample
df_disease<- df2[, which(df2[7,] =="TCX.AD")]
diseaseN<- dim(df_disease)[2]

dataProcessed=cbind(rownames(df2), df_normal,df_disease)

write.table(dataProcessed, paste("data_MAYO_TCX_limma_feed_Norm", normN, "_disease",diseaseN,".csv", sep=''), row.names=F, sep=',', col.names=F)
#

####################### start run limma different expression analysis
library(limma)
exprs_Data<-data.frame(dataProcessed, row.names=1)
exprs_Data=expression_Data[-c(1:18), ] # remove all the 18 clinical data rows 
exprs_Data<- data.matrix(exprs_Data)

condition=factor(c(rep("N",71),rep("D",80)))
# build a contrast matrix
design<-model.matrix(~-1+condition)
colnames(design)<-c("Control","Disease")
contranst.matrix<-makeContrasts(contrasts="Disease-Control",levels=design)

# linear fit
fit<-lmFit(exprs_Data,design)
# Compute Contrasts from Linear Model Fit
fit1<-contrasts.fit(fit,contranst.matrix)
# conduct bayes test
fit2<-eBayes(fit1)

# Output result of all gene
dif<-topTable(fit2,coef="Disease-Control",n=nrow(fit2),adjust="BH")
genesymbol<-rownames(dif)
dif<-cbind(genesymbol,dif)
write.table(dif,file="All_gene_DE_result.txt",sep='\t',quote=F,row.names=F)

# Output DE gene that satisfy lfc=log2(#) large than cutoff and FDR large than cutoff 
# log2(#) # replace to the fold change you want set
dif2<-topTable(fit2,coef="Disease-Control",n=nrow(fit2),lfc=log2(2),adjust="BH")
dif2<-dif2[dif2[,"adj.P.Val"]<0.01,] # change FDR cutoff here
genesymbol<-rownames(dif2)
dif2<-cbind(genesymbol,dif2)
write.csv(dif2,file="Limma.diff.2fold.gene.FDR0.01.csv",quote=F,row.names=F)
# End
