library(limma)
library(edgeR)

exprs_data<- read.table(file = "logCPM_normalized_expression_matrix.txt", sep = "\t", header = TRUE, row.names = 1, stringsAsFactors = FALSE)
# set labels that corresponse to conditions

phenotype_row<- apply(phenotype_row, 2, function(x) ifelse(x > AD_score_cutoff, "AD", "Ctrl"))
condition = factor(phenotype_row, levels=c("AD","Ctrl"), labels=c("AD", "Ctrl"))

# build a contrast matrix
design<-model.matrix(~-1+condition)
colnames(design)<-c("Control","Disease")
contranst.matrix<-makeContrasts(contrasts="Disease-Control",levels=design)
# linear fit
fit<-lmFit(eset,design)
# Compute Contrasts from Linear Model Fit
fit1<-contrasts.fit(fit,contranst.matrix)
# conduct bayes test
fit2<-eBayes(fit1)

# Output result of all gene
dif<-topTable(fit2,coef="Disease-Control",n=nrow(fit2),adjust="BH")
genesymbol<-rownames(dif)
dif<-cbind(genesymbol,dif)
# write.table(dif,file="All_gene_DE_result.txt",sep='\t',quote=F,row.names=F)

# Output DE gene that satisfy lfc=log2(#) large than cutoff and FDR large than cutoff 
# log2(#) # replace to the fold change you want set
dif2<-topTable(fit2,coef="Disease-Control",n=nrow(fit2),lfc=log2(2),adjust="BH")
dif2<-dif2[dif2[,"adj.P.Val"]<0.01,] # change FDR cutoff here
genesymbol<-rownames(dif2)
dif2<-cbind(genesymbol,dif2)
write.csv(dif2,file="Limma.diff.2fold.gene.FDR0.01.csv",quote=F,row.names=F)
# End
