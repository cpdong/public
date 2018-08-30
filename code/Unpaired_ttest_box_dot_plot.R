setwd("C:/Users/Chuanpeng/Desktop/demo")

# Input geneName and fileName
geneName<- "TP53"
fileName<- "TCGA_BRCA_Norm_Tumo_Matched_114Pairs_file.txt"
norm_Number<- 114
tumo_Number<- 114

#####DATA style#######
# sample n1 n2 t1 t2 t3 t4
# geneA	 1	1  2  2  3  2
# geneB	 2	1  2  3  4  2
# geneC	 2	1  1  2  1  2
######################

data<- read.table(paste(fileName) , row.names=1, header=T, stringsAsFactors=FALSE)

nor<- as.numeric(data[geneName,][1:norm_Number])
tum<- as.numeric(data[geneName,][(norm_Number+1):(norm_Number+tumo_Number)])

# define optimal ylim range
ext<- 0.2*(max(data[geneName,]) - min(data[geneName,]))

# rebuilt matrix to satisfy boxplot
exprs<- c(nor,tum)
label<- c(rep("Norm",norm_Number), rep("Tumo",tumo_Number))

# pdf("Unpaired test.pdf")
boxplot(exprs ~ label, boxwex = 0.5, ylim=c(min(data[geneName,])-ext,max(data[geneName,])+ext),
        ylab =paste(geneName, " mRNA expression", sep=''), names=c("Normal","Tumor"), outline=F)

stripchart(exprs ~ label, vertical = TRUE, method = "jitter", add = TRUE, 
                 pch = 20,cex=1, col = c('green','red'))

p<- t.test(nor, tum, paired=FALSE)$p.value
legend("top",paste("p=", round(p,3),sep=''),col=2, bty='n')

# dev.off()
