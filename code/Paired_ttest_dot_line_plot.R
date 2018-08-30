setwd("C:/Users/Chuanpeng/Desktop/demo")

# Input geneName and fileName
geneName<- "TP53"
fileName<- "TCGA_BRCA_Norm_Tumo_Matched_114Pairs_file.txt"

#####DATA style#######
# sample n1 n2 t1 t2
# geneA	 1	1  2  2
# geneB	 2	1  2  2
# geneC	 2	1  1  1
######################

data<- read.table(paste(fileName) , row.names=1, header=T, stringsAsFactors=FALSE)

# read in all numeric value
pairNum<- dim(data)[2]/2
nor<- as.numeric(data[geneName,][1:pairNum])
tum<- as.numeric(data[geneName,][(pairNum+1):(2*pairNum)])

# define optimal ylim range
ext<- 0.2*(max(data[geneName,]) - min(data[geneName,]))

# pdf("Paired_mRNA_exprs.pdf")
plot(c(),xlim=c(0.5,2.5), ylim=c(min(data[geneName,])-ext,max(data[geneName,])+ext),
         axes=F,xlab=NA,ylab=paste(geneName," mRNA expression", sep=''))
		 
for(i in 1:pairNum){
	x<-rnorm(1,1,0.08)
	y<-rnorm(1,2,0.08)
	segments(x,nor[i],y,tum[i],col=1,lwd=1)
	points(x,nor[i],pch=16,col="green4",cex=1)
	points(y,tum[i],pch=16,col=2,cex=1)
}
axis(1,at=c(1,2),labels=c("Normal","Tumor"))
axis(2)
p<- t.test(nor, tum, paired=TRUE)$p.value
text(1.5,round(max(data[geneName,])+ext),paste("p=", round(p,3),sep=''),col=2)

# dev.off()
