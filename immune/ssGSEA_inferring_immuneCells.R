########## A demo for running the ssgsea with given gene sets for inferring the immune cells

library("GSVA")
library("GSEABase")
# choose a immune cell gene set or make it your self as example in
# https://raw.githubusercontent.com/cpdong/public/master/immune/ImmSig_PMID27777979_ref19_24_immuneCell_signatures.csv
gene_file<- read.csv( "https://raw.githubusercontent.com/cpdong/public/master/immune/ImmSig_PMID27777979_ref19_24_immuneCell_signatures.csv", header=T)

####################### prepare the gene set gmt file format
geneSets<- GeneSet(c('g1'), setName='start')
for(i in 1:length(unique(gene_file$Cell.type))){
	cellname<- unique(gene_file$Cell.type)[i]
	geneIds<- as.character(unique(gene_file[which(gene_file$Cell.type ==cellname),"entrezgene_id"]))
	if(length(geneIds)>=5){
		geneSets<- GeneSetCollection(c(geneSets, GeneSet(geneIds, setName=cellname)))
	}
}
geneSets<- geneSets[-1]
############################################################
#
#
#
# we will use entrezgeneid to perform ssgsea
mixture_file<- "OSF_TCGA_BRCA_tumor_profile.txt"
outfile<-  "/N/u/cpdong/Carbonate/Desktop/TCGA_CCLE/gene_level_primary/ssGSEA/OSF_TCGA_BRCA_ssGSEA_result.txt"

mixture=read.table(mixture_file, sep="\t",header=T,row.names=1)#
########################################################
## change gene symbol to entrez id
anno<- read.csv("/N/u/cpdong/Carbonate/Desktop/TCGA_CCLE/gencode.v24.annotation.information.txt",header=T,sep='\t')
anno<- anno[which(anno$entrezgene_id!="NA"),c("hgnc_symbol","entrezgene_id")]
anno<- unique(anno)
rownames(anno)<- anno$hgnc_symbol

intersectNames<- intersect(rownames(anno),rownames(mixture))
#
mixture<- mixture[intersectNames,]
rownames(mixture)<- anno[intersectNames, "entrezgene_id"]
########################################################
### remove rows with 0 value large than 50%
mixture<- mixture[rowMeans(mixture!=0)>0.50, ]
#
ssGSEA_Score=gsva(data.matrix(mixture), geneSets, method='ssgsea', kcdf='Gaussian', abs.ranking=TRUE)#ssGSEA
normalize=function(x){return((x-min(x))/(max(x)-min(x)))}# normalized to 0-1 value
ssGSEA_result=normalize(ssGSEA_Score)#
ssGSEA_result=cbind(Cell.type=rownames(ssGSEA_result),ssGSEA_result)

write.csv(ssGSEA_result,file=outfile, quote=F, row.names=F)
#
