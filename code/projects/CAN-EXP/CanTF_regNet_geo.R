# source("https://raw.githubusercontent.com/cpdong/public/master/code/projects/CanTF_regNet_tcga.R")
# regNet= regNet_geo

# This is for generating geo gpl570 and gpl96 based regNet

# The aim of this code is to transfer ChIPBase V2 reguar Matrix ID to entrez gene id, for next step analysis
# Why entrez id? it's unique than ensembl id

# Change colnames to entrezgene id
mm<- read.csv("https://raw.githubusercontent.com/cpdong/public/master/data/CAN-EXP/ChIPBaseV2_Factors_ENCODE_regNet_curated_DongCP.csv", header=T,stringsAsFactors=F)
id_match<- read.csv("https://raw.githubusercontent.com/cpdong/public/master/data/CAN-EXP/ENCODE_Factors_ID.csv", header=T,row.names=1, stringsAsFactors=F)	 
m<- mm[,-c(1,2)]

for(i in 1:dim(m)[2]){
    colnames(m)[i] <- id_match[colnames(m)[i],1]
}
m<- cbind(mm[,c(1,2)], m)

# Change rownames to entrezgene id
# source("https://bioconductor.org/biocLite.R")
# biocLite(c("biomaRt", "org.Hs.eg.db"))
library(org.Hs.eg.db)
library("biomaRt")
mart = useMart(biomart = "ensembl", dataset = "hsapiens_gene_ensembl")

genelist<- m$Ensembl_ID
result <- getBM(attributes = c( "ensembl_gene_id", "hgnc_symbol", "entrezgene"), 
            filter = "ensembl_gene_id", values = genelist,  mart = mart)

matched_m<- merge(result, m, by.x="ensembl_gene_id", by.y="Ensembl_ID", all.x=T)
mat<- matched_m[! is.na(matched_m$entrezgene),]
mat<- mat[,-c(1,2,4)]

regNet<- mat[!duplicated(mat$entrezgene),]
geo_id<- read.csv("https://raw.githubusercontent.com/cpdong/public/master/data/CAN-EXP/GPL570_ID_convert.csv", header=T, stringsAsFactors=F)

regNet_geo<- regNet[which(regNet$entrezgene %in% (intersect(regNet$entrezgene, geo_id$ENTREZ_GENE_ID))), ]
entrezgene<- regNet_geo[,1]
regNet_geo<- regNet_geo[,-1]

regNet_geo<- cbind(entrezgene, regNet_geo)

write.csv(regNet_geo, "ChIPBaseV2_regNet_GPL570.csv", row.names=F, quote=F)
#
