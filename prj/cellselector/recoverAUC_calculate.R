# run recovery curve enrichment analysis
# ranks--- the rank file of different expression fold change
# rank_order--should be Increase or Decrease order
# the target genesets to be enriched
# cutoff represent the top 1k or 4K gene
# cutoff should be integer or percentage of all gene

recoverAUC<- function (rank_file, rank_order, geneSet, cutoff){
	if(rank_order=="increase" |rank_order=="+"){
		ranks<- rank_file[order(rank_file[,2], decreasing =FALSE), ]
		ranks[,3]<- c(1:dim(ranks)[1])
		colnames(ranks)<- c("gene", "foldChange", "rank")
	}else{
		ranks<- rank_file[order(rank_file[,2], decreasing =TRUE), ]
		ranks[,3]<- c(1:dim(ranks)[1])
		colnames(ranks)<- c("gene", "foldChange", "rank")
	}
	# judge whether the cutoff was given percentage or interger value
	if(cutoff%%1==0){
		cutoff<- cutoff
	}else{
		cutoff<- round(cutoff *dim(ranks)[1])
	}
	#	
	rank_temp<- ranks[which(ranks$gene %in% geneSet),]
	geneSet_size<- dim(rank_temp)[1]
	rank_auc_data<- ranks[which(ranks$rank <= cutoff),]
	geneSet_cutoff_size<- dim(rank_auc_data)[1]
	normalized_value<- length(geneSet)*dim(ranks)[1]
	#
	recoverAUC<- (cutoff*dim(rank_auc_data)[1] - sum(rank_auc_data$rank))/normalized_value
	return(c(recoverAUC,geneSet_size, geneSet_cutoff_size))
}
