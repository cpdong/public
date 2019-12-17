# Input genomics profile matrix format
#          TCGA.69.7978.01 TCGA.62.8399.01 TCGA.78.7539.01 TCGA.73.4658.01
# 100130426  0.00          0.00          0.00          0.00
# 100133144  2.12          1.13          2.51          0.12
# 100134869  3.51          2.35          2.58          2.86
# 10357      6.07          6.85          5.92          6.16
#
# cutoff mean the maximum of same value of one gene across samples
# cutoff should be given as 0.25/0.5/0.75
dataFilter<- function(genomeMatrix, cutoff){
	# remove column with sd=0
	# remove gene with too much same value
	maxSameValue<- apply(genomeMatrix, 1, function(x) max(as.numeric(table(as.numeric(x)))))
	#
	genomeMatrix<- genomeMatrix[!which(maxSameValue >= round(dim(genomeMatrix)[,2]*cutoff)),]
	return(genomeMatrix)
}
