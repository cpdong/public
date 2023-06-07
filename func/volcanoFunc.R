#######################
### @author: cpdong ###
#######################

volcanoFunc<- function(data_file, logfcCol, pvalCol, fc_cutoff, pval_cutoff, genenameCol, xmin=-5, xmax=5,
                            ymax=50, cexSize=1, label_gene){
  library(ggplot2)
  library(ggrepel)
  if(grepl("\\.txt$", data_file) | grepl("\\.tsv$", data_file)){
    data<- read.csv(data_file, header=T, sep='\t')
  }else if(grepl("\\.csv$", data_file)){
    data<- read.csv(data_file, header=T)
  }else{
    print('Check your file, only accept .csv/.txt/.tsv suffix files')
  }
  if( logfcCol =="" |  pvalCol == ""){ return(NULL) 
  }else{
    data["group"]<- "insignificant"
    data[which(as.numeric(data[,pvalCol]) < as.numeric(pval_cutoff) & (as.numeric(data[,logfcCol]) > log(as.numeric(fc_cutoff), 2))),"group"] <- "increased"
    data[which((as.numeric(data[,pvalCol])< as.numeric(pval_cutoff)) & (as.numeric(data[,logfcCol]) < -log(as.numeric(fc_cutoff), 2))),"group"] <- "decreased"
    
    colors = c("grey","#D62728FF", "#1F77B4FF")
    names(colors) = c("insignificant","increased","decreased")
    labeled_genes<- intersect(strsplit(gsub(" ","",label_gene),"\n|,")[[1]], data[,genenameCol])
    data["labeled"]<- ifelse(data[,geneCol] %in% labeled_genes, "yes", "no")
    data$lgPval = -log(as.numeric(data[,pvalCol]), 10)
    data$logfcColumn = as.numeric(data[,logfcCol])
    
    options(ggrepel.max.overlaps = Inf)
    plot<- ggplot(data, aes(x = logfcColumn, y = lgPval, color = group))+
      geom_text_repel(aes(label=ifelse(labeled=="yes", data[,geneCol],'')),
                      box.padding = unit(0.35, "lines"), point.padding = unit(0.35, "lines"), force = 1,
                      colour='black',size=3) + 
      scale_colour_manual(values = colors) +
      scale_y_continuous(limits=c(0,ymax),expand=c(0,NA)) +
      scale_x_continuous(limits=c(xmin, xmax)) + 
      xlab("log2FC") + ylab(paste("-log10(",pvalCol,")", sep="")) + 
      geom_point(size=as.numeric(cexSize), alpha=1, shape=16, na.rm = T) +
      theme_bw(base_size = 14) + 
      theme(legend.position = "top",
            legend.title= element_blank(),
            panel.border = element_blank(), 
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(), 
            axis.line = element_line(colour = "black")) + 
      geom_hline(yintercept = -log(as.numeric(pval_cutoff),10), colour="#990000", linetype="dashed")+ 
      geom_vline(xintercept = log(as.numeric(fc_cutoff),2), colour="#990000", linetype="dashed") + 
      geom_vline(xintercept = -log(as.numeric(fc_cutoff),2), colour="#990000", linetype="dashed")
  return(plot)
  }
}
