setwd("C:\\Users\\Chuanpeng\\Desktop\\demo")
aa<- read.csv("aa.csv", row.names=1, header=T)
library(pheatmap) 

clinic<- read.csv("https://raw.githubusercontent.com/cpdong/public/master/data/CAN-EXP/BRCA_Xena_clinicalMatrix_20180910_simple_Usage_NA.csv", row.names=1, header=T)

anno_col<- clinic[colnames(aa), ]
anno_col1 = data.frame(PAM50 = factor(anno_col[, c(1)]))
rownames(anno_col1)<- rownames(anno_col)
anno_color = list(PAM50 = c(Basal = "deeppink", BLANK="white",Her2="blue", LumA="forestgreen", LumB="gold",Normal="grey"))

# pdf("test5_PAM50.pdf")
my_palette <- colorRampPalette(c("green4", "green3","green2", "green1", "green","black","red","red1", "red2","red3", "red4"))(n = 299)
pheatmap(aa, col=my_palette, scale="row", cluster_row = T, cluster_col=T, annotation_col=anno_col1, annotation_colors = anno_color, show_colnames = F,
         cellwidth = 0.2, cellheight = 0.8, border_color=NA, width =10, height=10)
# dev.off()
