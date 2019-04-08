# dir=''
# filename=''
#
#
#
setwd(paste(dir))
df<- read.table(paste(filename), header=T, stringsAsFactors=F)
########################################################
# R function
# Donwload online data that hidden in the html source

audit_status<- function(fileURL){

    # Get download link from http links
    ####################################################
    library(stringr)
    library(rvest)
    library(XML)
    library(RCurl)

    link <- getURL(fileURL,.encoding='utf-8')
    doc <-htmlParse(link)
    nodess<- getNodeSet(doc, "//script")

    # Get NODE content with XML
    node_content<- xpathSApply(doc, "//script",xmlValue)[3]

    # Extract the substrings: https://stackoverflow.com/questions/39086400/extracting-a-string-between-other-two-strings-in-r
    strings1<- str_match(as.character(node_content), "biological_replicates(.*?)assembly")[,2]
    bio_rep<- str_extract_all(strings1,"\\(?[0-9,.]+\\)?")[[1]][1]

    # Extract the Assembly information hg38 or hg19
    strings2<- str_match(as.character(node_content), "assembly(.*?)file_type")[,2]
    assembly<- str_extract_all(strings2,"\\(?[0-9,.]+\\)?")[[1]][1]
	
    return(c(bio_rep,assembly))
}
########################################################
# Function END
#
#
# 
# invoke the definded function of ENCODE html
# source("https://raw.githubusercontent.com/cpdong/public/master/code/ENCODE_bed_select.R")
# df<- read.table("files.txt", header=T, stringsAsFactors=F)

auditStatus<- c(NA)
assemblyBy<- c(NA)
for(i in 1:dim(df)[1]){
    links<- substr(df[i,1], 0, 48)
    getlinks<- audit_status(links)

    auditStatus[i]<- getlinks[1]
    assemblyBy[i]<- getlinks[2]

    print(i)
}

df$auditStatus<- auditStatus
df$assemblyBy<- assemblyBy

write.csv(df, "test.csv", row.names=F, quote=F)
#
