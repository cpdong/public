# dir=''
# filename=''
#
#
#
setwd(paste(dir))
#
#
########################################################
# R function
# Donwload online data that hidden in the html source

extract_ENCODE<- function(fileURL){

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
    # strings1<- str_match(as.character(node_content), "biological_replicates(.*?)assembly")[,2]
    strings1<- sub('.*biological_replicates', '', node_content)
    bio_rep<- substr(strings1, 4, 6)

    # Extract the Assembly information hg38 or hg19
    # strings2<- str_match(as.character(node_content), "assembly(.*?)file_type")[,2]
    strings2<- sub('.*assembly', '', node_content)
    assembly<- substr(strings2, 4, 9)

    # Extract the submitted file names
    strings3<- sub('.*submitted_file_name', '', node_content)
    strings3<- substr(strings2, 3, 200)
    submitted_file<- sub('.gz.*','',strings3)
    
    # if we use another approaches
    # page <- read_html(paste(fileURL))
    # find all nodes with a class of "listing_row_price"
    # lists <- html_nodes(page, css = '.sequence')
    # submitted_file<- html_text(lists[2])

    return(c(bio_rep,assembly,submitted_file))
}
########################################################
# Function END
#
#
# 
# invoke the definded function of ENCODE html
# source("https://raw.githubusercontent.com/cpdong/public/master/code/ENCODE_bed_select.R")
df<- read.table(paste(filename), header=T, stringsAsFactors=F)

auditStatus<- c(NA)
assemblyBy<- c(NA)
fileName<- c(NA)
for(i in 1:dim(df)[1]){
    links<- substr(df[i,1], 0, 48)
    getlinks<- extract_ENCODE(links)

    auditStatus[i]<- getlinks[1]
    assemblyBy[i]<- getlinks[2]
    fileName[i]<- getlinks[3]

    print(i)
}

df$auditStatus<- auditStatus
df$assemblyBy<- assemblyBy
df$fileName<- fileName

# audit_status megre, assembly genome
df1<- df[which((df[,2] =='1,2') & (df[,3] !='GRCh38')),]

write.table(df1, "test.txt", row.names=F, quote=F, sep='\t')
#
