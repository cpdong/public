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
	
    return(bio_rep,assembly)
}
########################################################
# Function END
