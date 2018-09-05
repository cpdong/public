# Usage example of this function
# source("https://raw.githubusercontent.com/cpdong/public/master/code/Extract_figshare_download_link.R")
# library(data.table)
# figlink<- download_link("https://figshare.com/s/ea86f50e6df67e81b873")
# data<- fread(figlink)

########################################################
# R function
# Donwload online data that hidden in the html source
# This is special for Figshare weblinks

download_link<- function(fileURL){

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
    node_content<- xpathSApply(doc,"//script",xmlValue)[2]

    # Discriminate whether strings contain: http://r.789695.n4.nabble.com/Match-beginning-and-end-of-string-grepl-td4696363.html
    # cc<- grepl("^.*previewLocation.*preview.*$",cont)

    # Extract the substrings: https://stackoverflow.com/questions/39086400/extracting-a-string-between-other-two-strings-in-r
    part1<- str_match(as.character(node_content), "baseUrl\": \"(.*?)/preview")[2]
    part2<-  str_match(as.character(node_content), "token\": \"(.*?)\",\n")[2]

    download_link<- paste(part1, "?private_link=", part2, sep='')

    return(download_link)
}
########################################################
# Function END
