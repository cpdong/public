library(survival)

# Declaration survival function formula
pKMSurv <- function (surv, ind_surv, classification) {
  ss <- survdiff(Surv(surv, ind_surv) ~ classification)
  pValue <- 1-pchisq(ss$chisq, 1)
  return (pValue)
}

pCoxSurvUni <- function (surv, ind_surv, classification) {
  cc<-coxph(Surv(surv, ind_surv) ~ classification)
  pValue <- as.numeric(summary(cc)$coefficient[5])
  hr <- as.numeric(summary(cc)$coefficients[2])
  lower <- as.numeric(summary(cc)$conf.int[3])
  upper <- as.numeric(summary(cc)$conf.int[4])
  return (c(pValue, hr, lower, upper))
}

####################
# begin to run
####################
# d1<- read.csv("TCGA_BRCA_patients_regulon_activity_only_tumor.csv", header=T, row.names=1, stringsAsFactors=F)
# d2<- read.csv("https://raw.githubusercontent.com/cpdong/public/master/data/CAN-EXP/BRCA_Xena_clinicalMatrix_20180910.csv",header=T, row.names=1, stringsAsFactors=F)

d2<- d2[, c("OS.time", "OS")]

data<- merge(t(d1), d2, by="row.names", all.x=T)
# data<-read.csv("dataAll.csv", stringsAsFactors=F)

result<-data.frame(Gene=c(NA), OS_median_km=c(NA), OS_median_cox_u_p=c(NA), OS_median_cox_u_hr=c(NA),
                    OS_median_cox_u_lower=c(NA), OS_median_cox_u_upper=c(NA))

newdata<-data[which(data$OS.time != "NA"), ] 

attach(newdata)
geneNum <- dim(newdata)[2] -2

sur<-data.frame("OS"=as.numeric(OS.time), "OS_IND"=as.numeric(OS), stringsAsFactors=F)
detach(newdata)

index<-1
for (i in 2:geneNum) {
  attach(newdata)
  geneName <- names(newdata)[i]
  geneexp<- as.numeric(newdata[,i])
  rr<-summary(as.numeric(newdata[,i]))
  median <- as.numeric(rr[3])  
  
  medianB <- as.numeric(as.numeric(newdata[,i])>=median)
  #l <- as.numeric(newdata[,i])
  
  calculate<-data.frame("Gene"=newdata[,i], medianB, geneexp, sur, stringsAsFactors=F)
  detach(newdata)
  
  # Keeping to figure if this gene RNAseq was normal to Run
  if(!mean(medianB)==medianB[1]){
    
    medianOs <- pKMSurv (calculate$OS, calculate$OS_IND, calculate$medianB)
    medianOsCoxUp <- pCoxSurvUni (calculate$OS, calculate$OS_IND, calculate$geneexp)[1]
    medianOsCoxUhr <- pCoxSurvUni (calculate$OS, calculate$OS_IND, calculate$geneexp)[2]
    medianOsCoxUlow <- pCoxSurvUni (calculate$OS, calculate$OS_IND, calculate$geneexp)[3]
    medianOsCoxUupp <- pCoxSurvUni (calculate$OS, calculate$OS_IND, calculate$geneexp)[4]
    
  } else {
    
    medianOs <- "NA"
    medianOsCoxUp <- "NA"
    medianOsCoxUhr <- "NA"
    medianOsCoxUlow <- "NA"
    medianOsCoxUupp <- "NA"
  } 
  
  result[i-1, 1] <- geneName
  result[i-1, 2] <- medianOs
  result[i-1, 3] <- medianOsCoxUp
  result[i-1, 4] <- medianOsCoxUhr
  result[i-1, 5] <- medianOsCoxUlow
  result[i-1, 6] <- medianOsCoxUupp
  
  if(i==1000*index) {
    print(i)
    index<-index+1
  } 
}
# write.csv(result, file = "TCGA_BRCA_regulon_OS_continuous_KM_cox_u_result.csv", row.names = F, quote = F)
