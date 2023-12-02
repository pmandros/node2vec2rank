# BiocManager::install("DESeq2")


library(DESeq2)
library(readr)
library(ggplot2)
library(ggpubr)
library(dplyr)


##read data (the preprocessed rds with metadata and the gene expression used for every network)
one_fn = "data/LUAD_mVSf/recount3_tcga_luad_purity03_normlogtpm_mintpm1_fracsamples01_tissueall_batchnull_adjnull_MALE-tumor.txt"
two_fn = "data/LUAD_mVSf/recount3_tcga_luad_purity03_normlogtpm_mintpm1_fracsamples01_tissueall_batchnull_adjnull_FEMALE-tumor.txt"
whole_rds_fn = "data/LUAD_mVSf/recount3_tcga_luad_purity03_normlogtpm_mintpm1_fracsamples01_tissueall_batchnull_adjnull.rds"

one_exp <- read.table(one_fn, header = TRUE, sep = "\t", row.names = 1)
two_exp <- read.table(two_fn, header = TRUE, sep = "\t", row.names = 1)
whole_rds <- readRDS(whole_rds_fn)
rm(one_fn,two_fn,whole_rds_fn)

## remove the gene version
whole_rds_rownames <- rownames(whole_rds)
whole_rds_rownames_no_dot <-  sub("^(.*?)\\..*$", "\\1", whole_rds_rownames)
rownames(whole_rds)<-whole_rds_rownames_no_dot
rm(whole_rds_rownames,whole_rds_rownames_no_dot)

##shorten the barcodes
tcga_barcodes_small <- sub("^((?:[^-]*-){3}[^-]*)-.*$", "\\1", whole_rds$tcga.tcga_barcode)
tcga_barcodes_small <- gsub("-", ".", tcga_barcodes_small)
whole_rds$tcga.tcga_barcode <- tcga_barcodes_small
one_patient_barcodes <- colnames(one_exp)
two_patient_barcodes <- colnames(two_exp)
all_barcodes <- c(one_patient_barcodes,two_patient_barcodes)

#get the common samples (rds might have more samples)
common_elements <- intersect(rownames(one_exp),rownames(whole_rds) )
# setdiff_elem <- setdiff(genes,rownames(whole_rds) )
# print(length(setdiff_elem))

# # Get the count of common elements
# count_common <- length(common_elements)
# print(count_common)

#matching patients and genes
whole_rds_matched <- whole_rds[common_elements, whole_rds$tcga.tcga_barcode %in% all_barcodes]
rm(whole_rds)

## prepare for deseq

#get raw count matrix from rds
count_matrix <- assays(whole_rds_matched)$raw_counts
colnames(count_matrix) <- whole_rds_matched$tcga.tcga_barcode
rownames(count_matrix)<- common_elements

# variable_to_contrast <- "tcga.cgc_sample_sample_type"
variable_to_contrast <- "tcga.gdc_cases.demographic.gender"

colData = as.matrix(factor(whole_rds_matched[[variable_to_contrast]]))
rownames(colData) = colnames(count_matrix)
colnames(colData) = c(variable_to_contrast)

dds <- DESeqDataSetFromMatrix(countData = count_matrix,
                              colData = colData,
                              design= formula(paste("~",variable_to_contrast)))# Run differential expression
dds <- DESeq(dds)
# Get results
res <- results(dds)
# Summary of results
summary(res)

res$fcsign <- sign(res$log2FoldChange)
res$gene <- rownames(res)
res$absLogPadj=-log10(res$padj)
res$signedLogPadj=-log10(res$padj)/res$fcsign

DE_df <- data.frame(sign=res$fcsign, log2FoldChange=res$log2FoldChange,padj=res$padj, absLogPadj=res$absLogPadj, signedLogPadj=res$signedLogPadj,   row.names = rownames(res))
DE_for_gsea_df <- data.frame(absLogPadj=res$absLogPadj,  row.names = rownames(res))

rm(one_exp)
rm(two_exp)
rm(dds)

#
#
#

write.table(DE_df, file = "luad_mVSf_dif_deseq.txt", sep = "\t",col.names = TRUE,row.names = TRUE)
write.table(DE_for_gsea_df, file = "luad_mVSf_dif.rnk", sep = "\t",col.names = TRUE,row.names = TRUE)



## plot expression for significant DE
logcpm_df <- data.frame(t(assays(whole_rds_matched)$logtpm))
colnames(logcpm_df) <- common_elements
rownames(logcpm_df)<- whole_rds_matched$tcga.tcga_barcode

gender <- whole_rds_matched$tcga.cgc_case_gender
stage <- whole_rds_matched$tcga.gdc_cases.diagnoses.tumor_stage
merged_stages <- list()
for(i in 1:length(stage)) {       # for-loop over rows
  subject_stage <- stage[i]
  subject_stage <- gsub('stage ', '', subject_stage)
  subject_stage <- gsub('a', '', subject_stage)
  subject_stage <- gsub('b', '', subject_stage)
  subject_stage <- gsub('c', '', subject_stage)
  #create stages list
  merged_stages <-append(merged_stages,subject_stage)
  
}
logcpm_df$gender <-gender
logcpm_df$stage <- unlist(merged_stages)

logcpm_df<-logcpm_df[!is.na(logcpm_df$gender),]
logcpm_df<-logcpm_df[!is.na(logcpm_df$stage),]
rownames(logcpm_df) <- substring(rownames(logcpm_df),1,15)


gen26map <- read.table('gen_v26_mapping.csv', header = TRUE, sep = ",")
gen26map$ens_no_ver <- substring(gen26map$gene_id,1,15)

gene_to_check_name <- "ZNF267"
gene_to_check_ens <- gen26map[gen26map$gene_name == gene_to_check_name,"ens_no_ver"]
print(gene_to_check_ens)


ggplot(logcpm_df, aes(x=gender, y=ENSG00000137033)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000137033",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y=gene_to_check_name)

p <- ggboxplot(logcpm_df, x = "gender", y = "ENSG00000100030",
               color = "gender", palette = "jco",
               add = "jitter")
p + stat_compare_means()


#get cell comps
xcell_comp <- read.table('data/LUAD_mVSf/TCGA_xcell.txt', header = TRUE, sep = "\t", row.names = 1)
common_samples<-intersect(colnames(xcell_comp),rownames(logcpm_df))

xcell_subset <- xcell_comp[, common_samples, drop = FALSE]
logcpm_df_subset <- logcpm_df[common_samples,]

logcpm_df_subset$eos <- t(xcell_subset)[common_samples,"Eosinophils"]
logcpm_df_subset$baso <- t(xcell_subset)[common_samples,"Basophils"]
logcpm_df_subset$bcell <- t(xcell_subset)[common_samples,"B-cells"]
logcpm_df_subset$cd4plust <- t(xcell_subset)[common_samples,"CD4+ T-cells"]



logcpm_df_subset$basoeosbcell <- (logcpm_df_subset$eos+logcpm_df_subset$baso+logcpm_df_subset$bcell)
logcpm_df_subset$basoeosbcellmean <- (logcpm_df_subset$eos+logcpm_df_subset$baso+logcpm_df_subset$bcell)/3



# ggplot(logcpm_df_subset, aes(x=gender, y=eosinophils)) + geom_boxplot() 


p <- ggboxplot(logcpm_df_subset, x = "gender", y = "cd4plust",
               color = "gender", palette = "jco",
               add = "jitter")
p + stat_compare_means(method = "t.test") + geom_smooth(method=lm)


ggplot(logcpm_df_subset,aes(x=ENSG00000091181,y=basoeosbcellmean, col=gender))+geom_point()

p <- ggplot(logcpm_df_subset, aes(x = ENSG00000091181, y = bcell,
                                  color = gender, shape=gender))+ geom_point()+geom_smooth(method=glm)+stat_cor(method="pearson") + labs(y="Eosinophils",x="IL5RA")
p


# mutations <- read.table('data/LUAD_mVSf/tcga_luad_mutations_pivot.csv', header = TRUE, sep = ",")
# 
# mutations['Entrez_Gene_Id'] <- NULL
# 
# mutations <- t(mutations)
# colnames(mutations) <- mutations['Hugo_Symbol',]
# mutations <- mutations[-1,]
# rownames(mutations) <- substring(rownames(mutations),1,15 )
# 
# common_samples<-intersect(rownames(mutations),rownames(logcpm_df))
# logcpm_df_subset <- logcpm_df[common_samples,]
# mutations <- data.frame(mutations[common_samples,])
# mutations$gender <- logcpm_df_subset$gender
# 
# # mutations$IL5RA <- as.numeric(mutations$IL5RA)
# # mutations$SOS1 <- as.numeric(mutations$SOS1)
# # mutations$EP300 <- as.numeric(mutations$EP300)
# mutations<-as.data.frame(apply(mutations, 2, function(x) gsub('\\s+', '', x)))
# # ggplot(mutations, aes(y=IL5RA)) + geom_bar() 
# 
# 
# colnames(mutations)<-  paste("mut", colnames(mutations), sep = "")
# 
# 
# 
# exp_mut <- cbind(mutations, logcpm_df_subset)
# 
# 
# xcell_comp <- read.table('data/LUAD_mVSf/TCGA_xcell.txt', header = TRUE, sep = "\t", row.names = 1)
# 
# common_samples<-intersect(rownames(exp_mut),colnames(xcell_comp))
# 
# exp_mut <- exp_mut[common_samples,,drop = FALSE]
# xcell_eosin<- xcell_comp['Eosinophils', common_samples, drop = FALSE]
# xcell_eosin <- xcell_eosin[,rownames(exp_mut)]
# xcell_bcells<- xcell_comp['B-cells', common_samples, drop = FALSE]
# xcell_bcells <- xcell_bcells[,rownames(exp_mut)]
# 
# exp_mut$xcell_eosin <- t(xcell_eosin)
# exp_mut$xcell_bcells <- t(xcell_bcells)
# 
# 
# ggplot(exp_mut, aes(x = mutIL5RA, fill = gender)) +
#   geom_bar(position = "identity", alpha = 0.4)
# 
# 
# 
# 
# 
# 
# ggplot(exp_mut, aes(y=ENSG00000091181, x=gender, fill=factor(mutIL5RA))) +
#   geom_boxplot() 
# 
# gene_to_check_name <- "CD274"
# gene_to_check_ens <- gen26map[gen26map$gene_name == gene_to_check_name,"ens_no_ver"]
# print(gene_to_check_ens)


#EP300
counts = exp_mut %>% group_by(mutEP300, gender) %>% tally
ggplot(exp_mut, aes(y=ENSG00000100393, x=gender, fill=mutEP300)) + labs(y="EP300") +
  geom_boxplot(position=position_dodge(0.9)) +
  geom_text(data=counts, aes(label=n, y=min(exp_mut$ENSG00000100393) - 0.6), position=position_dodge(0.9)) 

#SOS1
counts = exp_mut %>% group_by(mutSOS1, gender) %>% tally
ggplot(exp_mut, aes(y=ENSG00000115904, x=gender, fill=mutSOS1)) + labs(y="SOS1") +
  geom_boxplot(position=position_dodge(0.9)) +
  geom_text(data=counts, aes(label=n, y=min(exp_mut$ENSG00000115904) - 0.6), position=position_dodge(0.9)) 

#IL5RA
counts = exp_mut %>% group_by(mutIL5RA, gender) %>% tally
ggplot(exp_mut, aes(y=ENSG00000091181, x=gender, fill=mutIL5RA)) + labs(y="IL5RA") +
  geom_boxplot(position=position_dodge(0.9)) +
  geom_text(data=counts, aes(label=n, y=min(exp_mut$ENSG00000091181) - 0.6), position=position_dodge(0.9)) 

#MSH6
counts = exp_mut %>% group_by(mutMSH6, gender) %>% tally
ggplot(exp_mut, aes(y=ENSG00000116062, x=gender, fill=mutMSH6)) + labs(y="MSH6") +
  geom_boxplot(position=position_dodge(0.9)) +
  geom_text(data=counts, aes(label=n, y=min(exp_mut$ENSG00000116062) - 0.6), position=position_dodge(0.9)) 

#PIAS4
counts = exp_mut %>% group_by(mutPIAS4, gender) %>% tally
ggplot(exp_mut, aes(y=ENSG00000105229, x=gender, fill=mutPIAS4)) + labs(y="PIAS4") +
  geom_boxplot(position=position_dodge(0.9)) +
  geom_text(data=counts, aes(label=n, y=min(exp_mut$ENSG00000105229) - 0.6), position=position_dodge(0.9)) 

#RAD51
counts = exp_mut %>% group_by(mutKRAS, gender) %>% tally
ggplot(exp_mut, aes(y=ENSG00000051180, x=gender, fill=mutKRAS)) + labs(y="PIAS4") +
  geom_boxplot(position=position_dodge(0.9)) +
  geom_text(data=counts, aes(label=n, y=min(exp_mut$ENSG00000051180) - 0.6), position=position_dodge(0.9)) 

#MAPK1
counts = exp_mut %>% group_by(mutMAPK1, gender) %>% tally
ggplot(exp_mut, aes(y=ENSG00000100030, x=gender, fill=mutMAPK1)) + labs(y="MAPK1") +
  geom_boxplot(position=position_dodge(0.9)) +
  geom_text(data=counts, aes(label=n, y=min(exp_mut$ENSG00000100030) - 0.6), position=position_dodge(0.9)) 

#SUFU
counts = exp_mut %>% group_by(mutSUFU, gender) %>% tally
ggplot(exp_mut, aes(y=ENSG00000107882, x=gender, fill=mutSUFU)) + labs(y="MAPK1") +
  geom_boxplot(position=position_dodge(0.9)) +
  geom_text(data=counts, aes(label=n, y=min(exp_mut$ENSG00000107882) - 0.6), position=position_dodge(0.9)) 

#LAMA4
counts = exp_mut %>% group_by(mutLAMA4, gender) %>% tally
ggplot(exp_mut, aes(y=ENSG00000112769, x=gender, fill=mutLAMA4)) + labs(y="LAMA4") +
  geom_boxplot(position=position_dodge(0.9)) +
  geom_text(data=counts, aes(label=n, y=min(exp_mut$ENSG00000112769) - 0.6), position=position_dodge(0.9)) 

#CDK12
counts = exp_mut %>% group_by(mutCDK12, gender) %>% tally
ggplot(exp_mut, aes(y=ENSG00000167258, x=gender, fill=mutCDK12)) + labs(y="CDK12") +
  geom_boxplot(position=position_dodge(0.9)) +
  geom_text(data=counts, aes(label=n, y=min(exp_mut$ENSG00000167258) - 0.6), position=position_dodge(0.9)) 



#IL5RA
ggplot(logcpm_df_subset, aes(x=gender, y=ENSG00000091181)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000091181",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="IL5RA")+geom_jitter(width = 0.25)

#EP300
ggplot(logcpm_df_subset, aes(x=gender, y=ENSG00000100393)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000100393",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="EP300")  +geom_jitter(width = 0.25)

#SOS1
ggplot(logcpm_df_subset, aes(x=gender, y=ENSG00000115904)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000115904",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="SOS1")+geom_jitter(width = 0.25)


#MSH6
ggplot(logcpm_df_subset, aes(x=gender, y=ENSG00000105229)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000105229",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="MSH6")+geom_jitter(width = 0.25)

#PIAS4
ggplot(logcpm_df_subset, aes(x=gender, y=ENSG00000116062)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000116062",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="PIAS4")+geom_jitter(width = 0.25)

#RAD51
ggplot(logcpm_df_subset, aes(x=gender, y=ENSG00000051180)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000051180",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="RAD51")+geom_jitter(width = 0.25)

#MAPK1
ggplot(logcpm_df_subset, aes(x=gender, y=ENSG00000100030)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000100030",'padj'][1],2)) ,x=1.5,y=4.5,size=6) + labs(y="MAPK1") +geom_jitter(width = 0.25)



#PLC2
ggplot(logcpm_df, aes(x=gender, y=ENSG00000154822)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000154822",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="PLC2")+geom_jitter(width = 0.25)

#NDC1
ggplot(logcpm_df, aes(x=gender, y=ENSG00000058804)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000058804",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="NDC1")+geom_jitter(width = 0.25)

#nup205
ggplot(logcpm_df, aes(x=gender, y=ENSG00000155561)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000155561",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="NUP205")+geom_jitter(width = 0.25)

#znf267
ggplot(logcpm_df, aes(x=gender, y=ENSG00000185947)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000185947",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="ZNF267")+geom_jitter(width = 0.25)

#RP11-809N8.4
ggplot(logcpm_df, aes(x=gender, y=ENSG00000256448)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000256448",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="RP11-809N8.4")+geom_jitter(width = 0.25)

#DNMT1
ggplot(logcpm_df, aes(x=gender, y=ENSG00000130816)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000130816",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="DNMT1")+geom_jitter(width = 0.25)


#LAM4
ggplot(logcpm_df, aes(x=gender, y=ENSG00000112769)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000112769",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="LAM4")+geom_jitter(width = 0.25)

#MED1
ggplot(logcpm_df, aes(x=gender, y=ENSG00000125686)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000125686",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="MED1")+geom_jitter(width = 0.25)

#PTPN11
ggplot(logcpm_df, aes(x=gender, y=ENSG00000179295)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000179295",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="PTPN11")+geom_jitter(width = 0.25)


#CDK12
ggplot(logcpm_df, aes(x=gender, y=ENSG00000167258)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000167258",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="CDK12")+geom_jitter(width = 0.25)

#AKT2
ggplot(logcpm_df, aes(x=gender, y=ENSG00000105221)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000105221",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="AKT2")+geom_jitter(width = 0.25)

#ILL33
ggplot(logcpm_df, aes(x=gender, y=ENSG00000137033)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000137033",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="IL33")+geom_jitter(width = 0.25)

#CD274
ggplot(logcpm_df, aes(x=gender, y=ENSG00000120217)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000120217",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="CD274")+geom_jitter(width = 0.25)

#ZNF267
ggplot(logcpm_df, aes(x=gender, y=ENSG00000185947)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000185947",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="ZNF267")+geom_jitter(width = 0.25)




ggplot(exp_mut, aes(x=gender, y=cell_comp)) + geom_boxplot() 
logcpm_df_subset$IL33plus5ra <-(logcpm_df_subset$ENSG00000091181 +logcpm_df_subset$ENSG00000137033)/2

p <- ggplot(logcpm_df_subset, aes(x = gender, y = ENSG00000120217,
               color = stage))+ geom_point()+geom_smooth(method=glm)+stat_cor(method="pearson") + labs(y="Eosinophils",x="IL33 and 5ra")
p



ggplot(logcpm_df_subset, aes(x=gender, y=ENSG00000120217)) + geom_boxplot() +   annotate(geom="text",label=paste0("padj = ", round(res["ENSG00000120217",'padj'][1],2)),x=1.5,y=4.5,size=6) + labs(y="IL33 +5RA")


compare.coeff <- function(b1,se1,b2,se2){
  return((b1-b2)/sqrt(se1^2+se2^2))
}

lm1 = lm(cell_comp ~ ENSG00000091181,data=subset(exp_mut,exp_mut$gender=="MALE"))
lm2 = lm(cell_comp ~ ENSG00000091181,data=subset(exp_mut,exp_mut$gender=="FEMALE"))

b1 <- summary(lm1)$coefficients[2,1]
se1 <- summary(lm1)$coefficients[2,2]
b2 <- summary(lm2)$coefficients[2,1]
se2 <- summary(lm2)$coefficients[2,2]
p_value = 2*pnorm(-abs(compare.coeff(b1,se1,b2,se2)))
p_value


lm.2lines <- lm(cell_comp ~ ENSG00000091181 + gender + ENSG00000091181:gender, data=exp_mut)

#get the "Estimates" automatically:
b <- coef(lm.2lines)
# Then b will have 4 estimates:
# b[1] is the estimate of beta_0: -9.0099
# b[2] is the estimate of beta_1:  1.4385
# b[3] is the estimate of beta_2: -14.5107
# b[4] is the estimate of beta_3: 1.3214

ggplot(exp_mut, aes(x = ENSG00000091181, y = cell_comp,
                   color = gender, shape=gender)) +
  geom_point(pch=21, bg="gray83") +
  #geom_smooth(method="lm", se=F) + #easy way, but only draws the full interaction model. The manual way using stat_function (see below) is more involved, but more dynamic.
  stat_function(fun = function(x) b[1] + b[2]*x, color="skyblue") + #am==0 line
  stat_function(fun = function(x) (b[1]+b[3]) + (b[2]+b[4])*x,color="orange") + #am==1 line 
  scale_color_manual(name="Transmission (am)", values=c("skyblue","orange")) +
  labs(title="Two-lines Model using mtcars data set") 

summary(lm.2lines)

ggscatter(exp_mut, x = "ENSG00000091181", y = "cell_comp", fill="gender",
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Miles/(US) gallon", ylab = "Weight (1000 lbs)")


hallmarks <- readRDS("/Users/panos/Desktop/code_for_reviwers/data_processing/hallmarks_analysis/pathways_objects/10_main_hallmarks.RDS")


write.table(hallmarks, file='test.tsv', quote=FALSE, sep='\t')

writeLines(hallmarks, "outfile.txt")

hallmakrs_df = data.frame(hallmarks)


