install.packages('BiocManager')

library(WGCNA)
library(flashClust)
library(doParallel)
library(data.table)
library("genefilter")
registerDoParallel(cores=7)
library(ggplot2)


tcga_xprs <-data.frame(read.csv("expression/xprs_panc_tcga_snail.tsv",header = TRUE, 
                      row.names = 1,sep = "\t"))
gtex_xprs <- data.frame(read.csv("expression/xprs_panc_gtex_snail.tsv",header = TRUE, 
                      row.names = 1,sep = "\t"))

tcga_xprs_t <-transpose(tcga_xprs)
colnames(tcga_xprs_t) <- rownames(tcga_xprs)
rownames(tcga_xprs_t) <- colnames(tcga_xprs)

gtex_xprs_t <-transpose(gtex_xprs)
colnames(gtex_xprs_t) <- rownames(gtex_xprs)
rownames(gtex_xprs_t) <- colnames(gtex_xprs)

vars_tcga <- apply(tcga_xprs, 1, var)
vars_gtex <- apply(gtex_xprs, 1, var)

filtered_tcga <- tcga_xprs[vars_tcga >= quantile(vars_tcga,0.5),]
filtered_gtex <- gtex_xprs[vars_gtex >= quantile(vars_gtex,0.5),]


common_index <- intersect(row.names(filtered_tcga),row.names(filtered_gtex))
filtered_tcga<- filtered_tcga[common_index,]
filtered_gtex<- filtered_gtex[common_index,]


spt_tcga <- pickSoftThreshold(filtered_tcga) 

par(mar=c(1,1,1,1))
plot(spt_tcga$fitIndices[,1], spt_tcga$fitIndices[,5],
     xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
     main = paste("Mean connectivity"))
text(spt_tcga$fitIndices[,1], spt_tcga$fitIndices[,5], labels= spt_tcga$fitIndices[,1],col="red")

adjacency_tcga <- round(adjacency(t(filtered_tcga),power = 16),3)

spt_gtex <- pickSoftThreshold(filtered_gtex) 

par(mar=c(1,1,1,1))
plot(spt_gtex$fitIndices[,1], spt_gtex$fitIndices[,5],
     xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
     main = paste("Mean connectivity"))
text(spt_gtex$fitIndices[,1], spt_gtex$fitIndices[,5], labels= spt_gtex$fitIndices[,1],col="red")


adjacency_gtex <- round(adjacency(t(filtered_gtex),power = 16),3)

write.csv(adjacency_tcga,"networks/wgcna_PAAD_TCGA.csv", row.names = TRUE)
write.csv(adjacency_gtex,"networks/wgcna_PANC_GTEX.csv", row.names = TRUE)



sum(tcga_flat_df==0)
sum(gtex_flat_df==0)

tcga_df <- data.frame(read.csv("networks/wgcna_PAAD_TCGA.csv",header = TRUE, 
                                    row.names = 1,sep = ","))

gtex_df <- data.frame(read.csv("networks/wgcna_PANC_GTEX.csv",header = TRUE, 
                                    row.names = 1,sep = ","))

tcga_diag <- diag(as.matrix(tcga_df))
gtex_diag <- diag(as.matrix(gtex_df))

tcga_flat_df <- data.frame(weight_dist=as.vector(t(tcga_df)))
gtex_flat_df <- data.frame(weight_dist=as.vector(t(gtex_df)))




idx_tcga<-which(colSums(tcga_df)==tcga_diag)
idx_gtex<-which(colSums(gtex_df)==gtex_diag)

common_index <-intersect(idx_tcga,idx_gtex)


tcga_flat_df_pos <- data.frame(tcga_flat_df[idx,])
colnames(tcga_flat_df_pos) <- c("pos_weight")

ggplot(tcga_flat_df_pos, aes(x=pos_weight)) + geom_density()



