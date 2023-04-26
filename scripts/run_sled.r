
# install.packages("devtools") ## if not installed
# install.packages("rARPACK",dependencies = TRUE) ## if not installed

# library("devtools")
# devtools::install_github("lingxuez/sLED",dependencies = TRUE)



library("sLED")
# # library(rhdf5)
library(data.table)
library(doParallel)
library(parallel)

setDTthreads(48)

net_one <- read.table("/home/ubuntu/projects/n2v2r/data/networks/TCGA_mf_contrast/tcga_luad_sex/wgcna/wgcna_recount3_tcga_luad_purity03_normlogtpm_mintpm1_fracsamples01_tissueall_batchnull_adjnull_MALE-tumor_power10.csv",sep=",", nrows=25914, row.names=1, header = T)
print('read first')
net_two <- read.table("/home/ubuntu/projects/n2v2r/data/networks/TCGA_mf_contrast/tcga_luad_sex/wgcna/wgcna_recount3_tcga_luad_purity03_normlogtpm_mintpm1_fracsamples01_tissueall_batchnull_adjnull_FEMALE-tumor_power10.csv",sep=",",nrows=25914,row.names=1,header = T)
print('read second')

# net_one <- read.table("/home/ubuntu/projects/n2v2r/data/networks/locscn/avg_csn_ctl.csv",sep=",", row.names=1, header = T)
# print('read first')
# net_two <- read.table("/home/ubuntu/projects/n2v2r/data/networks/locscn/avg_csn_asd.csv",sep=",", row.names=1, header = T)
# print('read second')

# # net_one <- h5read("/home/ubuntu/projects/n2v2r/data/networks/TCGA_mf_contrast/tcga_luad_sex/wgcna/wgcna_recount3_tcga_luad_purity03_normlogtpm_mintpm1_fracsamples01_tissueall_batchnull_adjnull_MALE-tumor_power10.h5",'df')
# # net_two <- h5read("/home/ubuntu/projects/n2v2r/data/networks/TCGA_mf_contrast/tcga_luad_sex/wgcna/wgcna_recount3_tcga_luad_purity03_normlogtpm_mintpm1_fracsamples01_tissueall_batchnull_adjnull_FEMALE-tumor_power10.h5", 'df')


net_one_ma <- data.matrix(net_one)
net_two_ma <- data.matrix(net_two)

ptm <- proc.time()
print('start')

result_multicore <- sLED(X=net_one_ma, Y=net_two_ma, npermute=50, sumabs=0.3,  useMC=TRUE, mc.cores=48)
print(proc.time() - ptm)

gene_names = colnames(net_one)
result_genes = gene_names[which(result_multicore$leverage != 0)]

gene_leverages <- data.frame(sled_ranks=matrix(result_multicore$leverage))
rownames(gene_leverages) <- gene_names

write.csv(gene_leverages,'/home/ubuntu/projects/n2v2r/results/sled/wgcna_luad_mf_50_sumabs-03.csv', row.names=T)

