
install.packages("devtools") ## if not installed
library("devtools")
devtools::install_github("lingxuez/sLED")

source('https://raw.githubusercontent.com/xuranw/locCSN/main/Rcode/sLEDmodify.R')


log.mc.cpm.L = read.table('Velme_log_mc_cpm_L.txt')
meta.mc.L = read.table('Velme_meta_mc_L.txt')

# Let's take L4 as an example
ct.name = 'L4'
meta.mc.diag = as.numeric(meta.mc.L$diagnosis[meta.mc.L$cluster == ct.name] == 'ASD')
log.mc.L = data.matrix(log.mc.cpm.L[, meta.mc.L$cluster == ct.name])

log.mc.L[1:5, 1:5]
#          mc_L_4   mc_L_7  mc_L_10  mc_L_25  mc_L_28
#SAMD11  0.000000 0.000000 0.000000 0.000000 0.000000
#SKI     5.797950 4.036630 5.298243 0.000000 3.842033
#SLC45A1 0.000000 2.814837 0.000000 0.000000 2.269254
#RERE    6.489579 5.775307 5.702040 5.917348 5.959781
#CA6     0.000000 1.965827 0.000000 0.000000 1.894637

# rownames of expression are ASD genes
asd.genes = rownames(log.mc.L)

# 
# 
# csn.flat.temp = read.table(paste0('csn_asd_loc_flat_',ct.name, '.txt'))
# csn.flat.temp = data.matrix(csn.flat.temp)
# csn.t.flat = (csn.flat.temp > qnorm(0.99)) + 0 #Threshold at alpha = 0.01
# 
# X = csn.t.flat[, meta.mc.diag == 0]
# Y =  csn.t.flat[, meta.mc.diag == 1]
# 
# hmm=flat.to.matrix(X)
# hmm2 = flat.to.matrix(Y)


avgcsn.flat = read.table(paste0('avgcsn_asd_data_', ct.name, '.txt'))
avg.csn.ctl = flat.to.matrix(avgcsn.flat[, 1])
avg.csn.asd = flat.to.matrix(avgcsn.flat[, 2])

X_df = data.frame(avg.csn.ctl)
Y_df = data.frame(avg.csn.asd)
colnames(X_df) <- asd.genes
rownames(X_df) <- asd.genes
colnames(Y_df) <- asd.genes
rownames(Y_df) <- asd.genes

write.csv(X_df, "avg_csn_ctl.csv", row.names=TRUE) 
write.csv(Y_df, "avg_csn_asd.csv", row.names=TRUE) 




