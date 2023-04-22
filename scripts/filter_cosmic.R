library(dplyr)




cosmic <- data.frame(fread("Census_allSun Nov 13 20_44_37 2022.csv"))

colon_cosmic <- cosmic[grepl('panc', cosmic$Tumour.Types.Somatic.),]
colon_cosmic <- cosmic

gene_mapping <- data.frame(fread("gen_v26_mapping.csv"))




gene_id_map_cosmic <- gene_mapping[gene_mapping$gene_name %in% colon_cosmic$Gene.Symbol,]$gene_id
gene_id_map_cosmic = substr(gene_id_map_cosmic,1,15)


write.table(data.frame(gene_id_map_cosmic), "cancer_drivers_id.csv", 
            col.names=FALSE, row.names = F, quote = FALSE)
