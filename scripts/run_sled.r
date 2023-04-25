# install.packages("devtools") ## if not installed
# library("devtools")
# devtools::install_github("lingxuez/sLED")


library("sLED")

net_one <- read.csv("/Users/panos/git/n2v2r/data/networks/locscn/avg_csn_ctl.csv", header = T,row.names=1)
net_two <- read.csv("/Users/panos/git/n2v2r/data/networks/locscn/avg_csn_asd.csv", header = T,row.names=1)

net_one_ma <- data.matrix(net_one)
net_two_ma <- data.matrix(net_two)

result <- sLED(X=net_one_ma, Y=net_two_ma, npermute=50, useMC=TRUE, mc.cores=2)
