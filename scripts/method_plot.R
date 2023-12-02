library(igraph)
library(ggplot2)
library(ggraph)

# Read the adjacency matrix from CSV file
adj_matrix <- read.csv("graph_one.csv", header = FALSE)

# Convert adjacency matrix to an igraph object
graph_one <- graph_from_adjacency_matrix(as.matrix(adj_matrix), mode = "undirected")

coords_one = layout_with_fr(graph_one)


# Add node names from 1 to n
V(graph)$name <- 1:vcount(graph_one)


# Read the adjacency matrix from CSV file
adj_matrix <- read.csv("graph_two.csv", header = FALSE)

# Convert adjacency matrix to an igraph object
graph_two <- graph_from_adjacency_matrix(as.matrix(adj_matrix), mode = "undirected")

coords_two = layout_with_fr(graph_two)

coords_two[which(V(graph_two)$name %in% V(graph_one)$name), ] <- 
  coords_one[which(V(graph_one)$name %in% V(graph_two)$name),]
xlim <- range(c(coords_one[,1], coords_two[,1]))
ylim <- range(c(coords_one[,2], coords_two[,2]))

# Add node names from 1 to n
V(graph_two)$name <- 1:vcount(graph)


# Plot the graph with a community layout
p_first <-ggraph(graph_one, layout = coords_one, xlim=xlim, ylim=ylim, rescale=FALSE) +
  geom_edge_fan( color = 'gray', width=0.2) +
  scale_edge_color_manual(values = "gray") +
  geom_node_point(fill = "#2c7bb6", 
                  colour = "black", 
                  size =3,
                  shape = 21) +
  # geom_node_text(aes(label = name), vjust = -1) +
  theme_void() 

p_first


# Plot the graph with a community layout
p_second<-ggraph(graph_two, layout = coords_two, xlim=xlim, ylim=ylim, rescale=FALSE) +
  geom_edge_fan( color = 'gray',width = 0.2) +
  geom_node_point(fill = "#d7191c", 
                  colour = "black", 
                  size =3,
                  shape = 21)+
  # geom_node_text(aes(label = name), vjust = -1) +
  theme_void() 

p_second


ggsave(filename="control_graph_small_test.pdf",plot=p_first,width=3,height=2,scale=1)

ggsave(filename="case_graph_small_test.pdf",plot=p_second,width=3,height=2,scale=1)


first_UASE <- read.csv("graph_one_UASE_muaha.csv", header = FALSE)
second_UASE <- read.csv("graph_two_UASE_muaha.csv", header = FALSE)

data <-rbind(first_UASE, second_UASE)
colnames(data) <- c('Dim1','Dim2')

vector <- 1:nrow(first_UASE)
appended_vector <- c(vector, vector)
data$label = appended_vector

# selected_indices <- c(10, 110, 20, 120, 30,130, 40, 140, 50, 150, 60, 160, 70, 170, 80, 180, 90, 190, 100, 200)
# selected_data <- data[selected_indices, ]


data$nodes <- ifelse(seq_len(nrow(first_UASE)*2) <= nrow(first_UASE), "control","case")

# # Create the scatter plot
# p_scatter<-ggplot(data, aes(Dim1, Dim2)) +
#   geom_point(aes(color = nodes), shape=21, size = 3) +
#   geom_line(aes(group = label), linewidth=0.4) +
#   # geom_text(data = selected_data, aes(label = label), nudge_y = 0.05) +
#   scale_color_manual(values = c( "#d7191c","#2c7bb6")) +
#   theme_classic() +
#   theme(axis.text=element_text(size=10),axis.title=element_text(size=10))
# 
# p_scatter

# ggsave(filename="scatter_smal_testhmm.pdf",plot=p_scatter,width=4.5,height=3)




emb<-ggplot(data, aes(x=Dim1, y=Dim2)) + 
  geom_point(color='black', shape=21, size=3, aes(fill=nodes)) + 
  scale_fill_manual(values=c('#d7191c', '#2c7bb6')) +
  geom_line(aes(group = label), linewidth=0.05) +  theme_classic() 
  # geom_text(data = selected_data, aes(label = label), nudge_y = 0.05) +

emb

ggsave(filename="emb.pdf",plot=emb,width=4.5,height=3,scale=1)





