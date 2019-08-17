
library('ggplot2')
library('reshape2')
library('ggpubr')

# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

setwd('C:/Users/dat/Documents/SemEval2017Task4/4B-English/PredictTopic')
# red sox','rolling stones','miss usa','twilight
label = c('iran', 'gay', 'islam', 'muslims', 'christians', 'red sox', 'rolling stones', 'miss usa', 'twilight')

f1 = as.matrix ( read.csv("muslims_pos_cor.txt",header=F,stringsAsFactors=F,sep=" ") )
f2 = as.matrix ( read.csv("muslims_neg_cor.txt",header=F,stringsAsFactors=F,sep=" ") )

f1 = f1[-4,]
f1 = f1[,-4]

f2 = f2[-4,]
f2 = f2[,-4]

label = label[-4]

make_plot = function(f1){
  colnames(f1) = label
  rownames(f1) = label
  melted_cormat <- melt( get_upper_tri(f1), na.rm = TRUE)
  plot1 = ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
    midpoint = .5, limit = c(-0.1,1), space = "Lab", 
    name="Spearman Rank Cor.") +
    theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
      size = 12, hjust = 1) , axis.title.x = element_blank(),
    axis.title.y = element_blank() )+
  coord_fixed()
  return (plot1)
}

plotf1 = make_plot(f1)
plotf2 = make_plot(f2)

ggarrange(plotf1, plotf2, 
          labels = c("Positive on Muslims", "Negative on Muslims"), vjust=1.1,
          ncol = 2, nrow = 1)

