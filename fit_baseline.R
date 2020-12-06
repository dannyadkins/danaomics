
###
## Train baseline model to compare deep learning model against
##
## danaomics: understanding mRNA degradation using Transformers
## https://github.com/dannyadkins/danaomics
###

# setwd("C:\\Users\\Dana Udwin\\Documents\\Fall-2020\\CSCI 2952G Deep Learning in Genomics\\Final Project\\danaomics")

library(dplyr)
library(jsonlite)
library(monier)
library(stringr)
library(onehot)
library(e1071)


# -- Read in data


# assumes train.json in same directory as this script
# data comes from https://www.kaggle.com/c/stanford-covid-vaccine/
train <- stream_in(file('train.json'))



# -- Format input

# one-hot encode

# DNA sequence
# 107 bases --> 107*4=428 columns

train.sequences <- sapply(train[, "sequence"], oneHot) %>% t()  # matrix format
colnames(train.sequences) <- paste0(rep(paste0("base", 
                                               1:(ncol(train.sequences)/4)), 
                                        each=4), 
                                    "=",
                                    c("A", "C", "G", "I"))  # default oneHot order
rownames(train.sequences) <- 1:nrow(train.sequences)

# DNA structure
train.structure <- sapply(train[, "structure"], 
                          function(structure) { 
                            str.vec <- strsplit(structure, "")[[1]]
                            str.vec
                            }) %>% t()
train.structure.onehot <- data.frame(train.structure)
names(train.structure.onehot) <- paste0("structure", 1:ncol(train.structure.onehot))
rownames(train.structure.onehot) <- 1:nrow(train.structure.onehot)
train.structure.onehot.dat <- lapply(train.structure.onehot, function(col) {
  factor(col, levels = c(".", ")", "("))
})
train.structure.onehot.dat <- as.data.frame(train.structure.onehot.dat)
encoder <- onehot(train.structure.onehot.dat)
train.structure.onehot.encoded <- predict(encoder, train.structure.onehot.dat) # matrix format

# Loop type
train.loop <- sapply(train[, "predicted_loop_type"], 
                          function(loop) { 
                            loop.vec <- strsplit(loop, "")[[1]]
                            loop.vec
                          }) %>% t()
train.loop.onehot <- data.frame(train.loop)
names(train.loop.onehot) <- paste0("loop", 1:ncol(train.loop.onehot))
rownames(train.loop.onehot) <- 1:nrow(train.loop.onehot)
train.loop.onehot.dat <- lapply(train.loop.onehot, function(col) {
  factor(col, levels = c("S", "M", "I", "B", "H", "E", "X"))
})
train.loop.onehot.dat <- as.data.frame(train.loop.onehot.dat)
encoder <- onehot(train.loop.onehot.dat)
train.loop.onehot.encoded <- predict(encoder, train.loop.onehot.dat) # matrix format
rownames(train.loop.onehot.encoded) <- 1:nrow(train.loop.onehot.encoded)

# -- Format output

output.vars <- c("reactivity", 
                 "deg_Mg_pH10", 
                 "deg_pH10", 
                 "deg_Mg_50C", 
                 "deg_50C")

for (output.var in output.vars) {
  output.temp <- as.data.frame(lapply(train[[output.var]], unlist)) %>% t()
  colnames(output.temp) <- paste0(output.var, "_base", 1:ncol(output.temp))
  rownames(output.temp) <- 1:nrow(output.temp)
  
  assign(paste0("train.", output.var), 
         output.temp)
}



# assemble all inputs and outputs

train.mat.temp <- cbind(train.sequences, train.structure.onehot.encoded)
train.mat <- cbind(train.mat.temp, train.loop.onehot.encoded) # 2400 x 1498

train.out.temp1 <- cbind(train.reactivity, train.deg_Mg_pH10)
train.out.temp2 <- cbind(train.out.temp1, train.deg_pH10)
train.out.temp3 <- cbind(train.out.temp2, train.deg_Mg_50C)
train.out <- cbind(train.out.temp3, train.deg_50C) # 2400 x 340


# split into train and validation
train.prop <- 0.7
train.idx <- sample(1:nrow(train.mat), 
                    size=floor(train.prop*nrow(train.mat)))
train.x <- train.mat[train.idx,]
train.y <- train.out[train.idx,]
val.x <- train.mat[-train.idx,]
val.y <- train.out[-train.idx,]



# for each independent output (1 of 5 measurements, for each of 
#   68 bases), train SVM. collect predictions on validation set.

val.pred.y <- apply(train.y, 2, function(col) {
  mod <- svm(train.x, col)
  predict(mod, val.x)
})

# write.csv(val.pred.y, "val_svm_pred.csv", row.names=F)
# write.csv(val.y, "val_truth.csv", row.names=F)

# get MSE
# 0.8594025
sum((val.y-val.pred.y)**2)/(ncol(val.y)*nrow(val.y)) 

# get MSE by output type
# reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
# 0.4416079 0.3824822 1.6505779 0.6440313 1.1783130
sapply(1:5, function(output) {
  idx <- (1+((output-1)*68)):(68*output)
  sum((val.y[,idx]-val.pred.y[,idx])**2)/(ncol(val.y[,idx])*nrow(val.y[,idx]))
})










# now that we've done that, experiment with different settings
# (1) scale y to mean 0 sd 0.5, or leave raw
# (2) use only sequencing data, or use sequencing + structure + loop type
# 
# also, collect MSE on training set AND MSE on validation set

results.table <- matrix(NA, nrow=8, ncol=4)
results.table <- data.frame(results.table)
names(results.table) <- c("mse", "raw.y", "all.covars", "mse.type")

i <- 1 # iter

for (raw.y in c("Y", "N")) {
  for (all.covars in c("Y", "N")) {
    
    # 2 rows x 68*5=340 columns for each of 340 targets
    # first row gives train MSE, second row gives validation MSE
    results <- sapply(1:ncol(train.y), function(col) {
      
      if (raw.y == "Y") {
        train.y.temp <- train.y[, col]
        val.y.temp <- val.y[, col]
      } else {
        train.y.temp <- (train.y[, col] - mean(train.y[, col])) / sd(train.y[, col])
        val.y.temp <- (val.y[, col] - mean(val.y[, col])) / sd(val.y[, col])
      }
      
      if (all.covars == "Y") {
        train.x.temp <- train.x
        val.x.temp <- val.x
      } else {
        # recall X matrix is assembling by cbinding the three different input types
        #   together, and sequence data is leftmost
        train.x.temp <- train.x[, 1:ncol(train.sequences)]
        val.x.temp <- val.x[, 1:ncol(train.sequences)]
      }
      
      mod <- svm(train.x.temp, train.y.temp)
      train.mse <- mean((train.y.temp - fitted(mod))**2)
      
      res <- predict(mod, val.x.temp)
      val.mse <-  mean((val.y.temp - res)**2)
      
      return(c(train.mse, val.mse))
    })
    
    # 1 row x 2 columns
    # first column gives train MSE averaged over all targets, 
    # second column gives validation MSE averaged over all targets
    results.avg <- rowMeans(results)
    
    # save results
    results.table[i, ] <- c(results.avg[1], raw.y, all.covars, "train")
    results.table[i+1, ] <- c(results.avg[2], raw.y, all.covars, "val")
    
    # print(results.table)
    
    i <- i+2
  }
}

write.csv(results.table, "svm_all_settings_results.csv", row.names=F)

results.table.rounded <- results.table
results.table.rounded$mse <- round(as.numeric(results.table.rounded$mse), 2)
write.csv(results.table.rounded, "svm_all_settings_results_rounded.csv", row.names=F)


