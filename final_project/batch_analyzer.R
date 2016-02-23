# The "Classify POIs from the Enron Scandal" classifier analyzer
#
# This R script reads performance data written by a batch run of multiple 
# machine learning classifiers and offers a few functions to display the
# data in ggplot2 style
# 
# You can use from your Rmd file:
# 
# Variables:
# all.ranked - all batch runs with configurations and metrics
# best.ranked - only the best performing runs of each classifier
# bestest.ranked - if multiple runs performed best from F1 score,
#   show them only with one observation
# featurescores.ranked - the feature scores for each run with automatic
#   feature selection
# featurescores.display - like featurescores.ranked, but with dummy data to show
#   the algorithm symbol
# clf.levels - the classifier ranking
# clf.labels - like clf.levels, but with word wrap for more economic display
# featurescores.levels - the automatically selected features, ordered by
#   mean feature score
#
# Functions:
# plot_table(all.ranked, TRUE)
# plot_table(bestest.ranked, FALSE)
# plot_performance(all.ranked)
# plot_runtime(all.ranked)
# plot_features(featurescores.display, clf.levels)


library(dplyr)
library(ggplot2)
library(tidyr)
library(scales)
library(gridExtra)
library(zoo)


# In Python logical values are expressed (and serialized to CSV using)
# the notation True/False instead of TRUE/FALSE, the following gives us a type
# "python.logical", which can be serialized from such a string
setClass("python.logical")
setAs("character", "python.logical", function(from) return(from == "True"))

# Helper function; takes a vector of variables and returns a value
# if all the elements are the same
merge_vector <- function(x) return(if (all(x[1] == x)) x[1] else NA)


# ---------------
# STEP 1 
# Reading the data about the classifier performance
# ---------------

# The CSV file "poi_id_clf_ids" must contain the name of all classifiers
clf.ids.file = "poi_id_clf_ids.csv"
clf.ids.col.names = c("clf.id","clf.name") 
clf.ids.col.types = c("numeric","character")
clf.ids <- read.csv(clf.ids.file, header=FALSE,
                    col.names=clf.ids.col.names,
                    colClasses=clf.ids.col.types)

# The CSV file "poi_id_batch.csv" must contain the batch data of all 
# classifiers
batch.file = "poi_id_batch.csv"
batch.col.names <- c("feature.scaling", "feature.selection", "clf.id",
                    "f1", "precision", "recall", "accuracy", "runtime")
batch.col.types <- c("factor", "python.logical", "numeric",
                    "numeric", "numeric", "numeric", "numeric", "numeric")
batch <- read.csv(batch.file, header=FALSE,
                  col.names=batch.col.names,
                  colClasses=batch.col.types)

# Let's add the rows/configurations to "batch" that were not recorded 
# because the batch run timed out. 
# First, this data frame will contain every possible combination
clf.each.configs <- expand.grid(feature.scaling=c("off","on","pca"),
                                feature.selection=c(F,T))
# We also need a column which represents this cross-product as integer
clf.each.configs$config.id <- factor(seq.int(nrow(clf.each.configs))-1)
# We multiply all potential configurations with all potential classifiers
clf.all.configs <- clf.ids %>% merge(clf.each.configs)
# Left joining all classifier-configuration combinations with batch data
# should add the missing rows
all <- clf.all.configs %>% 
    left_join(batch, by=c("clf.id","feature.scaling","feature.selection"))

# Add column to see if precision & recall > 0.3 (need that for sorting)
all$gt.0.3 = (all$precision > 0.3 & all$recall > 0.3)

# For each classifier, find best configuration (configuration with highest)
# F1 value (therefore the summarize(f1=max(...))); then sort them with
# the classifier with precision & recall > 0.3 on top and descending by
# F1 score, and, if identical, by algorithm runtime.
best.ranked <- all %>%
    group_by(clf.id) %>%
    summarise(f1=max(f1, na.rm=T)) %>%
    left_join(all, by=c("clf.id","f1")) %>%
    arrange(desc(gt.0.3), desc(f1), runtime)

# The last statement implicitly ranks our clf.ids / clf.names by performance
# We can create a factor out of this order to more easily sort by that
# ranking in the future.
clf.levels <- unique(best.ranked$clf.name)
# Outputting this factor on the screen produces quite long lines.
# The following regular expression puts a line feed at the space behind the
# 15th character of each label
clf.labels <- gsub("^(.{,15})([^ ]*) (.*)$","\\1\\2\n\\3", clf.levels)

# From this ranked table, we can "transport" the factor back to the "all"
# dataframe and the clf.ids dataframe, so it is available there later
best.ranked$clf.factor <- factor(best.ranked$clf.name, levels=clf.levels)
all$clf.factor <- factor(all$clf.name, levels=clf.levels)
clf.ids$clf.factor <- factor(clf.ids$clf.name, levels=clf.levels)
all$clf.factor <- factor(all$clf.name, levels=clf.levels)

# With the new factor, we can finally re-arrange our table
all.ranked <- all %>% arrange(clf.factor, desc(f1), runtime)

# "best.ranked" might still mention some classifiers multiple times, if
# F1 score is equal for some configurations. So let's eliminate those
# from the dataframe and put them in "bestest.ranked".
bestest.ranked <- best.ranked %>%
    group_by(clf.id, f1, clf.name, precision, recall, accuracy, gt.0.3,
             clf.factor) %>% 
    summarize(feature.scaling=merge_vector(feature.scaling),
              feature.selection=merge_vector(feature.selection),
              config.id=merge_vector(config.id)) %>%
    ungroup() %>%
    arrange(clf.factor)
bestest.ranked$config.id = as.factor(bestest.ranked$config.id)


# ---------------
# STEP 2 
# Reading the data about the automatic feature selection feature scores
# ---------------

# For getting the features used by authomatic feature selection, use
# file poi_id_featurescores.csv
featurescores.file = "poi_id_featurescores.csv"
featurescores.col.names <- c("feature.scaling", "clf.id", "feature", "score")
featurescores.col.type <- c("factor", "numeric", "character", "numeric")
featurescores <- read.csv(featurescores.file, header=FALSE,
                          col.names=featurescores.col.names,
                          colClasses=featurescores.col.type)

# add some derived variables like clf.name, clf.factor, config.id
# and all the nice things we have in the other dataframes, defined above
# we will need those variables for plotting later on
featurescores.configs <- clf.each.configs %>%
    subset(feature.selection==TRUE) %>%
    select(config.id, feature.scaling)
featurescores.ranked <- featurescores %>%
    left_join(clf.ids, by=c("clf.id")) %>%
    left_join(featurescores.configs, by=c("feature.scaling"))

# Find out which features are ranked "best" 
featurescores.means <- featurescores %>%
    spread(feature, score) %>%
    select(-feature.scaling, -clf.id) %>%
    summarise_each(funs(mean))
featurescores.means <- featurescores.means %>%
    gather("feature","meanscore",1:length(names(featurescores.means))) %>%
    arrange(desc(meanscore))

# Construct an ordered factor from the data frame above; we will need an
# additional level "clf.id" for later plotting
featurescores.levels <-
    c("clf.id", unique(as.character(featurescores.means$feature)))
# convert the character representation in the data frame to a factor
featurescores.ranked$feature <-
  factor(as.character(featurescores.ranked$feature),
         levels=featurescores.levels)

# Make a new dataframe "featurescores.display" which also contains some
# "dummy" observations (stored intermediately in "featurescores.labels")
# that will be used to display the algorithm icon.
# This should be the one you use for displaying.
featurescores.labels <- merge(clf.ids, featurescores.configs) %>%
  select(feature.scaling, clf.id, clf.name, clf.factor, config.id) %>%
  mutate(feature=factor("clf.id",levels=featurescores.levels),
         score=as.numeric(NA))
featurescores.display <- featurescores.ranked %>%
  union(featurescores.labels)


# ---------------
# STEP 3
# A few graphical objects to help us along
# ---------------

# ggplot gradient for displaying algorithm performance (precision, recall etc.)
metrics_scale <- scale_fill_gradientn(
  name="Performance", na.value="white", 
  colours=c("grey","lightgrey","lightgreen","darkgreen"),
  limits=c(0,1), breaks=c(0,0.3,1.0),
  values=c(
    rescale(1:201,to=c(0.0,0.3)),
    rescale(1:201,to=c(0.3,1.0))
  )
)

# ggplot gradient for displaying algorithm configuration
config_scale <- scale_shape_manual(
    name="Configuration", 
    values=c("0"=0,"1"=1,"2"=2,"3"=15,"4"=16,"5"=17,"-1"=63),
    label=c("0"="no feature scaling,\nmanual feature selection",
            "1"="MinMax feature scaling,\nmanual feature selection",
            "2"="PCA,\nmanual feature selection",
            "3"="no feature scaling,\nautomatic feature selection",
            "4"="MinMax feature scaling,\nautomatic feature selection",
            "5"="PCA,\nautomatic feature selection",
            "6"="indifferent",
            "-1"="indifferent")
  )

# ggplot colors for different classifiers; as color without guide
clf_scale_noguide <- scale_color_brewer(
    type="qual", palette="Dark2", guide=FALSE,
    breaks=clf.levels,
    labels=clf.labels
  )

# ggplot colors for different classifiers; as color with guide
clf_scale_guide <- scale_color_brewer(
    name="Classifier",
    type="qual", palette="Dark2",
    breaks=clf.levels,
    labels=clf.labels
  )

# ggplot colors for different classifiers; as fill with guide
clf_scale_fill <- scale_fill_brewer(
    name="Classifier",
    type="qual", palette="Dark2",
    breaks=clf.levels,
    labels=clf.labels
  )

# labeller for facet wrap of "feature selection"
feature.scaling_labeller <- function(var, value){
  value <- as.character(value)
  if (var=="feature.scaling") { 
    value[value=="off"] <- "no feature scaling"
    value[value=="on"] <- "MinMax feature scaling"
    value[value=="pca"] <- "PCA"
  }
  return(value)
}

# ---------------
# STEP 4
# Functions for plotting data
# ---------------


# Function for plotting a table with classifiers (and, if use.config is TRUE)
# their configurations with their F1 score, precision, recall and accuracy on
# a color scale.
plot_table <- function(t, use.configs) {
  
  # We need a running number in order to plot on the y axis and emulate a table
  t$y <- seq.int(nrow(t))
  
  # All potential classifiers
  clfs <- c("clf.factor")
  
  # Build vector of stuff & labels we want to show on the x axis
  # (configs & metrics)
  if (use.configs)
    configs <- c("feature.scaling", "feature.selection")
  else
    configs <- c()
  metrics <- c("f1", "precision", "recall", "accuracy")
  xs <- c("clf.factor", configs, metrics)
  xlabs <- c("F1 Score", "Precision", "Recall", "Accuracy")
  if (use.configs)
    xlabs <- c("Feature\nScaling", "Feature\nSelection", xlabs)
  xlabs <- c("", xlabs)
  
  # For each column of the data frame that we want to show we need
  # a separate "observation" / data row now (tile plots work like that,
  # every cell is one "observation")
  t.plot <- t %>% gather("x", "z", match(xs, names(t)))
  # Manipulate config.id to replace NA values with -1
  t.plot$config.id.shape <- factor(as.character(t.plot$config.id), 
                                   c(levels(t.plot$config.id),"-1"))
  t.plot$config.id.shape[is.na(t.plot$config.id.shape)] <- "-1"
  # Build the tile plot
  p <- ggplot(data=t.plot, aes(x=x, y=clf.factor)) +
    # The tiles themselves for all the metrics
    geom_tile(aes(y=y, fill=as.numeric(z)),
              data=subset(t.plot, x %in% metrics),
              size=0.75) +
    # The texts on the tiles, but this time only the classifer names
    geom_text(aes(y=y, label=as.factor(z)),
              data=subset(t.plot, x == "clf.factor"),
              size=3, hjust=1, position=position_dodge(width=5)) +
    # Also put a nice classifier symbol next to the name
    geom_point(aes(y=y, color=factor(z,levels=clf.levels), 
                   shape=config.id.shape),
               data=subset(t.plot, x == "clf.factor"),
               x = 1.2, size=4, stroke=2, alpha=0.5) + 
    # The texts on the tiles, this time only the configurations (logicals)
    geom_text(aes(y=y, label=as.logical(z)),
              data=subset(t.plot, x == "feature.selection"),
              size=4) +
    geom_text(aes(y=y, label=as.character(z)),
              data=subset(t.plot, x == "feature.scaling"),
              size=4) +
    # The text on the tiles, this time only the metrics (numerics)
    geom_text(aes(y=y, label=round(as.numeric(z),3)),
              data=subset(t.plot, x %in% metrics)) +
    # Scales for the tile coloring and the classifier bullet point layout
    metrics_scale + config_scale + clf_scale_noguide + 
    # Make sure to show the right things on the x axis
    scale_x_discrete(limits=xs, labels=xlabs) + 
    # We want to go top-down not bottom up
    scale_y_reverse() + 
    # Remove all the unnecessary stuff
    theme_bw() + 
    theme(axis.text.y=element_blank(), panel.border=element_blank(),
          panel.grid.major=element_blank(),
          axis.ticks=element_blank()) +
    # Put the axis labels in place
    ylab("Configuration") + xlab("Setting/Metric")
  
  return(p)
  
}


# Function for plotting an x-y-plot with precision and recall and all
# classifiers present in the diagram
plot_performance <- function(t) {
  
  # Plot a x-y diagram of all the records, coloring the data points
  # according to which classifyer was used and using a shape representing
  # the configuration
  p <- ggplot(aes(x=precision, y=recall, color=clf.factor,
                  shape=config.id), data=t) + 
      geom_point(size=4, alpha=0.5) + 
      # Make sure the right colors and symbols are used
      clf_scale_guide + config_scale +
      geom_vline(x=.3, color="gray") +
      geom_hline(y=.3, color="gray") +
      xlab("Precision") + ylab("Recall") + 
      # Remove unnecessary stuff
      theme_bw()
  
  return(p)
  
}


# Function to plot a bar chart comparing the runtime for all classifiers
plot_runtime <- function(t) {
  
  # We need a running number in order to plot smoothly on the x axis
  t$x <- seq.int(nrow(t))
  
  # Plot a bar chart showing the runtime
  p <- ggplot(data=t, aes(x=x, y=runtime)) +
    geom_bar(aes(fill=clf.factor), stat="identity", alpha=0.5) +
    # Put the symbol for each classifier and configuration below every
    # bar chart
    geom_point(aes(shape=config.id, color=clf.factor),
               y=-0.03, size=4, stroke=2, alpha=0.5) + 
    # Make sure shape and symbol are correct
    clf_scale_guide + clf_scale_fill + config_scale + 
    # Nothing to display on the x axis but make sure it's as long as we need
    scale_x_discrete(breaks=c(), lim=c(1,length(t$clf.factor))) + 
    # Add a few sensible ticks to the y (runtime) axis
    scale_y_continuous(lim=c(-1.0,2.5),
                       breaks=c(0.001,0.01,0.1,0.5,1.0,1.5,2.0,2.5)) +
    # sqrt scaling for runtime worked best, at least in this example
    scale_y_sqrt() + 
    # Remove all unnecessary things
    theme_bw() + 
    theme(axis.text.x=element_blank()) + 
    # Group together 4 bars since they are from the same classifier
    geom_vline(x=seq(6.5,(length(unique(t$clf.factor))-1)*6+6.5,6)) + 
    # Add the classifier name spanning all four of their configurations
    # The "gsub" makes sure there is a word wrap after every space following 
    # the 10th and the 21st character (admittetly a bit hacky)
    annotate("text", x=seq(3.5,(length(unique(t$clf.factor))-1)*6+6.5,6), y=2.0,
             label=gsub("^(.{,21})([^ ]*) (.*)$","\\1\\2\n\\3",
                        gsub("^(.{,10})([^ ]*) (.*)$","\\1\\2\n\\3", 
                             unique(t$clf.factor))),
             size = 6) + 
    # No label needed on x axis, y axis = Runtime
    xlab(NULL) + ylab("Average runtime per train-test-split fold in seconds")
  
  return(p)
  
}


# Function to plot the feature scores for automatic feature selection
plot_features <- function(t, ylabels) {
  
  # Plot a tile plot with the features on x and the classifiers 
  # on y - we will facet wrap by the feature scaling criterion 
  # later on
  p <- ggplot(data=t, aes(x=feature, y=clf.factor)) + 
    geom_tile(aes(fill=score)) + 
    # put texts over the tiles
    geom_text(aes(label=round(as.numeric(score),2))) + 
    # put symbol for algorithm in the first column
    geom_point(aes(color=clf.factor, shape=config.id),
               x=1, size=4, stroke=2, alpha=0.5) + 
    # facet wrap by feature scaling (TRUE/FALSE)
    facet_grid(feature.scaling~., 
               labeller=feature.scaling_labeller) + 
    # make sure algorithm symbols (points) show appropriately
    config_scale + clf_scale_guide + 
    # fill the tiles correctly, first column is reserved 
    # for algorithm symbol and should be all "NA", therefore white 
    scale_fill_gradient("Score", low="lightyellow",
                        high="green3", na.value="white") + 
    # remove the first label from the x-axis
    scale_x_discrete(labels=c("",levels(t$feature)
                              [2:length(levels(t$feature))])) + 
    # label y-axis as supplied by parameter
    scale_y_discrete(labels=ylabels) + 
    # adhere to style and add labels
    theme_bw() + 
    theme(axis.text.x=element_text(angle=25, hjust=1.0)) + 
    xlab("Feature") + ylab("Classifier")
  
  return(p)
  
}

