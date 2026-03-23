library(tidyverse)
library(glmmTMB)
library(tidyr)
library(dplyr)

setwd('~/Desktop/HVRD/workspace/manifold-dynamics/aux_controls/')
ed_main <- read.csv('sampling_strength_summary.csv')


# distribution of local ED values
ggplot(ed_main, aes(x=local_ed)) + 
  geom_density()

# scatter plot of global vs local ED 
ggplot(ed_main, aes(x=global_ed, y=local_ed)) + 
  geom_point()

# sampling strength vs local ED
ggplot(ed_main, aes(x=avg_topk_zscore, y=local_ed)) + 
  geom_point() 

# long format
ed_long <- ed_main %>%
  pivot_longer(
    cols = c(local_ed, global_ed),
    names_to = "type",
    values_to = "ED"
  ) %>%
  mutate(type = sub("_ed", "", type))

str(ed_long)
# turn variables into factors as needed
ed_long$type <- as.factor(ed_long$type)
ed_long$major_selectivity <- as.factor(ed_long$major_selectivity)
ed_long$roi_key <- as.factor(ed_long$roi_key)
str(ed_long)

# histogram of ED values
ggplot(ed_long, aes(x=ED)) + 
  geom_density()

# slightly right skewed, strictly positive
ggplot(ed_long, aes(x=ED, color=type)) + 
  geom_density()

# by major selectivity
ggplot(ed_long, aes(x=major_selectivity, y=ED, color=type)) + 
  geom_boxplot()
ggplot(ed_long, aes(x=type, y=ED, color=major_selectivity)) + 
  geom_boxplot()

# simple glmm fit
fit03.1 <- glmmTMB(data=ed_long, ED ~ type)
summary(fit03.1)
# no difference in ED across 3 ROIs
fit03.2 <- glmmTMB(data=ed_long, ED ~ type + major_selectivity)
summary(fit03.2)
# not what this means. face ROIs have larger ED in general?
fit03.3 <- glmmTMB(data=ed_long, ED ~ type * major_selectivity)
summary(fit03.3)

# interaction between ED change and top-k sampling strength
# both effects are present: local --> global and sampling strength --> low ED
fit04 <- glmmTMB(data=ed_long, ED ~ type * avg_topk_zscore)
summary(fit04)

# keeping it maximal
fit05 <- glmmTMB(data=ed_long, ED ~ type * avg_topk_zscore * major_selectivity)
summary(fit05)
