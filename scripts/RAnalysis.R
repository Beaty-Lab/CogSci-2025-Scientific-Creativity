library(readr)
library(cocor)
library(dplyr)
library(tidyr)

# Read data and handle NAs
data <- read_csv("cleanded_data_explanations_gold.csv") %>%
  drop_na(cleverness, remoteness, uncommonness, originality)  # Drop rows with NA in these columns

condition_oracle <- data %>% 
  filter(condition == "oracle")

condition_no_oracle <- data %>%
  filter(condition == "no_oracle")

cor_cleverness_remoteness_oracle <- cor(condition_oracle$cleverness, condition_oracle$remoteness)
cor_cleverness_remoteness_no_oracle <- cor(condition_no_oracle$cleverness, condition_no_oracle$remoteness)

cor_cleverness_uncommonness_oracle <- cor(condition_oracle$cleverness, condition_oracle$uncommonness)
cor_cleverness_uncommonness_no_oracle <- cor(condition_no_oracle$cleverness, condition_no_oracle$uncommonness)

cor_remoteness_uncommonness_oracle <- cor(condition_oracle$remoteness, condition_oracle$uncommonness)
cor_remoteness_uncommonness_no_oracle <- cor(condition_no_oracle$remoteness, condition_no_oracle$uncommonness)

cor_originality_cleverness_oracle <- cor(condition_oracle$originality, condition_oracle$cleverness)
cor_originality_cleverness_no_oracle <- cor(condition_no_oracle$originality, condition_no_oracle$cleverness)

cor_originality_remoteness_oracle <- cor(condition_oracle$originality, condition_oracle$remoteness)
cor_originality_remoteness_no_oracle <- cor(condition_no_oracle$originality, condition_no_oracle$remoteness)

cor_originality_uncommonness_oracle <- cor(condition_oracle$originality, condition_oracle$uncommonness)
cor_originality_uncommonness_no_oracle <- cor(condition_no_oracle$originality, condition_no_oracle$uncommonness)

c

comparison_cleverness_uncommonness <- cocor.indep.groups(
    r1.jk = cor_cleverness_uncommonness_oracle,
    r2.hm = cor_cleverness_uncommonness_no_oracle,
    n1 = nrow(condition_oracle),
    n2 = nrow(condition_no_oracle),
)

comparison_remoteness_uncommonness <- cocor.indep.groups(
    r1.jk = cor_remoteness_uncommonness_oracle,
    r2.hm = cor_remoteness_uncommonness_no_oracle,
    n1 = nrow(condition_oracle),
    n2 = nrow(condition_no_oracle),
)

comparison_originality_cleverness <- cocor.indep.groups(
    r1.jk = cor_originality_cleverness_oracle,
    r2.hm = cor_originality_cleverness_no_oracle,
    n1 = nrow(condition_oracle),
    n2 = nrow(condition_no_oracle),
)

comparison_originality_remoteness <- cocor.indep.groups(
    r1.jk = cor_originality_remoteness_oracle,
    r2.hm = cor_originality_remoteness_no_oracle,
    n1 = nrow(condition_oracle),
    n2 = nrow(condition_no_oracle),
)

comparison_originality_uncommonness <- cocor.indep.groups(
    r1.jk = cor_originality_uncommonness_oracle,
    r2.hm = cor_originality_uncommonness_no_oracle,
    n1 = nrow(condition_oracle),
    n2 = nrow(condition_no_oracle),
)