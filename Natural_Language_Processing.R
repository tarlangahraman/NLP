library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)

data <-  read.csv("emails.csv")
data %>% names()

set.seed(123)
split <- data$spam %>% sample.split(SplitRatio = 0.8)
train <- data %>% subset(split == T)
test <- data %>% subset(split == F)

it_train <- train$text %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$X,
         progressbar = F) 

install.packages("stopwords")
library(stopwords)

SW = stopwords("en", source = "stopwords-iso")
ngram = c(1L,2L)

vocab <- it_train %>% create_vocabulary(stopwords = SW, ngram=ngram)

pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 30, 
                   doc_proportion_max = 0.5)

vocab %>% 
  arrange(desc(term_count)) %>% 
  head(110) %>% 
  tail(10) 

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()
identical(rownames(dtm_train), (train$X %>% as.character()))


glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")


it_test <- test$text %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$X,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)

