library(keras)
vocab_size <- 1e4

# Load data.
imdb <- dataset_imdb(num_words = vocab_size)
word_index <- dataset_imdb_word_index()
word_index_df <- data.frame(word = names(word_index),
                            idx = unlist(word_index, use.names = FALSE),
                            stringsAsFactors = FALSE)

# Transform the sequences using the dataset_imdb_word_index dictionary (may be done with R base).
# The first indices are reserved
library(dplyr)
word_index_df <- word_index_df %>% mutate(idx = idx + 3)
word_index_df <- word_index_df %>%
    add_row(word = '<PAD>', idx = 0) %>%
    add_row(word = '<START>', idx = 1) %>%
    add_row(word = '<UNK>', idx = 2) %>%
    add_row(word = '<UNUSED>', idx = 3)

word_index_df <- word_index_df %>% arrange(idx)

library(purrr)
decode_review <- function(text){
    paste(map(text, function(number) word_index_df %>%
                  filter(idx == number) %>%
                  select(word) %>%
                  pull()),
          collapse = ' ')
}

# Pad/trim sequences to fixed length.
train_data <- pad_sequences(imdb$train$x,
                            value = word_index_df %>% filter(word == '<PAD>') %>% select(idx) %>% pull(),
                            padding = 'post',
                            maxlen = 512)
test_data <- pad_sequences(imdb$test$x,
                           value = word_index_df %>% filter(word == '<PAD>') %>% select(idx) %>% pull(),
                           padding = 'post',
                           maxlen = 512)

# Load GloVe.6B.50d weights.
glove <- read.table('~/Downloads/glove.6B.50d.txt', sep = ' ', quote = NULL, fill = TRUE)
glove <- glove[!apply(is.na(glove), 1, any), ]
glove.weights <- matrix(0, nrow = vocab_size + 4, ncol = ncol(glove) - 1)
for (i in 1:vocab_size + 4) {
   glove.weights[word_index_df$idx[i], ] <- as.numeric(glove[word_index_df$word[i] == glove$V1, -1])
}

# Build model.
l2 <- 5e-4
model <- keras_model_sequential()
model %>%
    layer_embedding(input_dim = vocab_size, output_dim = 32) %>%
    layer_global_average_pooling_1d() %>%
    layer_dropout(0.8) %>%
    layer_dense(units = 32,
                activation = 'relu',
                kernel_initializer = 'he_uniform',
                kernel_regularizer = regularizer_l2(l = l2)) %>%
    layer_dropout(0.5) %>%
    layer_dense(units = 1,
                activation = 'sigmoid',
                kernel_regularizer = regularizer_l2(l = l2))

model %>% compile(optimizer = 'nadam',
                  loss = 'binary_crossentropy',
                  metrics = list('accuracy'))

history <- model %>% fit(rbind(train_data, test_data),
                         c(imdb$train$y ,imdb$test$y),
                         batch_size = 512,
                         epochs = 20,
                         validation_split = 0.2)
