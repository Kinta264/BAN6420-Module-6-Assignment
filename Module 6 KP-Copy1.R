# 1. Load Libraries
library(keras)
library(tensorflow)
library(ggplot2)

# 2. Load Data
fashion_mnist <- dataset_fashion_mnist()
X_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
X_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

# 3. Normalize
X_train <- X_train / 255
X_test <- X_test / 255

# 4. Reshape for CNN input
X_train <- array_reshape(X_train, c(nrow(X_train), 28, 28, 1))
X_test <- array_reshape(X_test, c(nrow(X_test), 28, 28, 1))

# 5. Class Labels
class_names <- c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
								 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# 6. Model Architecture
model <- keras_model_sequential() %>%
	layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28,28,1)) %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_flatten() %>%
	layer_dense(units = 128, activation = 'relu') %>%
	layer_dropout(rate = 0.3) %>%
	layer_dense(units = 10, activation = 'softmax')

# 7. Compile
model %>% compile(
	optimizer = 'adam',
	loss = 'sparse_categorical_crossentropy',
	metrics = 'accuracy'
)

# 8. Train
history <- model %>% fit(
	X_train, y_train,
	epochs = 10,
	validation_data = list(X_test, y_test)
)

# 9. Evaluate
score <- model %>% evaluate(X_test, y_test)
cat(sprintf("Test Accuracy: %.2f\n", score[[2]]))

# 10. Plot Accuracy
df <- data.frame(
	epoch = 1:10,
	train_acc = history$metrics$accuracy,
	val_acc = history$metrics$val_accuracy
)

ggplot(df, aes(x = epoch)) +
	geom_line(aes(y = train_acc, color = "Train Accuracy")) +
	geom_line(aes(y = val_acc, color = "Validation Accuracy")) +
	labs(title = "Accuracy over Epochs", y = "Accuracy") +
	scale_color_manual(values = c("Train Accuracy" = "blue", "Validation Accuracy" = "red")) +
	theme_minimal()

# 11. Prediction Example
predict_image <- function(index) {
	img <- X_test[index,,,drop=FALSE]
	pred <- model %>% predict(img)
	predicted_label <- class_names[which.max(pred)]
	actual_label <- class_names[y_test[index] + 1]
	cat(sprintf("Predicted: %s | Actual: %s\n", predicted_label, actual_label))
	image(as.matrix(X_test[index,,,1]), col = gray.colors(256), main = "Image")
}

# Example
predict_image(1)
