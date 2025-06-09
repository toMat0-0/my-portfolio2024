
library(ggplot2)
library(dplyr)
library(nnet)
library(MASS)
setwd("~/Gwen/UV/U2/Machine Learning/final/pythonProject1")
#data
spectrum <- read.csv("all_spectrum_features.csv")
spectrum_data <- spectrum[, grepl("^spec_", names(spectrum))]
n <- nrow(spectrum_data)
freq_bins <- ncol(spectrum_data)

#all sepctra
spectrum.df <- data.frame(
  frequencies = rep(1:freq_bins, n),
  all_spectra = c(t(spectrum_data)),
  bark = rep(1:n, each = freq_bins),
  emotion = rep(spectrum$label, each = freq_bins)
)

# Overlay spectra
ggplot(spectrum.df, aes(x=frequencies, y=all_spectra, group=bark, color=emotion)) +
  geom_line(alpha=0.5) +
  labs(title="All Bark Spectra Colored by Emotion") +
  theme_minimal()
#mean +- sd
spectrum_summary <- spectrum.df %>%
  group_by(frequencies, emotion) %>%
  summarise(mean_val = mean(all_spectra),
            sd_val = sd(all_spectra), .groups = "drop")

ggplot(spectrum_summary, aes(x=frequencies, y=mean_val, color=emotion)) +
  geom_line(size=1) +
  geom_ribbon(aes(ymin=mean_val - sd_val, ymax=mean_val + sd_val, fill=emotion), alpha=0.2, color=NA) +
  labs(title="Mean Spectrum ± SD by Emotion", y="Amplitude (dB)", x="Frequency Bin") +
  theme_minimal()


#pca
X <- spectrum[, grepl("^spec_", names(spectrum))]
y <- factor(spectrum$label)
pca_result <- prcomp(X, center = TRUE, scale. = TRUE)

pca_df <- data.frame(PC1 = pca_result$x[,1], PC2 = pca_result$x[,2], emotion = y)

ggplot(pca_df, aes(x = PC1, y = PC2, color = emotion)) +
  geom_point(size = 3) +
  labs(title = "PCA of Dog Bark Spectrum Features") +
  theme_minimal()
summary(pca_result)

plot(pca_result, type = "l", main = "Scree Plot of PCA")

library(plotly)
PC3= pca_result$x[,3]
plot_ly(pca_df, x = ~PC1, y = ~PC2, z = ~PC3, color = ~emotion, colors = c("blue", "orange"), type = "scatter3d", mode = "markers")

# Create a data frame for export
pca_export <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  PC3 = pca_result$x[, 3],
  emotion = spectrum$label
)

# Write to CSV for use in JS
write.csv(pca_export, "pca_data.csv", row.names = FALSE)
file.exists("pca_data.csv")

#model
# Load train/test data
train_df <- read.csv("all_spectrum_features.csv")
test_df <- read.csv("all_spectrum_features_test.csv")

X_train <- train_df[, grepl("^spec_", names(train_df))]
X_test  <- test_df[, grepl("^spec_", names(test_df))]
y_train <- as.factor(train_df$label)
y_test  <- as.factor(test_df$label)

# PCA transformation
pca_model <- prcomp(X_train, center = TRUE, scale. = TRUE)
train_pca <- as.data.frame(pca_model$x[, 1:10])
test_pca  <- as.data.frame(predict(pca_model, newdata = X_test)[, 1:10])
train_pca$label <- y_train

library(GGally)
pca_df <- data.frame(pca_result$x[, 1:10])  # Take PCs 1 to 10
pca_df$label <- y  # Add emotion labels

# Pair plot of first few PCs (this can be slow if too many PCs)
ggpairs(pca_df, columns = 1:5, mapping = aes(color = label),
        title = "Pairwise PCA Plots of Bark Emotions")

#pca10 vs 20
train_pca_5 <- as.data.frame(pca_model$x[, 1:5])
train_pca_10 <- as.data.frame(pca_model$x[, 1:10])
train_pca_20 <- as.data.frame(pca_model$x[, 1:20])

train_pca_5$label <- y_train
train_pca_10$label <- y_train
train_pca_20$label <- y_train

# Project test set into PC space
test_pca_5 <- as.data.frame(predict(pca_model, newdata = X_test)[, 1:5])
test_pca_10 <- as.data.frame(predict(pca_model, newdata = X_test)[, 1:10])
test_pca_20 <- as.data.frame(predict(pca_model, newdata = X_test)[, 1:20])
library(nnet)  # For multinom

# 7 PCs
model_5 <- multinom(label ~ ., data = train_pca_5)
pred_5 <- predict(model_5, newdata = test_pca_5)
acc_5 <- mean(pred_5 == y_test)

# 10 PCs
model_10 <- multinom(label ~ ., data = train_pca_10)
pred_10 <- predict(model_10, newdata = test_pca_10)
acc_10 <- mean(pred_10 == y_test)

# 20 PCs
model_20 <- multinom(label ~ ., data = train_pca_20)
pred_20 <- predict(model_20, newdata = test_pca_20)
acc_20 <- mean(pred_20 == y_test)

cat("Accuracy with 5 PCs:", round(acc_5, 3), "\n")
cat("Accuracy with 10 PCs:", round(acc_10, 3), "\n")
cat("Accuracy with 20 PCs:", round(acc_20, 3), "\n")




#Multinomial Logistic Regression
mult_model <- multinom(label ~ ., data = train_pca)
y_pred <- predict(mult_model, newdata = test_pca)
table(Predicted = y_pred, Actual = y_test)
mean(y_pred == y_test)

# Get predicted class and max probability
probs <- predict(mult_model, newdata = test_pca, type = "prob")
probs_df <- as.data.frame(probs)
probs_df$true_label <- y_test
probs_df$predicted <- colnames(probs)[apply(probs, 1, which.max)]
probs_df$max_prob <- apply(probs, 1, max)

ggplot(probs_df, aes(x = predicted, y = max_prob, color = true_label)) +
  geom_jitter(width = 0.2, height = 0) +
  labs(title = "Confidence of Predictions (Multinomial Logistic)", x = "Predicted Class", y = "Max Probability") +
  theme_minimal()


#lda-raw
lda_model_raw <- lda(x = X_train, grouping = y_train)
y_pred_raw <- predict(lda_model_raw, newdata = X_test)$class
table(Predicted = y_pred_raw, Actual = y_test)
mean(y_pred_raw == y_test)
#lda-lpot
lda_proj <- predict(lda_model_raw)$x
plot_df <- data.frame(LD1 = lda_proj[, 1], label = y_train)

ggplot(plot_df, aes(x = LD1, fill = label)) +
  geom_density(alpha = 0.5) +
  labs(title = "LDA Projection of Bark Emotions") +
  theme_minimal()

#lda-pca
lda_model_pca <- lda(label ~ ., data = train_pca)
y_pred_pca <- predict(lda_model_pca, newdata = test_pca)$class
table(Predicted = y_pred_pca, Actual = y_test)
mean(y_pred_pca == y_test)
lda_proj <- predict(lda_model_pca)$x
plot_df <- data.frame(LD1 = lda_proj[,1], label = y_train)

ggplot(plot_df, aes(x = LD1, fill = label)) +
  geom_density(alpha = 0.4) +
  labs(title = "LDA Projection on First Linear Discriminant", x = "LD1") +
  theme_minimal()

#binary
train_pca$binary_label <- ifelse(train_pca$label == "normal", 1, 0)
test_pca$binary_label <- ifelse(y_test == "normal", 1, 0)
#pca
logit_model <- glm(binary_label ~ ., data = train_pca[, 1:10] %>% mutate(binary_label = train_pca$binary_label), family = "binomial")

pred_probs <- predict(logit_model, newdata = test_pca[, 1:10], type = "response")

# Class prediction: threshold at 0.5
y_pred_logit <- ifelse(pred_probs > 0.5, 1, 0)

# Actual binary labels
y_test_bin <- test_pca$binary_label

pred_probs <- predict(logit_model, newdata = test_pca, type = "response")
residuals <- y_test_bin == y_pred_logit  # TRUE if correct

plot_df <- data.frame(prob = pred_probs, actual = y_test_bin, correct = residuals)

ggplot(plot_df, aes(x = prob, fill = correct)) +
  geom_histogram(bins = 10, position = "identity", alpha = 0.6) +
  facet_wrap(~ actual) +
  labs(title = "Prediction Probabilities vs. Actual Class", x = "Predicted Probability", fill = "Correct") +
  theme_minimal()

# Confusion matrix
table(Predicted = y_pred_logit, Actual = y_test_bin)
library(ggplot2)
library(tidyr)
library(dplyr)
# Extract original spectral features and label
X_raw <- spectrum[, grepl("^spec_", names(spectrum))]
X_raw$label <- spectrum$label

# Pick a few frequency bins (e.g., spec_100, spec_500, spec_900)
bins_to_check <- c("spec_100", "spec_500", "spec_900")
long_df <- pivot_longer(X_raw, cols = all_of(bins_to_check), names_to = "bin", values_to = "amplitude")

# Plot distributions
ggplot(long_df, aes(x = amplitude, fill = label)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~ bin, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Selected Frequency Bins by Emotion")

library(ggplot2)
# Assuming train_df$label exists and is a factor with "normal", "defensive", "aggressive"
y_train_bin <- ifelse(train_df$label == "normal", 1, 0)  # 1 = normal, 0 = other

plot_data <- data.frame(PC1 = train_pca$PC1,
                        PC2 = train_pca$PC2,
                        label = y_train_bin)  # binary labels

# Fit binary logistic on first 2 PCs
logit_2D <- glm(label ~ PC1 + PC2, data = plot_data, family = "binomial")

# Generate grid to visualize decision boundary
x_range <- seq(min(plot_data$PC1), max(plot_data$PC1), length.out = 100)
y_range <- seq(min(plot_data$PC2), max(plot_data$PC2), length.out = 100)
grid <- expand.grid(PC1 = x_range, PC2 = y_range)
grid$prob <- predict(logit_2D, newdata = grid, type = "response")

ggplot(plot_data, aes(x = PC1, y = PC2, color = factor(label))) +
  geom_point(size = 3) +
  geom_contour(data = grid, aes(z = prob), breaks = 0.5, color = "black") +
  labs(title = "PCA Projection with Logistic Regression Boundary",
       color = "Class (Normal=1)") +
  theme_minimal()

library(ggplot2)
library(reshape2)

cm <- table(Predicted = y_pred, Actual = y_test)
cm_df <- as.data.frame(cm)

ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix: PCA + Logistic", x = "Predicted label", y = "True label") +
  theme_minimal()



# In R
residuals <- residuals(logit_2D, type = "deviance")
plot(fitted(logit_2D), residuals)
abline(h = 0, col = "red")
plot(fitted(logit_2D), residuals,
     xlab = "Fitted Probability",
     ylab = "Deviance Residual",
     main = "Residuals vs Fitted (Binary PCA Logit)")
abline(h = 0, col = "red")

plot_df <- plot_data
plot_df$fitted <- fitted(logit_2D)
plot_df$residuals <- residuals
plot_df$correct <- ifelse((fitted(logit_2D) > 0.5) == y_train_bin, "Correct", "Wrong")

library(ggplot2)
ggplot(plot_df, aes(x = fitted, y = residuals, color = correct)) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residuals vs Fitted (Colored by Prediction Accuracy)",
       x = "Fitted Probability", y = "Deviance Residual") +
  theme_minimal()
library(arm)
binnedplot(fitted(logit_2D), 
           residuals(logit_2D, type = "response"), 
           nclass = NULL, 
           xlab = "Expected Values", 
           ylab = "Average residual", 
           main = "Binned residual plot", 
           cex.pts = 0.8, 
           col.pts = 1, 
           col.int = "gray")

###result
# Accuracy
acc_multinom_pca10 <- mean(y_pred == y_test)

acc_multinom_pca5 <- acc_5
acc_multinom_pca10 <- acc_10
acc_multinom_pca20 <- acc_20
acc_lda_raw <- mean(y_pred_raw == y_test)

acc_lda_pca <- mean(y_pred_pca == y_test)

acc_lda_pca <- mean(y_pred_pca == y_test)

acc_logit_binary <- mean(y_pred_logit == y_test_bin)

accuracy_results <- data.frame(
  Model = c("Multinom (5 PCs)", "Multinom (10 PCs)", "Multinom (20 PCs)",
            "Multinom (10 PCs, Overall)", "LDA (Raw)", "LDA (PCA)", "Binary Logit (Normal vs Other)"),
  Accuracy = c(acc_multinom_pca5, acc_multinom_pca10, acc_multinom_pca20,
               acc_multinom_pca10, acc_lda_raw, acc_lda_pca, acc_logit_binary)
)

print(accuracy_results)
