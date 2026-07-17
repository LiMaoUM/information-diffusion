# Required libraries

library(estimatr)
library(dplyr)
library(ggplot2)
library(lme4)
library(splines)
library(nlme)       # for gls and varWeights
library(MASS)       # for rlm (robust regression)
library(car)        # for VIF

df <- read.csv("/home/maolee/projects/information-diffusion/src/df.csv")

# --- Step 1: 导入必要的库 ---
library(MASS)        # for rlm
library(sandwich)    # for robust standard errors
library(lmtest)      # for coeftest
library(ggplot2)     # for plotting
library(dplyr)       # for data manipulation
library(robustbase)

library(robustbase)
library(tidyr)

# Step 1: 预处理
df_clean <- df %>%
  filter(size > 0, breadth > 0) %>%
  mutate(
    log_size = log10(size),
    log_breadth = log10(breadth)
  ) %>%
  filter(is.finite(log_size), is.finite(log_breadth)) %>%
  drop_na(log_size, log_breadth, platform)

# Step 2: 模型拟合
model2 <- robustbase::lmrob(log_breadth ~ log_size * platform , data = df, method='KS2014',)
summary(model2)


# 打印摘要，里面的 Std. Error 就是稳健的
summary(model2)

evaluate_model <- function(model, data, response_var) {
  # 实际值和拟合值
  y <- data[[response_var]]
  y_hat <- fitted(model)

  # TSS 和 RSS
  tss <- sum((y - mean(y))^2)
  rss <- sum((y - y_hat)^2)

  # Metrics
  pseudo_r2 <- 1 - rss / tss
  mae <- mean(abs(y - y_hat))
  rmse <- sqrt(mean((y - y_hat)^2))
  if (all(y != 0)) {
    mape <- mean(abs((y - y_hat) / y)) * 100
  } else {
    mape <- NA
  }

  # 打印结果
  cat("---- Model Evaluation ----\n")
  cat("Pseudo R^2:", round(pseudo_r2, 4), "\n")
  cat("MAE       :", round(mae, 4), "\n")
  cat("RMSE      :", round(rmse, 4), "\n")
  if (!is.na(mape)) {
    cat("MAPE (%)  :", round(mape, 2), "\n")
  } else {
    cat("MAPE      : N/A (division by zero)\n")
  }
}

# 假设 df 是你的数据框
df$log_size <- log10(df$size)
df$log_breadth <- log10(df$breadth)
df$platform <- as.factor(df$platform)
# --- Step 2: 构建模型 ---


# 拟合 Huber 回归模型
model1 <- rlm(log_breadth ~ log_size * platform, data = df, psi = psi.huber, maxit=100)

# --- Step 3: 打印模型摘要（使用稳健标准误）---
robust_se <- sqrt(diag(vcovHC(model1, type = "HC0")))
coeftest(model1, vcov = vcovHC(model1, type = "HC0"))

# --- Step 4: 残差分析 ---
residuals <- residuals(model1)
fitted <- fitted(model1)

ggplot(data = NULL, aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "Residuals vs Fitted (Huber Regression)",
       x = "Fitted Values",
       y = "Residuals") +
  theme_minimal()

# Model 2
df$outlier <- as.factor(df$outlier)
model2 <- rlm(log_breadth ~ log_size * outlier, data = df, psi = psi.huber, maxit=100)
# --- Step 3: 打印模型摘要（使用稳健标准误）---
robust_se <- sqrt(diag(vcovHC(model2, type = "HC0")))
coeftest(model2, vcov = vcovHC(model2, type = "HC0"))

evaluate_model(model2, df, "log_breadth")

# Model3
# --- Step 1: 加载必要库 ---
library(MASS)         # for rlm()
library(splines)      # for bs() to create splines
library(ggplot2)
library(dplyr)
library(splines2)
# --- Step 2: 数据预处理 ---
df <- df %>%
  filter(size > 0, breadth > 0) %>%
  mutate(
    log_size = log10(size),
    log_breadth = log10(breadth),
    weights = 1 / size
  ) %>%
  drop_na(platform, log_size, log_breadth, alignment_ratio, weights)

# --- Step 3: 建模 ---
model3 <- rlm( log_breadth ~ log_size * platform * bsp(alignment_ratio, df = 5), data = df, psi = psi.huber, maxit = 5000)

summary(model3)
