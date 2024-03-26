# Introduction

Understanding causality is crucial before implementing machine learning algorithms. Why is it important? Because it allows us to identify the causal relationships between the target variable and predictor columns in our dataset.

What are the steps involved?

Correlation Analysis:
After obtaining correlations using techniques like chi-square, Spearman correlation, etc., we need to determine which variables we believe truly affect our target variable. If the correlations are not conclusive, we must verify them using a causal model.

Causal Model:
A popular approach to causal modeling is using Bayesian networks. The network describes the relationships between variables, indicating that the closer a variable is to the target variable, the higher the likelihood of it having a causal effect. However, it's essential to avoid biased assumptions by employing counterfactual analysis.

Counterfactual Analysis:
To ensure the correctness of our assumptions, we perform counterfactual analysis. If the resulting p-value is below a significance threshold (typically 0.05), we reject assumptions that suggest causality.

What are the benefits?

By obtaining supporting information through these steps, we can make informed decisions and avoid biased metrics when building prediction models. It's essential to combine this data-driven approach with domain knowledge for optimal decision-making.

Important Note: Decision-making should always be informed by both data-driven analysis and domain expertise.
