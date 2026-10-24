Here you can find the 4th winning model of Wids datathon 2026.
Short description:
The code builds a hybrid model to estimate the probability of an event occurring within 12, 24, 48, and 72 hours. It first creates additional features related to distance, speed, fire growth, threat level, and time. Then, it trains both a GradientBoostingSurvivalAnalysis model and several LightGBM classifiers using cross-validation and multiple random seeds. Their predictions are combined using horizon-specific weights, calibrated, and adjusted to ensure that probabilities increase consistently over time.
