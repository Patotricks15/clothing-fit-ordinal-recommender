# Clothing Fit Ordinal Recommender

> Classical machine learning portfolio project: an ordinal logistic
> regression (proportional-odds / cumulative-link model), trained with
> `scipy.optimize`, wrapped in split conformal prediction so every
> recommendation ships with a calibrated uncertainty set instead of a bare
> point guess.

**Status: work in progress.** This README is a scaffold; see later commits
for the full problem statement, quickstart, architecture, and results.

## Why this project exists

Most tabular recommender portfolios stop at binary ("will this user like
this item?") or ranking ("which item first?") outputs. Clothing-fit
prediction is naturally **ordinal** — a garment runs *small*, *fits*, or
*runs large*, and those outcomes are ordered, not just distinct classes.
Treating `fit` as an ordinal target (not one-hot multinomial) lets the model
share information across adjacent levels, and wrapping the fitted model in
**split conformal prediction** turns a point prediction into a calibrated set
of plausible outcomes with a marginal coverage guarantee — a more honest
signal for a sizing recommendation than a single label.

## License

MIT — see [LICENSE](LICENSE).
