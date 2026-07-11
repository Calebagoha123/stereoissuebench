#!/usr/bin/env python3
"""Shared regression helpers for the calibration + robustness analyses:
OLS-through-origin, free-intercept OLS, Deming (errors-in-variables), and the CES
x-error-variance from the ground-truth CI. Imported by the 04_calibration and
05_robustness scripts.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

Z = 1.96


def ols_through_origin(x, y):
    """Slope + SE of y = b*x with no intercept."""
    b = np.sum(x * y) / np.sum(x * x)
    resid = y - b * x
    dof = len(x) - 1
    s2 = np.sum(resid ** 2) / dof
    se = np.sqrt(s2 / np.sum(x * x))
    return b, se


def ols_free(x, y):
    """scipy linregress result: slope, intercept, rvalue, pvalue, stderr, intercept_stderr."""
    return stats.linregress(x, y)


def deming(x, y, delta):
    """Deming slope with delta = var(y-error)/var(x-error).

    Convention (Wikipedia 'Deming regression'): y = y* + eps, x = x* + eta,
    delta = var(eps)/var(eta). Reduces to OLS(y|x) as delta->inf and OLS(x|y) as
    delta->0, so the slope lies between the two OLS fits. Returns (slope, intercept).
    """
    xb, yb = x.mean(), y.mean()
    sxx = np.sum((x - xb) ** 2)
    syy = np.sum((y - yb) ** 2)
    sxy = np.sum((x - xb) * (y - yb))
    beta = (syy - delta * sxx + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)) / (2 * sxy)
    alpha = yb - beta * xb
    return beta, alpha


def x_var(df):
    """CES-shift error variance from its (issue/design) CI half-width."""
    half = (df["ces_shift_ci_high"] - df["ces_shift_ci_low"]) / 2.0
    return (half / Z) ** 2
