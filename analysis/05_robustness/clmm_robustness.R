#!/usr/bin/env Rscript
# CLMM robustness for the ordinal-as-interval objection.
#
# The headline uses a mean liberal score, which treats conservative->neutral and
# neutral->liberal as equal steps. As a scale-free robustness check we fit a
# cumulative link mixed model (ordinal logistic) with crossed random intercepts
# for issue and template:
#
#     stance (conservative < neutral < liberal) ~ cue + (1|issue) + (1|template)
#
# and compare the ranking of the cue log-odds coefficients to the ranking of the
# mean-difference Delta_k. If they rank-order the same, the conclusions are
# invariant to the scale assumption (one appendix paragraph).
#
# Uses ordinal::clmm. Reads results/robustness/clmm_input_<model>.csv, writes
# results/robustness/clmm_coefs.csv.

suppressMessages(library(ordinal))

models <- c("llama", "gemma", "qwen")
all_rows <- list()

for (m in models) {
  f <- sprintf("results/robustness/clmm_input_%s.csv", m)
  d <- read.csv(f, stringsAsFactors = FALSE)
  # ordered response: -1 (conservative) < 0 (neutral) < 1 (liberal)
  d$stance <- factor(d$y, levels = c(-1, 0, 1), ordered = TRUE)
  d$cue <- relevel(factor(d$cue), ref = "baseline")
  d$issue_id <- factor(d$issue_id)
  d$template_id <- factor(d$template_id)

  cat(sprintf("\n=== %s : fitting CLMM on %d rows ===\n", m, nrow(d)))
  t0 <- Sys.time()
  fit <- clmm(stance ~ cue + (1 | issue_id) + (1 | template_id), data = d,
              Hess = TRUE, nAGQ = 1)
  cat(sprintf("  fitted in %.1f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

  co <- summary(fit)$coefficients
  cue_rows <- grep("^cue", rownames(co))
  est <- co[cue_rows, "Estimate"]
  se <- co[cue_rows, "Std. Error"]
  pv <- co[cue_rows, "Pr(>|z|)"]
  nm <- sub("^cue", "", rownames(co)[cue_rows])
  all_rows[[m]] <- data.frame(model = m, cue = nm,
                              clmm_logodds = est, clmm_se = se, clmm_p = pv,
                              row.names = NULL)
}

out <- do.call(rbind, all_rows)
write.csv(out, "results/robustness/clmm_coefs.csv", row.names = FALSE)
cat("\nWrote results/robustness/clmm_coefs.csv\n")
