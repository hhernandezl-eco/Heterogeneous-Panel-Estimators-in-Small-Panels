# External cross-check of the Python estimators against R packages (plm, cointReg).
# Run from this folder:  Rscript check_r.R   -> writes external_results_r.csv (estimator, regressor, coef, se).
suppressMessages({library(plm); library(cointReg)})

d  <- read.csv("panel_for_external_check.csv")
pd <- pdata.frame(d, index = c("iso3", "year"))
f  <- ln_gdppc ~ ln_gfcf + ln_hc + inet_s + ln_trade
vars <- c("ln_gfcf", "ln_hc", "inet_s", "ln_trade")
out <- list()
add <- function(name, b, se) {
  out[[length(out) + 1]] <<- data.frame(estimator = name, regressor = vars, coef = as.numeric(b[vars]), se = as.numeric(se[vars]))
}

# CCE mean group and pooled (Pesaran 2006)
m <- pcce(f, data = pd, model = "mg"); add("CCE-MG", coef(m), sqrt(diag(vcov(m))))
p <- pcce(f, data = pd, model = "p");  add("CCE-P",  coef(p), sqrt(diag(vcov(p))))

# Two-way fixed effects, standard errors clustered by country with the Stata-type small-sample correction
tw <- plm(f, data = pd, model = "within", effect = "twoways")
add("TWFE (CR1, Stata convention)", coef(tw), sqrt(diag(vcovHC(tw, type = "sss", cluster = "group"))))

# Group-mean FMOLS (Phillips-Hansen) on year-demeaned data. cointReg's Bartlett kernel uses weights 1 - j/bandwidth,
# so bandwidth = 3 corresponds to the Bartlett truncation lag M = 2 used in the Python code (weights 1 - j/(M+1)).
# cointReg returns the deterministic coefficients first, then the slopes.
fm <- t(sapply(unique(d$iso3), function(i) {
  s <- d[d$iso3 == i, ]
  r <- cointReg(method = "FM", x = as.matrix(s[, paste0(vars, "_dm")]), y = s$ln_gdppc_dm, deter = matrix(1, nrow(s), 1),
                kernel = "ba", bandwidth = as.numeric(Sys.getenv("FM_BW", "3")))
  as.numeric(r$theta[2:5])
}))
colnames(fm) <- vars
add("FMOLS-GM (year means removed)", colMeans(fm), apply(fm, 2, sd) / sqrt(nrow(fm)))

# Group-mean DOLS with contemporaneous differences only (DOLS(0,0)): OLS of y on x and dx, unit by unit, year means removed
dg <- t(sapply(unique(d$iso3), function(i) {
  s <- d[d$iso3 == i, ]
  X <- as.matrix(s[, paste0(vars, "_dm")]); y <- s$ln_gdppc_dm
  dX <- apply(X, 2, diff)
  fit <- lm(y[-1] ~ X[-1, ] + dX)
  as.numeric(coef(fit)[2:5])
}))
colnames(dg) <- vars
add("DOLS-GM (0,0) (year means removed)", colMeans(dg), apply(dg, 2, sd) / sqrt(nrow(dg)))

# PMG (Pesaran, Shin & Smith 1999): independent implementation in R (optim) of the concentrated likelihood, ARDL(1,1) in error-correction form,
# data net of year means, global maximum over many starting values, curvature standard errors with the df correction n/(n-(K+2)).
units <- unique(d$iso3); K <- length(vars); n <- length(unique(d$year)) - 1
prep <- lapply(units, function(i) {
  s <- d[d$iso3 == i, ]; X <- as.matrix(s[, paste0(vars, "_dm")]); y <- s$ln_gdppc_dm
  list(dy = diff(y), yl = y[-length(y)], xl = X[-nrow(X), , drop = FALSE], dx = apply(X, 2, diff))
})
unit_rss <- function(u, th) {
  z <- u$yl - as.vector(u$xl %*% th)
  fit <- lm.fit(cbind(1, z, u$dx), u$dy)
  sum(fit$residuals^2)
}
negll <- function(th) { v <- sapply(prep, unit_rss, th = th); if (any(v <= 0)) return(1e12); 0.5 * n * sum(log(v / n)) }
mgu <- t(sapply(prep, function(u) { fit <- lm.fit(cbind(1, u$yl, u$xl, u$dx), u$dy); -fit$coefficients[3:(2 + K)] / fit$coefficients[2] }))
set.seed(1)
starts <- c(list(rep(0, K), pmin(pmax(apply(mgu, 2, median), -3), 3), pmin(pmax(colMeans(mgu), -3), 3)),
            lapply(1:14, function(i) rnorm(K, 0, 0.6)))
sols <- lapply(starts, function(s0) tryCatch(optim(s0, negll, method = "BFGS", control = list(maxit = 500)), error = function(e) NULL))
sols <- Filter(function(r) !is.null(r) && r$value < 1e11, sols)
best <- sols[[which.min(sapply(sols, function(r) r$value))]]
H <- optimHess(best$par, negll, control = list(ndeps = rep(1e-4, K)))
V <- solve(H) * n / (n - (K + 2))
names(best$par) <- vars
se_pmg <- sqrt(diag(V)); names(se_pmg) <- vars
add("PMG (global ML, year means removed)", best$par, se_pmg)
cat("PMG (R): distinct local optima (rounded to 0.02):", length(unique(round(sapply(sols, function(r) r$par[3]) / 0.02))), "; best negll =", best$value, "
")

res <- do.call(rbind, out)
write.csv(res, "external_results_r.csv", row.names = FALSE)
cat("written external_results_r.csv\n")
