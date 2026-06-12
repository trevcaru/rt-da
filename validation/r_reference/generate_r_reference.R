# Gold-standard R harness: uses the authors' uvsdt.R verbatim for fitting,
# replicates their per-dataset count construction in base R, and records
# per-subject mu/sigma/da/logL/criteria + optim convergence + whether the
# authors' fit_uvsdt_mle errors (na.omit exclusion).
# Reads the raw datasets + uvsdt.R from ../../docs and writes sherman_jocn1.csv,
# sherman_jocn2.csv, mazor2020.csv next to this script. Requires the raw data
# CSVs (not committed) to be present in docs/. Run with the working directory
# set to this validation/r_reference/ folder.
out_dir <- getwd()
setwd("../../docs")
source("uvsdt.R")  # provides fit_uvsdt_mle() and uvsdt_logL() exactly as authors wrote

# faithful base-R port of dplyr::ntile (larger groups first, ties by row order)
ntile_base <- function(x, n) {
  ord <- order(x, na.last = TRUE)          # stable for numeric ties
  fin <- is.finite(x[ord])
  len <- sum(fin)
  out <- integer(length(x))
  if (len == 0) return(out)
  n_larger <- len %% n
  larger_size <- ceiling(len / n)
  smaller_size <- floor(len / n)
  larger_threshold <- larger_size * n_larger
  ranks <- seq_len(len)
  bins <- ifelse(ranks <= larger_threshold,
                 (ranks - 1) %/% max(larger_size, 1) + 1,
                 (ranks - larger_threshold - 1) %/% max(smaller_size, 1) + n_larger + 1)
  out[ord[seq_len(len)]] <- bins
  out
}

# fit wrapper that ALSO returns optim convergence + error flag, but uses the
# authors' uvsdt_logL and identical optim call (their fit_uvsdt_mle internals).
fit_full <- function(nr_s1, nr_s2, add_constant = TRUE) {
  err <- FALSE
  res <- tryCatch(fit_uvsdt_mle(nr_s1, nr_s2, add_constant = add_constant),
                  error = function(e) { err <<- TRUE; NULL })
  # separately capture convergence with an identical optim call
  s1 <- nr_s1; s2 <- nr_s2
  if (add_constant) { s1 <- s1 + 1/length(s1); s2 <- s2 + 1/length(s2) }
  rating_far <- cumsum(s1)/sum(s1); rating_hr <- cumsum(s2)/sum(s2)
  n_ratings <- length(s1)/2
  mu0 <- qnorm(rating_hr[n_ratings]) - qnorm(rating_far[n_ratings])
  cri0 <- (-qnorm(rating_far))[1:(2*n_ratings-1)]
  guess <- c(mu0, 1.5, cri0)
  conv <- NA
  fit <- tryCatch(suppressWarnings(optim(uvsdt_logL, par = guess,
            inputs = list(n_ratings = n_ratings, nr_s1 = s1, nr_s2 = s2),
            gr = NULL, method = "BFGS", control = list(maxit = 10000))),
            error = function(e) NULL)
  if (!is.null(fit)) conv <- fit$convergence
  if (err || is.null(res)) {
    return(list(err = TRUE, conv = conv, mu = NA, sigma = NA, da = NA,
                logL = NA, cri = rep(NA, 2*n_ratings-1)))
  }
  cri <- as.numeric(res[1, grep("^cri", names(res))])
  list(err = FALSE, conv = conv, mu = res$mu, sigma = res$sigma, da = res$da,
       logL = res$logL, cri = cri)
}

dp_from <- function(nr_s1, nr_s2) {
  s1 <- nr_s1 + 1/length(nr_s1); s2 <- nr_s2 + 1/length(nr_s2)
  h <- length(s1)/2
  qnorm(sum(s2[1:h])/sum(s2)) - qnorm(sum(s1[1:h])/sum(s1))
}

emit <- function(rows, path) {
  df <- do.call(rbind, lapply(rows, function(r) {
    data.frame(id = r$id, err_conf = r$ec, conv_conf = r$cc,
               mu_conf = r$muc, sigma_conf = r$sgc, da_conf = r$dac, logL_conf = r$llc,
               err_rt = r$er, conv_rt = r$cr, mu_rt = r$mur, sigma_rt = r$sgr,
               da_rt = r$dar, logL_rt = r$llr, dp = r$dp,
               cri_conf = paste(round(r$cric, 6), collapse = "|"),
               cri_rt = paste(round(r$crir, 6), collapse = "|"))
  }))
  write.csv(df, path, row.names = FALSE)
  df
}

## ---------------- Sherman JOCN_1 / JOCN_2 (4-level, inverted mapping) -------
sherman <- read.csv("data_Sherman_2016_JOCN(1).csv")
names(sherman)[1] <- "Subj_idx"   # strip BOM if present
run_sherman <- function(cond, out) {
  d0 <- sherman[sherman$Condition == cond, ]
  rows <- list()
  for (i in sort(unique(d0$Subj_idx))) {
    d <- d0[d0$Subj_idx == i, ]
    S <- d$Stimulus; R <- d$Response; C <- d$Confidence
    cc <- function(s, r, cv) sum(S == s & R == r & C == cv, na.rm = TRUE)
    n1c <- c(cc(1,0,4),cc(1,0,3),cc(1,0,2),cc(1,0,1),cc(1,1,1),cc(1,1,2),cc(1,1,3),cc(1,1,4))
    n2c <- c(cc(0,0,4),cc(0,0,3),cc(0,0,2),cc(0,0,1),cc(0,1,1),cc(0,1,2),cc(0,1,3),cc(0,1,4))
    rb <- ntile_base(d$RT_dec, 4)
    cr <- function(s, r, b) sum(S == s & R == r & rb == b, na.rm = TRUE)
    n1r <- c(cr(1,0,1),cr(1,0,2),cr(1,0,3),cr(1,0,4),cr(1,1,4),cr(1,1,3),cr(1,1,2),cr(1,1,1))
    n2r <- c(cr(0,0,1),cr(0,0,2),cr(0,0,3),cr(0,0,4),cr(0,1,4),cr(0,1,3),cr(0,1,2),cr(0,1,1))
    fc <- fit_full(n1c, n2c); fr <- fit_full(n1r, n2r)
    rows[[length(rows)+1]] <- list(id=i, ec=fc$err, cc=fc$conv, muc=fc$mu, sgc=fc$sigma,
      dac=fc$da, llc=fc$logL, cric=fc$cri, er=fr$err, cr=fr$conv, mur=fr$mu, sgr=fr$sigma,
      dar=fr$da, llr=fr$logL, crir=fr$cri, dp=dp_from(n1c,n2c))
  }
  emit(rows, out)
}
s1 <- run_sherman(1, file.path(out_dir, "sherman_jocn1.csv"))
s2 <- run_sherman(2, file.path(out_dir, "sherman_jocn2.csv"))

## ---------------- Mazor_2020 Detection (3-level, standard mapping) ----------
mazor <- read.csv("data_Mazor_2020(1).csv")
names(mazor)[1] <- "Subj_idx"
md <- mazor[mazor$Condition == "Detection", ]
rows <- list()
for (i in sort(unique(md$Subj_idx))) {
  d <- md[md$Subj_idx == i, ]
  S <- d$Stimulus; R <- d$Response
  Cb <- ntile_base(d$Confidence, 3); Rb <- ntile_base(d$RT_dec, 3)
  cc <- function(s, r, b) sum(S == s & R == r & Cb == b, na.rm = TRUE)
  crr <- function(s, r, b) sum(S == s & R == r & Rb == b, na.rm = TRUE)
  n1c <- c(cc(0,1,3),cc(0,1,2),cc(0,1,1),cc(0,0,1),cc(0,0,2),cc(0,0,3))
  n2c <- c(cc(1,1,3),cc(1,1,2),cc(1,1,1),cc(1,0,1),cc(1,0,2),cc(1,0,3))
  n1r <- c(crr(0,1,1),crr(0,1,2),crr(0,1,3),crr(0,0,3),crr(0,0,2),crr(0,0,1))
  n2r <- c(crr(1,1,1),crr(1,1,2),crr(1,1,3),crr(1,0,3),crr(1,0,2),crr(1,0,1))
  fc <- fit_full(n1c, n2c); fr <- fit_full(n1r, n2r)
  rows[[length(rows)+1]] <- list(id=i, ec=fc$err, cc=fc$conv, muc=fc$mu, sgc=fc$sigma,
    dac=fc$da, llc=fc$logL, cric=fc$cri, er=fr$err, cr=fr$conv, mur=fr$mu, sgr=fr$sigma,
    dar=fr$da, llr=fr$logL, crir=fr$cri, dp=dp_from(n1c,n2c))
}
mz <- emit(rows, file.path(out_dir, "mazor2020.csv"))

cat("DONE. Sherman1 n=", nrow(s1), " Sherman2 n=", nrow(s2), " Mazor n=", nrow(mz), "\n")
