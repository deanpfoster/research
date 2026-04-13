# papers/Makefile — download reference PDFs for "Proper Calibeating"
#
# Usage:  make          (download all freely available papers)
#         make clean    (remove all PDFs)
#         make list     (list all targets)
#
# PDFs are git-ignored via .gitignore in this directory.
# A few paywalled papers are listed in comments only; add manually.

# Download only if the target does not already exist.
define FETCH
@[ -f $@ ] || { echo "  fetching $@"; curl -sL -o $@ "$(1)"; }
endef

# ═══════════════════════════════════════════════════════════════════════
#  FOSTER–HART PAPERS
# ═══════════════════════════════════════════════════════════════════════

# Foster & Hart (2018). "Smooth Calibration, Leaky Forecasts, Finite
#   Recall, and Nash Dynamics." Games & Economic Behavior 109, 271–293.
FH2018 = foster-hart-2018-smooth-calibration.pdf
FH2018_URL = https://arxiv.org/pdf/2210.07152

# Foster & Hart (2021). "Forecast Hedging and Calibration."
#   Journal of Political Economy 129, 3447–3490.
FH2021 = foster-hart-2021-forecast-hedging.pdf
FH2021_URL = https://arxiv.org/pdf/2210.07169

# Foster & Hart (2023). "'Calibeating': Beating Forecasters at Their
#   Own Game." Theoretical Economics 18, 1441–1474.  arXiv:2209.04892.
FH2023 = foster-hart-2023-calibeating.pdf
FH2023_URL = https://arxiv.org/pdf/2209.04892

# Foster & Hart (2026). Errata for "Calibeating".
FH2026E = foster-hart-2026-calibeating-errata.pdf
FH2026E_URL = https://ma.huji.ac.il/hart/papers/calib-beat-errata.pdf

# Foster & Hart (2024). Addendum (Appendix A.10) for "Calibeating".
FH2024A = foster-hart-2024-calibeating-addendum.pdf
FH2024A_URL = https://ma.huji.ac.il/hart/papers/calib-beat-add.pdf

# ═══════════════════════════════════════════════════════════════════════
#  PROPER SCORING RULES — FOUNDATIONAL
# ═══════════════════════════════════════════════════════════════════════

# Gneiting & Raftery (2007). "Strictly Proper Scoring Rules, Prediction,
#   and Estimation." JASA 102, 359–378.
GR2007 = gneiting-raftery-2007-proper-scoring-rules.pdf
GR2007_URL = https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

# Gneiting, Balabdaoui & Raftery (2007). "Probabilistic Forecasts,
#   Calibration and Sharpness." JRSSB 69, 243–268.
GBR2007 = gneiting-balabdaoui-raftery-2007-calibration-sharpness.pdf
GBR2007_URL = https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf

# Schervish (1989). "A General Method for Comparing Probability
#   Assessors." Annals of Statistics 17, 1856–1879.
SCHERV = schervish-1989-comparing-assessors.pdf
SCHERV_URL = https://deanfoster.net/teaching/data_mining/schervish.pdf

# Jose, Nau & Winkler (2008). "Scoring Rules, Generalized Entropy,
#   and Utility Maximization." Working paper.
JNW2008 = jose-nau-winkler-2008-scoring-rules-generalized-entropy.pdf
JNW2008_URL = https://people.duke.edu/~rnau/scoring_rules_and_generalized_entropy.pdf

# Waghmare & Ziegel (2025). "Proper Scoring Rules for Estimation and
#   Forecast Evaluation." Annual Review of Statistics and Its Application.
WZ2025 = waghmare-ziegel-2025-proper-scoring-rules-estimation-forecast-evaluation.pdf
WZ2025_URL = https://arxiv.org/pdf/2504.01781

# Paywalled — add manually if needed:
#   Savage (1971). "Elicitation of Personal Probabilities and
#     Expectations." JASA 66, 783–801.
#   McCarthy (1956). "Measures of the Value of Information."
#     PNAS 42, 654–655.
#   DeGroot & Fienberg (1983). "The Comparison and Evaluation of
#     Forecasters." The Statistician 32, 12–22.
#   Brier (1950). "Verification of Forecasts Expressed in Terms of
#     Probability." Monthly Weather Review 78, 1–3.
#   Selten (1998). "Axiomatic Characterization of the Quadratic
#     Scoring Rule." Experimental Economics 1, 43–61.

# ═══════════════════════════════════════════════════════════════════════
#  CALIBRATION — FOUNDATIONAL
# ═══════════════════════════════════════════════════════════════════════

# Dawid (1982). "The Well-Calibrated Bayesian." JASA 77, 605–613.
DAWID = dawid-1982-well-calibrated-bayesian.pdf
DAWID_URL = https://fitelson.org/seminar/dawid.pdf

# Foster & Vohra (1998). "Asymptotic Calibration."
#   Biometrika 85, 379–390.
FV1998 = foster-vohra-1998-asymptotic-calibration.pdf
FV1998_URL = https://deanfoster.net/research/calibration.pdf

# Foster (1999). "A Proof of Calibration via Blackwell's
#   Approachability Theorem." Games & Economic Behavior 29, 73–78.
F1999 = foster-1999-calibration-blackwell.pdf
F1999_URL = https://deanfoster.net/research/calibration_via_blackwell.pdf

# Olszewski (2015). "Calibration and Expert Testing." Handbook of
#   Game Theory with Economic Applications 4, 949–984.
OLS2015 = olszewski-2015-calibration-expert-testing.pdf
OLS2015_URL = https://faculty.wcas.northwestern.edu/wol737/Hand.pdf

# Blackwell (1956). "An Analog of the Minimax Theorem for Vector
#   Payoffs." Pacific Journal of Mathematics 6, 1–8.
BLACKWELL = blackwell-1956-approachability.pdf
BLACKWELL_URL = https://msp.org/pjm/1956/6-1/pjm-v6-n1-p01-s.pdf

# Allen, Koh, Segers & Ziegel (2024/2025). "Tail Calibration of
#   Probabilistic Forecasts." JASA.
AKSZ2025 = allen-koh-segers-ziegel-2025-tail-calibration.pdf
AKSZ2025_URL = https://arxiv.org/pdf/2407.03167

# Wessel et al. (2025). "Enforcing Tail Calibration When Training
#   Probabilistic Forecast Models." arXiv:2506.13687.
WESSEL2025 = wessel-etal-2025-enforcing-tail-calibration.pdf
WESSEL2025_URL = https://arxiv.org/pdf/2506.13687

# Laeven et al. (2026). "Localizing Strictly Proper Scoring Rules."
#   JASA, early online / accepted manuscript.
LAEVEN2026 = laeven-etal-2026-localizing-strictly-proper-scoring-rules.pdf
LAEVEN2026_URL = https://www.rogerlaeven.nl/gallery/LSPS_JASA_main-accepted.pdf

# ═══════════════════════════════════════════════════════════════════════
#  CALIBRATION — GAME THEORY & LEARNING
# ═══════════════════════════════════════════════════════════════════════

# Hart & Mas-Colell (2000). "A Simple Adaptive Procedure Leading to
#   Correlated Equilibrium." Econometrica 68, 1127–1150.
HMC2000 = hart-mascolell-2000-adaptive-correlated.pdf
HMC2000_URL = http://www.dklevine.com/archive/refs4572.pdf

# Kakade & Foster (2008). "Deterministic Calibration and Nash
#   Equilibrium." JCSS 74, 115–130.  (Originally COLT 2004.)
KF2008 = kakade-foster-2008-deterministic-calibration.pdf
KF2008_URL = https://homes.cs.washington.edu/~sham/papers/gt/calibration.pdf

# Mannor & Stoltz (2010). "A Geometric Proof of Calibration."
#   Mathematics of Operations Research 35, 721–727.
MS2010 = mannor-stoltz-2010-geometric-calibration.pdf
MS2010_URL = https://arxiv.org/pdf/0912.3604

# Perchet (2011). "Internal Regret with Partial Monitoring:
#   Calibration-Based Optimal Algorithms." JMLR 12, 1893–1921.
PERCHET = perchet-2011-calibration-internal-regret.pdf
PERCHET_URL = https://arxiv.org/pdf/1006.1746

# Rakhlin, Sridharan & Tewari (2011). "Online Learning: Beyond
#   Regret." COLT 2011.
RST2011 = rakhlin-sridharan-tewari-2011-beyond-regret.pdf
RST2011_URL = https://arxiv.org/pdf/1011.3168

# Vovk (2009). "Prediction with Expert Advice for the Brier Game."
#   JMLR 10, 2445–2471.
VOVK = vovk-2009-brier-game.pdf
VOVK_URL = https://arxiv.org/pdf/0710.0485

# ═══════════════════════════════════════════════════════════════════════
#  CALIBEATING & EXTENSIONS
# ═══════════════════════════════════════════════════════════════════════

# Lee, Noarov, Pai & Roth (2022). "Online Minimax Multiobjective
#   Optimization: Multicalibeating and Other Applications." NeurIPS 2022.
LNPR2022 = lee-noarov-pai-roth-2022-multicalibeating.pdf
LNPR2022_URL = https://arxiv.org/pdf/2108.03837

# Chen, Huang, Jordan & Luo (2026). "Calibeating Made Simple."
#   arXiv:2603.22167.
CHJL2026 = chen-huang-jordan-luo-2026-calibeating-simple.pdf
CHJL2026_URL = https://arxiv.org/pdf/2603.22167

# Gupta & Ramdas (2023). "Online Platt Scaling with Calibeating."
#   ICML 2023.
GR2023 = gupta-ramdas-2023-platt-calibeating.pdf
GR2023_URL = https://arxiv.org/pdf/2305.00070

# ═══════════════════════════════════════════════════════════════════════
#  MULTICALIBRATION, FAIRNESS & OMNIPREDICTION
# ═══════════════════════════════════════════════════════════════════════

# Hébert-Johnson, Kim, Reingold & Rothblum (2018). "Multicalibration:
#   Calibration for the (Computationally-Identifiable) Masses." ICML 2018.
HJKRR2018 = hebert-johnson-etal-2018-multicalibration.pdf
HJKRR2018_URL = https://arxiv.org/pdf/1711.08513

# Gupta, Jung, Noarov, Pai & Roth (2021). "Online Multivalid Learning:
#   Means, Moments, and Prediction Intervals." ITCS 2022.
GJNPR2021 = gupta-jung-noarov-pai-roth-2021-multivalid.pdf
GJNPR2021_URL = https://arxiv.org/pdf/2101.01739

# Gopalan, Kalai, Reingold, Sharan & Wieder (2022). "Omnipredictors."
#   ITCS 2022.
GKRSW2022 = gopalan-etal-2022-omnipredictors.pdf
GKRSW2022_URL = https://arxiv.org/pdf/2109.05389

# Kleinberg, Mullainathan & Raghavan (2016). "Inherent Trade-Offs in
#   the Fair Determination of Risk Scores." ITCS 2017.
KMR2016 = kleinberg-mullainathan-raghavan-2016-fairness.pdf
KMR2016_URL = https://arxiv.org/pdf/1609.05807

# ═══════════════════════════════════════════════════════════════════════
#  CALIBRATION — COMPLEXITY & LOWER BOUNDS
# ═══════════════════════════════════════════════════════════════════════

# Qiao & Valiant (2021). "Stronger Calibration Lower Bounds via
#   Sidestepping." STOC 2021.
QV2021 = qiao-valiant-2021-calibration-lower-bounds.pdf
QV2021_URL = https://arxiv.org/pdf/2012.03454

# Blasiok, Gopalan, Hu & Nakkiran (2023). "When Does Optimizing a
#   Proper Loss Yield Calibration?" NeurIPS 2023.
BGHN2023 = blasiok-etal-2023-proper-loss-calibration.pdf
BGHN2023_URL = https://arxiv.org/pdf/2305.18764

# ═══════════════════════════════════════════════════════════════════════
#  RECENT CALIBEATING CITATIONS (2023-2026)
# ═══════════════════════════════════════════════════════════════════════

# Hu, Luo, Senapati & Sharan (2025). "Efficient Swap Multicalibration of Elicitable Properties."
#   arXiv:2511.04907.
HLSS2025 = hu-etal-2025-swap-multicalibration.pdf
HLSS2025_URL = https://arxiv.org/pdf/2511.04907

# Luo, Senapati & Sharan (2025). "Simultaneous Swap Regret Minimization via KL-Calibration."
#   arXiv:2502.16387.
LSS2025 = luo-etal-2025-kl-calibration.pdf
LSS2025_URL = https://arxiv.org/pdf/2502.16387

# Marx, Kuleshov & Ermon (2024). "Calibrated Probabilistic Forecasts for Arbitrary Sequences."
#   arXiv:2409.19157.
MKE2024 = marx-etal-2024-arbitrary-sequences.pdf
MKE2024_URL = https://arxiv.org/pdf/2409.19157

# Jain & Perchet (2024). "Calibrated Forecasting and Persuasion."
#   arXiv:2406.15680.
JP2024 = jain-perchet-2024-persuasion.pdf
JP2024_URL = https://arxiv.org/pdf/2406.15680

# Kleinberg, Leme, Schneider & Teng (2023). "U-Calibration: Forecasting for an Unknown Agent."
#   arXiv:2307.00168.
KLST2023 = kleinberg-etal-2023-u-calibration.pdf
KLST2023_URL = https://arxiv.org/pdf/2307.00168

# Berta, Bach & Jordan (2023). "Classifier Calibration with ROC-Regularized Isotonic Regression."
#   arXiv:2311.12436.
BBJ2023 = berta-etal-2023-roc-isotonic.pdf
BBJ2023_URL = https://arxiv.org/pdf/2311.12436

# Okoroafor, Kleinberg & Sun (2023). "Faster Recalibration of an Online Predictor via Approachability."
#   NeurIPS 2023.
OKS2023 = okoroafor-etal-2023-faster-recalibration.pdf
OKS2023_URL = https://proceedings.neurips.cc/paper_files/paper/2023/file/0668d27557adf80f2c5552f14cc2047c9baae155-Paper-Conference.pdf

# Hart (2022). "Calibrated Forecasts: The Minimax Proof."
#   SSRN 3986064.
HART2022 = hart-2022-minimax-proof.pdf
HART2022_URL = https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3986064_code154562.pdf?abstractid=3986064&mirid=1

# ═══════════════════════════════════════════════════════════════════════
#  TARGETS
# ═══════════════════════════════════════════════════════════════════════


ALL_PDFS = \
	$(FH2018) $(FH2021) $(FH2023) $(FH2026E) $(FH2024A) \
	$(GR2007) $(GBR2007) $(SCHERV) $(JNW2008) $(WZ2025) \
	$(DAWID) $(FV1998) $(F1999) $(OLS2015) $(BLACKWELL) \
	$(AKSZ2025) $(WESSEL2025) $(LAEVEN2026) \
	$(HMC2000) $(KF2008) $(MS2010) $(PERCHET) $(RST2011) $(VOVK) \
	$(LNPR2022) $(CHJL2026) $(GR2023) \
	$(HJKRR2018) $(GJNPR2021) $(GKRSW2022) $(KMR2016) \
	$(QV2021) $(BGHN2023) \
	$(HLSS2025) $(LSS2025) $(MKE2024) $(JP2024) $(KLST2023) $(BBJ2023) $(OKS2023) $(HART2022)

.PHONY: all clean list

all: $(ALL_PDFS)
	@echo ""
	@echo "Done. $(words $(ALL_PDFS)) papers fetched."
	@echo "(Paywalled papers listed in comments — add manually.)"

list:
	@echo "Papers ($(words $(ALL_PDFS)) targets):"; \
	for f in $(ALL_PDFS); do echo "  $$f"; done

clean:
	rm -f $(ALL_PDFS)

# ═══════════════════════════════════════════════════════════════════════
#  DOWNLOAD RULES
# ═══════════════════════════════════════════════════════════════════════

$(FH2018):  ; $(call FETCH,$(FH2018_URL))
$(FH2021):  ; $(call FETCH,$(FH2021_URL))
$(FH2023):  ; $(call FETCH,$(FH2023_URL))
$(FH2026E): ; $(call FETCH,$(FH2026E_URL))
$(FH2024A): ; $(call FETCH,$(FH2024A_URL))
$(GR2007):  ; $(call FETCH,$(GR2007_URL))
$(GBR2007): ; $(call FETCH,$(GBR2007_URL))
$(SCHERV):  ; $(call FETCH,$(SCHERV_URL))
$(JNW2008): ; $(call FETCH,$(JNW2008_URL))
$(WZ2025):  ; $(call FETCH,$(WZ2025_URL))
$(DAWID):   ; $(call FETCH,$(DAWID_URL))
$(FV1998):  ; $(call FETCH,$(FV1998_URL))
$(F1999):   ; $(call FETCH,$(F1999_URL))
$(OLS2015): ; $(call FETCH,$(OLS2015_URL))
$(BLACKWELL): ; $(call FETCH,$(BLACKWELL_URL))
$(AKSZ2025): ; $(call FETCH,$(AKSZ2025_URL))
$(WESSEL2025): ; $(call FETCH,$(WESSEL2025_URL))
$(LAEVEN2026): ; $(call FETCH,$(LAEVEN2026_URL))
$(HMC2000): ; $(call FETCH,$(HMC2000_URL))
$(KF2008):  ; $(call FETCH,$(KF2008_URL))
$(MS2010):  ; $(call FETCH,$(MS2010_URL))
$(PERCHET): ; $(call FETCH,$(PERCHET_URL))
$(RST2011): ; $(call FETCH,$(RST2011_URL))
$(VOVK):    ; $(call FETCH,$(VOVK_URL))
$(LNPR2022):  ; $(call FETCH,$(LNPR2022_URL))
$(CHJL2026):  ; $(call FETCH,$(CHJL2026_URL))
$(GR2023):    ; $(call FETCH,$(GR2023_URL))
$(HJKRR2018): ; $(call FETCH,$(HJKRR2018_URL))
$(GJNPR2021): ; $(call FETCH,$(GJNPR2021_URL))
$(GKRSW2022): ; $(call FETCH,$(GKRSW2022_URL))
$(KMR2016):   ; $(call FETCH,$(KMR2016_URL))
$(QV2021):    ; $(call FETCH,$(QV2021_URL))
$(BGHN2023):  ; $(call FETCH,$(BGHN2023_URL))


$(HLSS2025):  ; $(call FETCH,$(HLSS2025_URL))
$(LSS2025):   ; $(call FETCH,$(LSS2025_URL))
$(MKE2024):   ; $(call FETCH,$(MKE2024_URL))
$(JP2024):    ; $(call FETCH,$(JP2024_URL))
$(KLST2023):  ; $(call FETCH,$(KLST2023_URL))
$(BBJ2023):   ; $(call FETCH,$(BBJ2023_URL))
$(OKS2023):   ; $(call FETCH,$(OKS2023_URL))
$(HART2022):  ; $(call FETCH,$(HART2022_URL))
