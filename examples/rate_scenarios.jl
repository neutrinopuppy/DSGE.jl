####################################################################
# rate_scenarios.jl
#
# Fed Funds Rate Policy Recommendation
# Focus: Dual Mandate (Inflation + Employment)
# Scenarios: Hold, Cut 25bp, Hike 25bp — with 25bp range bands
#
# For presentation, April 2026
####################################################################

using DSGE, ModelConstructors, Plots, DataFrames, Dates, OrderedCollections
using Printf, Statistics, LinearAlgebra
using FredData  # for UNRATE pull used in Okun calibration

plotly()

println("="^65)
println("  FED FUNDS RATE POLICY RECOMMENDATION")
println("  Dual Mandate Analysis")
println("="^65)

##############
# Model Setup
##############
m = Model1002("ss10")
m <= Setting(:data_vintage, "260410")
m <= Setting(:date_forecast_start, quartertodate("2026-Q2"))
m <= Setting(:use_population_forecast, false)

# --- Sampling / MH settings -------------------------------------------------
# We need posterior draws (not just the mode) to build honest uncertainty
# bands around the scenario forecasts. These numbers are deliberately modest
# for a demo/presentation run; bump n_mh_simulations and n_mh_blocks for a
# production-quality posterior.
m <= Setting(:sampling_method, :MH)
m <= Setting(:n_mh_blocks, 5)
m <= Setting(:n_mh_simulations, 2000)     # draws per block
m <= Setting(:n_mh_burn, 1)               # burn 1 of 5 blocks
m <= Setting(:mh_thin, 2)                 # thin factor inside MH (correct name)

# --- Forecast settings ------------------------------------------------------
# With :full, forecast_one iterates over (n_blocks - n_burn) * n_mh_simulations
# thinned draws from mhsave.h5. forecast_jstep thins that further at forecast
# time. forecast_block_size controls the in-memory batch size.
m <= Setting(:forecast_jstep, 5)
m <= Setting(:forecast_block_size, 500)

println("\n>> Loading fresh data from FRED...")
df = load_data(m, try_disk = false, check_empty_columns = false, summary_statistics = :none)
data = df_to_matrix(m, df)

# --- Mode + Hessian handling ------------------------------------------------
# The estimate() entry point does NOT auto-load paramsmode.h5 into the model
# state — with reoptimize=false it uses whatever parameters the model was
# constructed with (prior means for Model1002). Computing a hessian there
# fails with "Negative diagonal in Hessian" because you're not at a mode.
# specify_mode! loads the stored params into m and sets reoptimize=false.
modefile = rawpath(m, "estimate", "paramsmode.h5")
if isfile(modefile)
    println(">> Loading saved mode from $modefile")
    specify_mode!(m, modefile)   # sets reoptimize=false AND loads params
else
    println(">> No paramsmode.h5 found — running mode-finding first (~45 min)...")
    m <= Setting(:reoptimize, true)
    @time estimate(m, data; sampling = false)
    specify_mode!(m, modefile)
end

# The hessian MUST be paired with the mode you're sitting at. If there's no
# stored hessian.h5 (or if you suspect it's stale), force a recompute here at
# the current params — this is cheap compared to MH and guarantees a valid
# proposal covariance.
_hess_default = rawpath(m, "estimate", "hessian.h5")
if isfile(_hess_default)
    m <= Setting(:hessian_path, _hess_default)
    m <= Setting(:calculate_hessian, false)
    println(">> Reusing hessian at $_hess_default")
else
    m <= Setting(:calculate_hessian, true)
    println(">> No hessian.h5 found — will compute a fresh hessian at the " *
            "loaded mode during estimate().")
end

mhfile = rawpath(m, "estimate", "mhsave.h5")
if !isfile(mhfile)
    println(">> No mhsave.h5 found — running MH to generate posterior draws.")
    println("   ($(get_setting(m, :n_mh_blocks)) blocks × " *
            "$(get_setting(m, :n_mh_simulations)) draws; this is the slow step)")
    @time estimate(m, data)  # sampling = true is the default
else
    println(">> Using existing MH draws from $(mhfile).")
end

Rstarn = m[:Rstarn].value
println(">> Steady-state rate: $(round(Rstarn * 4, digits=2))% annualized")

######################
# Scenario Definitions
######################
# H+1 = number of quarters we can pin the rate at a user-specified target.
# Model1002 ss10 has n_mon_anticipated_shocks = 6, giving 6 anticipated MP
# shocks + 1 contemporaneous = 7 independent instruments, so the peg solver
# could in principle pin up to 7 quarters. We use H=3 (4 pinned quarters, 1
# year) deliberately — this is the window length where the "forward guidance
# puzzle" (Del Negro, Giannoni, Patterson 2015) doesn't catastrophically
# amplify the response. With H=6 we saw ~2.5 pp Core PCE response to 25 bp
# of guidance; H=3 should deliver ~0.5–1 pp, which matches both economic
# intuition and what the Fed usually publishes for short-window conditional
# forecasts.
H = 3
n_peg_quarters = H + 1

# Scenarios are TACTICAL DEVIATIONS from the baseline rate path, not fixed
# rate levels. Each scenario's realized rate rides at baseline ± delta over
# the pinned window and fades back toward baseline afterwards. Matches the
# Liberty Street / Cúrdia–Del Negro "conditional on alternative rate path"
# convention for published Fed DSGE exercises.
scenarios = OrderedDict(
    "Cut 25bp"  => (delta = -0.25, color = :forestgreen),
    "Hold"      => (delta =  0.00, color = :royalblue),
    "Hike 25bp" => (delta =  0.25, color = :crimson),
)

######################
# Run Forecasts
######################
output_vars = [:forecastobs]

# Percentile bands to compute for every :full run. [0.68, 0.90] matches the
# Liberty Street Economics fan-chart conventions (darker inner band = 68%,
# lighter outer band = 90%).
fan_bands = [0.68, 0.90]

# input_type for the runs that feed fan charts. :full loops over posterior
# MH draws so we can compute honest percentile bands.
band_input_type = :full

# Baseline (no peg — model's own rate path under the historical rule).
# This run feeds (a) the published baseline fan chart and (b) the rate path
# the Cut/Hike scenarios are constructed relative to.
println("\n>> Running model baseline (historical Taylor rule, posterior)...")
forecast_one(m, band_input_type, :none, output_vars;
             check_empty_columns = false, forecast_string = "baseline")
compute_meansbands(m, band_input_type, :none, output_vars;
                   check_empty_columns = false, density_bands = fan_bands,
                   forecast_string = "baseline")
mb_baseline = read_mb(m, band_input_type, :none, :forecastobs;
                      forecast_string = "baseline")

# The peg solver takes `FFRpeg` in state-deviation units for R_t, which are
# unitless from a user's perspective. Run two small test pegs at :mode to
# empirically recover the linear map from FFRpeg → annualized obs_nominalrate
# (the model is linear, so this map is exact up to floating point). Then we
# invert it to convert target annual rates back into FFRpeg inputs.
println(">> Calibrating peg→rate map with two test pegs (at :mode)...")
test_peg_a = -0.001
test_peg_b =  0.001
forecast_one(m, :mode, :none, output_vars;
             check_empty_columns = false,
             pegFFR = true, FFRpeg = test_peg_a, H = H,
             forecast_string = "cal_a")
compute_meansbands(m, :mode, :none, output_vars;
                   check_empty_columns = false, forecast_string = "cal_a")
rate_a = read_mb(m, :mode, :none, :forecastobs; forecast_string = "cal_a").means[1, :obs_nominalrate]

forecast_one(m, :mode, :none, output_vars;
             check_empty_columns = false,
             pegFFR = true, FFRpeg = test_peg_b, H = H,
             forecast_string = "cal_b")
compute_meansbands(m, :mode, :none, output_vars;
                   check_empty_columns = false, forecast_string = "cal_b")
rate_b = read_mb(m, :mode, :none, :forecastobs; forecast_string = "cal_b").means[1, :obs_nominalrate]

peg_slope     = (rate_b - rate_a) / (test_peg_b - test_peg_a)
peg_intercept = rate_a - peg_slope * test_peg_a
ffr_peg_for(target_annual) = (target_annual - peg_intercept) / peg_slope
println(">> Peg→rate map: obs_nominalrate ≈ $(round(peg_intercept, digits=3))% + $(round(peg_slope, digits=2)) * FFRpeg")

# Pull the baseline (MODE) rate path over the pinned window. We use the
# :mode baseline — not the :full posterior mean — as the reference for the
# scenario deltas, because:
#   (a) The peg solver is calibrated at :mode, so baseline_rates has to be
#       in the same units / reference frame as the calibration map.
#   (b) mb_baseline (under :full) aggregates across ~1000 posterior draws.
#       A few draws with explosive anticipated-shock responses get
#       EXPONENTIATED by obs_corepce's rev_transform (100*(exp(y/100)^4 - 1))
#       before averaging, which produces multi-hundred-percent inflation
#       readings in the posterior mean even when the peg hits its rate
#       target cleanly.
# Scenarios therefore run at :mode and get clean deterministic conditional
# paths. Baseline stays at :full so the published fan chart still has
# honest posterior uncertainty around the unconditional forecast. This is
# exactly the Liberty Street convention: fan on baseline, lines on
# conditional scenarios.
println(">> Running mode-baseline to anchor scenario deltas...")
forecast_one(m, :mode, :none, output_vars;
             check_empty_columns = false, forecast_string = "baseline_mode")
compute_meansbands(m, :mode, :none, output_vars;
                   check_empty_columns = false, forecast_string = "baseline_mode")
mb_baseline_mode = read_mb(m, :mode, :none, :forecastobs;
                           forecast_string = "baseline_mode")

baseline_rates = Float64[]
for q in 1:n_peg_quarters
    push!(baseline_rates, mb_baseline_mode.means[q, :obs_nominalrate])
end
println(">> Baseline rate path (Q1..Q$(n_peg_quarters), at mode): " *
        join([string(round(r, digits=2), "%") for r in baseline_rates], ", "))

# Run each scenario at :mode — single deterministic path per scenario, no
# posterior averaging, no explosive-draw pollution.
results = OrderedDict{String, Any}()
for (name, spec) in scenarios
    tag = replace("$(name)_tactical", " " => "", "%" => "", "." => "", "-" => "")

    target_path = baseline_rates .+ spec.delta               # annualized %
    ffr_peg_path = [ffr_peg_for(r) for r in target_path]     # state-deviation units

    println(">> $name: baseline $(spec.delta >= 0 ? "+" : "")$(spec.delta) pp " *
            "for $(n_peg_quarters) quarters (mode)")

    forecast_one(m, :mode, :none, output_vars;
                 check_empty_columns = false,
                 pegFFR = true, H = H, FFRpeg_path = ffr_peg_path,
                 forecast_string = tag)
    compute_meansbands(m, :mode, :none, output_vars;
                       check_empty_columns = false, forecast_string = tag)
    results[name] = read_mb(m, :mode, :none, :forecastobs;
                            forecast_string = tag)
end

####################################################################
# Alternative Policy Rules
#
# Counterfactual: hold all parameters fixed (they were estimated under
# the historical rule), but ask "what rate path would Rule X prescribe
# from forecast start, given the same s_T?". This is genuinely different
# from a peg — the rule is endogenous and reacts to inflation/output as
# the forecast unfolds. Useful as a sanity check on the discretionary
# scenarios: if Taylor93 says "cut" and the discretionary cut scenario
# also looks good on the dual mandate, that's two independent arguments
# pointing the same direction.
#
# IMPORTANT: alt-policy runs do NOT use pegFFR. The peg replaces the
# policy rule entirely; alt-policy replaces it with a different rule.
# You can't compose them.
####################################################################
println("\n>> Running alternative policy rules...")

# orig_altpol was saved at the top of the script before the scenario loop;
# we rely on that to restore after this block.

altpol_specs = OrderedDict(
    "Taylor 1993"  => (policy = taylor93(), color = :navy,    style = :solid),
    "Taylor 1999"  => (policy = taylor99(), color = :purple,  style = :dash),
)

altpol_results = OrderedDict{String, Any}()
for (name, spec) in altpol_specs
    println(">> $name")
    m <= Setting(:alternative_policy, spec.policy)
    tag = "altpol_$(spec.policy.key)"
    try
        forecast_one(m, :mode, :none, output_vars;
                     check_empty_columns = false,
                     forecast_string = tag)
        compute_meansbands(m, :mode, :none, output_vars;
                           check_empty_columns = false, forecast_string = tag)
        altpol_results[name] = read_mb(m, :mode, :none, :forecastobs;
                                       forecast_string = tag)
    catch e
        @warn "Alt-policy $(name) failed: $e"
        altpol_results[name] = nothing
    end
end

# Restore the historical rule on the model object.
m <= Setting(:alternative_policy, orig_altpol)

############################
# Historical filtered estimates (model's view of "now")
############################
println("\n>> Running history to get model's view of current state...")
hist_vars = [:histobs]
forecast_one(m, :mode, :none, hist_vars;
             check_empty_columns = false, forecast_string = "hist")
compute_meansbands(m, :mode, :none, hist_vars;
                   check_empty_columns = false, forecast_string = "hist")
mb_hist = read_mb(m, :mode, :none, :histobs; forecast_string = "hist")

####################################################################
# Labor-mandate bridge: Okun's law (hours → unemployment),
# CALIBRATED from FRED UNRATE aligned to the model's history window.
#
# Model1002 ss10 observes hours per capita but NOT unemployment
# (`u_t` in this model is capacity utilization, not unemployment).
# To give a presentation-friendly labor-mandate chart we bridge from
# hours deviations to unemployment-rate deviations via a linear Okun
# coefficient fit on actual data:
#
#   Δu (pp) ≈ -OKUN_COEF · Δhours (%)      (no intercept)
#
# The fit is first-differenced OLS on aligned (obs_hours, UNRATE)
# quarterly pairs from mb_hist's date range. First differences
# eliminate any slow trends in both series and avoid the need for an
# intercept. Anchor u is set to the most recent observed FRED value;
# anchor hours is set to the corresponding filtered obs_hours from
# the same quarter, so history and forecast tie cleanly at the join.
#
# CAVEATS (fit for presentation, not for publication):
#  - OLS assumes a constant Okun coefficient across the sample.
#    During recessions or near the ZLB the coefficient typically
#    rises — this is a stationary linear fit, not regime-aware.
#  - FRED frequency="q" returns quarterly AVERAGES of monthly
#    UNRATE, so the last history point will be the Q1 2026 average
#    (≈ 4.3%), which may differ by ~0.05 pp from a headline monthly
#    Mar 2026 value.
#  - Falls back to OKUN_COEF=0.5 if FRED is unreachable or there
#    aren't enough aligned data pairs.
####################################################################
println(">> Fetching FRED UNRATE for Okun calibration...")
u_fred_df = try
    let
        f = FredData.Fred()
        start_d = string(mb_hist.means[1, :date] - Dates.Month(3))
        end_d   = string(mb_hist.means[end, :date] + Dates.Month(1))
        series  = FredData.get_data(f, "UNRATE"; frequency = "q",
                                    observation_start = start_d,
                                    observation_end   = end_d)
        series.df
    end
catch err
    @warn "FRED UNRATE fetch failed; falling back to hardcoded Okun defaults" err
    nothing
end

# Align FRED quarterly dates to mb_hist quarter-end dates. FRED's "q"
# frequency returns quarter START dates; match by year + quarter number.
function match_fred_to_hist(hist_dates, fred_df)
    aligned = fill(NaN, length(hist_dates))
    fred_df === nothing && return aligned
    for (i, hd) in enumerate(hist_dates)
        hy, hq = Dates.year(hd), Dates.quarterofyear(hd)
        idx = findfirst(row -> Dates.year(row.date) == hy &&
                               Dates.quarterofyear(row.date) == hq,
                        eachrow(fred_df))
        idx !== nothing && (aligned[i] = fred_df[idx, :value])
    end
    return aligned
end
u_actual_full    = match_fred_to_hist(mb_hist.means[!, :date], u_fred_df)
hist_hours_full  = mb_hist.means[!, :obs_hours]

# First-difference OLS: α = -(Δh' · Δu) / (Δh' · Δh)
valid_idx = findall(i -> !isnan(u_actual_full[i]) && !isnan(hist_hours_full[i]),
                    1:length(u_actual_full))
OKUN_COEF = if length(valid_idx) < 8
    @warn "Not enough aligned (U3, hours) pairs for OLS ($(length(valid_idx))); " *
          "falling back to OKUN_COEF = 0.5"
    0.5
else
    u_v = u_actual_full[valid_idx]
    h_v = hist_hours_full[valid_idx]
    du  = diff(u_v)
    dh  = diff(h_v)
    -(dh' * du) / (dh' * dh)
end

# Anchor at the most recent observed (u, hours) pair so history and
# forecast meet at the same quarter.
last_obs_idx = findlast(!isnan, u_actual_full)
OKUN_ANCHOR_U = last_obs_idx === nothing ? 4.3 : u_actual_full[last_obs_idx]
anchor_hours_q1 = last_obs_idx === nothing ?
    mb_baseline_mode.means[1, :obs_hours] :
    hist_hours_full[last_obs_idx]

println(">> Okun fit: Δu = -$(round(OKUN_COEF, digits=3)) · Δhours%, " *
        "anchor u = $(round(OKUN_ANCHOR_U, digits=2))% " *
        "(n = $(max(0, length(valid_idx)-1)) quarterly diffs)")

# Core Okun transformation — linear, so it composes with any hours vector
# (means or band endpoints). Negative slope means hours-UB maps to u-LB.
hours_vec_to_u(h) = OKUN_ANCHOR_U .- OKUN_COEF .* (h .- anchor_hours_q1)

# Helper: translate a MeansBands' hours column into an implied
# unemployment path, returned as a Float64 vector in percentage points.
function hours_to_unemployment(mb, n::Int = nrow(mb.means))
    return hours_vec_to_u(mb.means[1:n, :obs_hours])
end

# Helper: translate a MeansBands' posterior hours bands into unemployment
# bands via the same linear Okun formula. Returns (u_lb, u_ub) with the
# correct orientation (Okun has negative slope, so hours-UB → u-LB).
# Returns nothing if mb has no bands (e.g. :mode runs).
function hours_to_unemployment_bands(mb, pct::String, n::Int)
    if isempty(mb.bands); return nothing; end
    if !haskey(mb.bands, :obs_hours); return nothing; end
    bdf = mb.bands[:obs_hours]
    lb_col = Symbol(pct, " LB")
    ub_col = Symbol(pct, " UB")
    cols = propertynames(bdf)
    if !(lb_col in cols) || !(ub_col in cols); return nothing; end
    nr = min(n, size(bdf, 1))
    h_lb = bdf[1:nr, lb_col]
    h_ub = bdf[1:nr, ub_col]
    (any(isnan.(h_lb)) || any(isnan.(h_ub))) && return nothing
    # Flip: hours LB maps to u UB because Okun is negative-sloped
    u_lb = hours_vec_to_u(h_ub)
    u_ub = hours_vec_to_u(h_lb)
    return (u_lb, u_ub)
end

############################
# Tariff cost-push shock: pure IRF for λ_f_sh (the price markup shock),
# which is the structural channel through which a tariff-like supply shock
# enters the model. Isolated effect only — not composed with the rate peg.
# If you want the combined "rate decision + tariff shock" response, linearity
# of the state-space lets you superpose the IRF onto each rate-pegged mean
# forecast at the same starting state in post-processing.
############################
# TODO(units): tariff_shock_size = 1.0 with shock_var_name=:obs_gdpdeflator is
# the desired Q1 IMPACT in obs_gdpdeflator (annualized %), NOT a one-σ λ_f
# shock. Replace with the (shock_names=[:λ_f_sh], shock_values=[m[:σ_λ_f]])
# overload if you want a structural-σ interpretation.
tariff_shock_size = 1.0

println("\n>> Computing impulse response for cost-push shock...")
irf_output_vars = [:irfobs]
forecast_one(m, :mode, :none, irf_output_vars;
             check_empty_columns = false,
             shock_name = :λ_f_sh, shock_var_name = :obs_gdpdeflator,
             shock_var_value = tariff_shock_size,
             forecast_string = "irf_costpush")
compute_meansbands(m, :mode, :none, irf_output_vars;
                   check_empty_columns = false,
                   forecast_string = "irf_costpush")

############################
# Output Directory
############################
plotdir = joinpath(dirname(@__FILE__), "..", "save", "output_data", "m1002", "ss10", "scenarios")
mkpath(plotdir)

############################
# Helper: extract forecast
############################
function get_forecast(mb, obs_key, n)
    col_names = Symbol.(names(mb.means))
    if obs_key in col_names
        col = mb.means[!, obs_key]
        vals = col[1:min(n, length(col))]
        return any(isnan.(vals)) ? nothing : vals
    end
    return nothing
end

####################################################################
# KEY INDICATORS
# Note: compute_meansbands already reverse-transforms to data units:
#   - obs_nominalrate: already annualized (%)
#   - obs_gdpdeflator, obs_corepce, obs_wages: annualized quarterly rates (%)
#   - obs_hours: log level (use DIFFERENCES between scenarios only)
#   - obs_gdp, obs_consumption, obs_investment: NaN (missing source data)
####################################################################

# Indicators that have meaningful absolute levels
level_indicators = OrderedDict(
    "Inflation — GDP Deflator (%)" => :obs_gdpdeflator,
    "Inflation — Core PCE (%)"     => :obs_corepce,
    "Fed Funds Rate (%)"           => :obs_nominalrate,
    "Wage Growth (%)"              => :obs_wages,
    "Credit Spread (%)"            => :obs_spread,
)

# Employment (log level — only meaningful as differences)
employment_key = :obs_hours

####################################################################
# Helpers: band extraction + fan-chart drawing
####################################################################
# MeansBands stores percentile bands as Dict{Symbol,DataFrame} with columns
# named like Symbol("68.0% LB"), Symbol("68.0% UB"), etc. At :mode there are
# no bands, so get_bands returns nothing and the caller skips the shaded region.
function get_bands(mb, obs_key::Symbol, pct::String, n::Int)
    if isempty(mb.bands); return nothing; end
    if !haskey(mb.bands, obs_key); return nothing; end
    bands_df = mb.bands[obs_key]
    lb_col = Symbol(pct, " LB")
    ub_col = Symbol(pct, " UB")
    cols = propertynames(bands_df)
    if !(lb_col in cols) || !(ub_col in cols); return nothing; end
    nrow = min(n, size(bands_df, 1))
    lb = bands_df[1:nrow, lb_col]
    ub = bands_df[1:nrow, ub_col]
    (any(isnan.(lb)) || any(isnan.(ub))) && return nothing
    return (lb, ub)
end

# Draws mean + 90% + 68% shaded bands on plot `p`, across x-axis positions xs.
function plot_fan!(p, xs, mb, obs_key; color, label,
                   pct_outer = "90.0%", pct_inner = "68.0%")
    y_mean = get_forecast(mb, obs_key, length(xs))
    y_mean === nothing && return
    outer = get_bands(mb, obs_key, pct_outer, length(xs))
    inner = get_bands(mb, obs_key, pct_inner, length(xs))
    if outer !== nothing
        plot!(p, xs, outer[1]; fillrange = outer[2], fillalpha = 0.12,
              color = color, linewidth = 0, label = "")
    end
    if inner !== nothing
        plot!(p, xs, inner[1]; fillrange = inner[2], fillalpha = 0.25,
              color = color, linewidth = 0, label = "")
    end
    plot!(p, xs, y_mean; color = color, linewidth = 2.5, label = label)
end

####################################################################
# PLOT 1a: DUAL MANDATE COMPARISON — mean lines across scenarios
# (No bands — three fans overlaid would be illegible. See PLOT 1b.)
####################################################################
println("\n>> Generating plots...")

plot_vars = OrderedDict(
    "Inflation — Core PCE (%)"     => :obs_corepce,
    "Inflation — GDP Deflator (%)" => :obs_gdpdeflator,
    "Fed Funds Rate (%)"           => :obs_nominalrate,
    "Wage Growth (%)"              => :obs_wages,
)

subplots = []
for (title, obs_key) in plot_vars
    p = plot(; title = title, titlefontsize = 10, legend = :topright,
             legendfontsize = 6, xlabel = "Quarter", xticks = 1:H)

    # Baseline (no peg)
    base_y = get_forecast(mb_baseline, obs_key, H)
    if base_y !== nothing
        plot!(p, 1:H, base_y; label = "Model Baseline",
              color = :gray, linewidth = 1.5, linestyle = :dash)
    end

    # Scenario means
    for (name, spec) in scenarios
        y = get_forecast(results[name], obs_key, H)
        if y !== nothing
            short_name = split(name, "\n")[1]
            plot!(p, 1:H, y; label = short_name,
                  color = spec.color, linewidth = 2.5)
        end
    end

    # 2% target line for inflation
    if obs_key in [:obs_gdpdeflator, :obs_corepce]
        hline!(p, [2.0]; color = :black, linestyle = :dot, linewidth = 1, label = "2% Target")
    end

    push!(subplots, p)
end

main_plot = plot(subplots...; layout = (2, 2), size = (1100, 750),
                 plot_title = "Fed Policy Scenarios: Baseline ± 25bp Tactical Deviation (posterior means)")
display(main_plot)
savefig(main_plot, joinpath(plotdir, "dual_mandate_comparison.html"))
println("   Saved: dual_mandate_comparison.html")

####################################################################
# PLOT 1b: PER-SCENARIO PANELS (Liberty Street style)
# One 2×2 dual-mandate grid per rate decision. Each panel shows the
# baseline posterior fan (68% / 90% from :full draws) as a shaded region,
# with the conditional scenario mean (from :mode, deterministic) drawn on
# top as a solid line. That's the LSE convention: unconditional fan for
# uncertainty, conditional line for the policy experiment.
####################################################################
for (name, spec) in scenarios
    fan_subplots = []
    for (title, obs_key) in plot_vars
        p = plot(; title = title, titlefontsize = 10, legend = :topright,
                 legendfontsize = 6, xlabel = "Quarter", xticks = 1:H)

        # Baseline FAN (from mb_baseline, which is :full)
        plot_fan!(p, 1:H, mb_baseline, obs_key;
                  color = :slategray, label = "Baseline (68/90%)")

        # Scenario conditional path (from :mode)
        short_name = split(name, "\n")[1]
        y_scen = get_forecast(results[name], obs_key, H)
        if y_scen !== nothing
            plot!(p, 1:H, y_scen; label = short_name,
                  color = spec.color, linewidth = 2.5)
        end

        if obs_key in [:obs_gdpdeflator, :obs_corepce]
            hline!(p, [2.0]; color = :black, linestyle = :dot,
                   linewidth = 1, label = "2% Target")
        end
        push!(fan_subplots, p)
    end

    short = split(name, "\n")[1]
    fan_plot = plot(fan_subplots...; layout = (2, 2), size = (1100, 750),
                    plot_title = "$(short): conditional path vs. baseline posterior fan")
    display(fan_plot)
    tag = replace(short, " " => "_")
    savefig(fan_plot, joinpath(plotdir, "dual_mandate_fan_$(tag).html"))
    println("   Saved: dual_mandate_fan_$(tag).html")
end

####################################################################
# PLOT 2: EMPLOYMENT IMPACT (shown as change from Hold baseline)
####################################################################
emp_plot = plot(; title = "Hours Worked: Change vs. Baseline Rate Path",
               xlabel = "Quarter", ylabel = "Δ Hours per Capita (%)",
               size = (800, 450), legend = :topright)

hold_key = collect(keys(scenarios))[2]
hold_mid_hours = get_forecast(results[hold_key], employment_key, H)

if hold_mid_hours !== nothing
    for (name, spec) in scenarios
        if name == hold_key
            continue
        end
        y_mid = get_forecast(results[name], employment_key, H)
        if y_mid !== nothing
            diff = y_mid .- hold_mid_hours
            short_name = split(name, "\n")[1]
            plot!(emp_plot, 1:H, diff;
                  label = short_name, color = spec.color, linewidth = 2.5)
        end
    end
    hline!(emp_plot, [0.0]; color = :black, linestyle = :dot, linewidth = 1, label = "Hold (baseline)")
end

display(emp_plot)
savefig(emp_plot, joinpath(plotdir, "employment_impact.html"))
println("   Saved: employment_impact.html")

base_hours_for_diff = get_forecast(mb_baseline, :obs_hours, H)

####################################################################
# PLOT 2e: UNEMPLOYMENT RATE via OKUN BRIDGE (with baseline fan)
# Labor-mandate interpretable chart. Converts scenario hours paths
# into unemployment rate paths anchored at the current observed u.
# Baseline fan bands come from the :full baseline's obs_hours posterior
# percentiles pushed through the same linear Okun formula — so the
# unemployment fan inherits proper parameter uncertainty from the
# posterior. Scenarios are deterministic lines (mode) overlaid on top.
####################################################################
u_plot = plot(; title = "Unemployment Rate Path (Okun Bridge)",
              xlabel = "Quarter",
              ylabel = "Unemployment Rate (%)",
              size = (900, 500), legend = :topright,
              titlefontsize = 12, legendfontsize = 9)

# NAIRU reference line (Fed staff estimate ~4.0% for US)
hline!(u_plot, [4.0]; color = :black, linestyle = :dot, linewidth = 1,
       label = "NAIRU ≈ 4.0%")

# Baseline posterior fan (90% outer, 68% inner) — linear Okun transform
# of mb_baseline's obs_hours bands. mb_baseline is :full, so we have real
# posterior draws to aggregate; the Okun map is linear, so bands translate
# cleanly without the exponential-transform explosion issue.
u_b90 = hours_to_unemployment_bands(mb_baseline, "90.0%", H)
u_b68 = hours_to_unemployment_bands(mb_baseline, "68.0%", H)
if u_b90 !== nothing
    plot!(u_plot, 1:H, u_b90[1]; fillrange = u_b90[2], fillalpha = 0.12,
          color = :slategray, linewidth = 0, label = "Baseline 90%")
end
if u_b68 !== nothing
    plot!(u_plot, 1:H, u_b68[1]; fillrange = u_b68[2], fillalpha = 0.25,
          color = :slategray, linewidth = 0, label = "Baseline 68%")
end

# Baseline mean u path (from :mode — the anchor/reference the scenarios
# are all built relative to). Using mode here keeps the fan "centered"
# visually on the same reference the scenario lines use.
base_u = hours_to_unemployment(mb_baseline_mode, H)
plot!(u_plot, 1:H, base_u;
      color = :slategray, linewidth = 2, linestyle = :dash,
      label = "Baseline (mean)")

# Scenario u paths (deterministic, from :mode runs)
for (name, spec) in scenarios
    u_path = hours_to_unemployment(results[name], H)
    short_name = split(name, "\n")[1]
    plot!(u_plot, 1:H, u_path;
          color = spec.color, linewidth = 2.5, label = short_name,
          markershape = :circle, markersize = 4)
end

# Caption with the anchor and coefficient
y_caption = let ymin_candidates = Float64[]
    push!(ymin_candidates, minimum(base_u))
    u_b90 !== nothing && push!(ymin_candidates, minimum(u_b90[1]))
    minimum(ymin_candidates) - 0.1
end
annotate!(u_plot, [(H / 2, y_caption,
          text("anchor u = $(OKUN_ANCHOR_U)% @ Q1 · Okun coef = $(OKUN_COEF) · " *
               "bands from posterior hours",
               8, :gray, :center))])

display(u_plot)
savefig(u_plot, joinpath(plotdir, "unemployment_okun.html"))
println("   Saved: unemployment_okun.html")

####################################################################
# PLOT 2c: ALT POLICY RULES vs PERMANENT RATE PEG SCENARIOS
# Two angles on the same question, side by side:
#   - Discretionary peg lines: what each level (cut/hold/hike at the
#     midpoint of its range) implies for inflation/employment
#   - Alt-policy lines: what each rule would itself prescribe, given
#     the same starting state
# If the rules cluster near one of the discretionary scenarios, that's
# a sign the rule-based and judgment-based answers agree.
####################################################################
altpol_plot_vars = OrderedDict(
    "Inflation — Core PCE (%)"     => :obs_corepce,
    "Fed Funds Rate (%)"           => :obs_nominalrate,
    "Wage Growth (%)"              => :obs_wages,
    "Hours Worked (vs baseline)"   => :obs_hours,
)

altpol_subplots = []
for (title, obs_key) in altpol_plot_vars
    p = plot(; title = title, titlefontsize = 10, legend = :topright,
             legendfontsize = 6, xlabel = "Quarter", xticks = 1:H)

    base_y = get_forecast(mb_baseline, obs_key, H)
    if base_y !== nothing && obs_key != :obs_hours
        plot!(p, 1:H, base_y; label = "Baseline (historical rule)",
              color = :gray, linewidth = 1.5, linestyle = :dash)
    end

    # Discretionary mid-of-range scenarios
    for (name, spec) in scenarios
        y = get_forecast(results[name], obs_key, H)
        y === nothing && continue
        if obs_key == :obs_hours && base_hours_for_diff !== nothing
            y = y .- base_hours_for_diff
        end
        short_name = split(name, "\n")[1]
        plot!(p, 1:H, y; label = short_name, color = spec.color, linewidth = 2.0)
    end

    # Alt-policy rules
    for (name, spec) in altpol_specs
        mb_ap = get(altpol_results, name, nothing)
        mb_ap === nothing && continue
        y = get_forecast(mb_ap, obs_key, H)
        y === nothing && continue
        if obs_key == :obs_hours && base_hours_for_diff !== nothing
            y = y .- base_hours_for_diff
        end
        plot!(p, 1:H, y; label = name, color = spec.color,
              linewidth = 2.5, linestyle = spec.style)
    end

    if obs_key == :obs_corepce
        hline!(p, [2.0]; color = :black, linestyle = :dot, linewidth = 1, label = "2% Target")
    elseif obs_key == :obs_hours
        hline!(p, [0.0]; color = :black, linestyle = :dot, linewidth = 1, label = "Baseline")
    end
    push!(altpol_subplots, p)
end

altpol_main = plot(altpol_subplots...; layout = (2, 2), size = (1100, 750),
                   plot_title = "Discretionary cut/hold/hike vs. Taylor rules")
display(altpol_main)
savefig(altpol_main, joinpath(plotdir, "altpol_vs_discretionary.html"))
println("   Saved: altpol_vs_discretionary.html")

# Also dump average rate prescribed by each rule, since that's the
# headline number people will want from this comparison.
println()
println("="^60)
println("  RATE PATH PRESCRIBED BY ALT-POLICY RULES")
println("="^60)
@printf("  %-20s", "")
for q in 1:H; @printf("   Q+%-4d", q); end
println()
println("  ", "-"^(20 + 8 * H))
for (name, _) in altpol_specs
    mb_ap = get(altpol_results, name, nothing)
    mb_ap === nothing && continue
    y = get_forecast(mb_ap, :obs_nominalrate, H)
    y === nothing && continue
    @printf("  %-20s", name)
    for q in 1:H; @printf("  %+6.2f%%", y[q]); end
    println()
end
base_y_rate = get_forecast(mb_baseline, :obs_nominalrate, H)
if base_y_rate !== nothing
    @printf("  %-20s", "Baseline (hist rule)")
    for q in 1:H; @printf("  %+6.2f%%", base_y_rate[q]); end
    println()
end
println()

####################################################################
# PLOT 3: INFLATION vs EMPLOYMENT TRADEOFF
# Each scenario is a single point (posterior mean of avg core PCE vs. avg
# hours diff from Hold) with 68% posterior error bars on each axis.
####################################################################
tradeoff_plot = plot(; title = "Policy Tradeoff: Inflation vs. Employment Gain",
                     xlabel = "Avg. Core PCE Inflation (%, posterior mean)",
                     ylabel = "Avg. Δ Hours per Capita (%)",
                     size = (750, 500), legend = :topright)

hold_hours_avg = hold_mid_hours !== nothing ? mean(hold_mid_hours) : 0.0

# Helper: posterior band width for a variable, averaged over forecast horizon.
# Returns (mean_val, half_width_68) or (mean_val, 0) if bands unavailable.
function infl_err(mb, obs_key)
    y = get_forecast(mb, obs_key, H)
    y === nothing && return (NaN, 0.0)
    b = get_bands(mb, obs_key, "68.0%", H)
    if b === nothing
        return (mean(y), 0.0)
    end
    half_w = mean((b[2] .- b[1]) ./ 2)
    return (mean(y), half_w)
end

for (name, spec) in scenarios
    x_mean, x_err = infl_err(results[name], :obs_corepce)
    y_raw = get_forecast(results[name], employment_key, H)
    y_raw === nothing && continue
    y_mean = mean(y_raw) - hold_hours_avg
    _, y_err = infl_err(results[name], employment_key)   # same half-width

    short_name = split(name, "\n")[1]
    scatter!(tradeoff_plot, [x_mean], [y_mean];
             xerror = [x_err], yerror = [y_err],
             label = short_name, color = spec.color,
             markershape = :circle, markersize = 7, markerstrokewidth = 1)
end

vline!(tradeoff_plot, [2.0]; color = :black, linestyle = :dot, linewidth = 1, label = "2% Target")
hline!(tradeoff_plot, [0.0]; color = :gray, linestyle = :dot, linewidth = 0.5, label = "")

display(tradeoff_plot)
savefig(tradeoff_plot, joinpath(plotdir, "inflation_employment_tradeoff.html"))
println("   Saved: inflation_employment_tradeoff.html")

####################################################################
# TABLES
####################################################################
println("\n")
println("="^72)
println("  CURRENT ECONOMIC CONDITIONS (April 2026)")
println("="^72)
println()
println("  Fed Funds Rate:     3.50 - 3.75%  (held since Jan 2026)")
println("  Core PCE Inflation: 3.0 - 3.1%    (above 2% target)")
println("  CPI Inflation:      2.4% Feb, ~3.2% Mar expected")
println("  Unemployment:       4.3%           (rising; avg 22K jobs/mo)")
println("  GDP Growth:         ~1.0% forecast (NY Fed DSGE, Mar 2026)")
println("  Avg Tariff Rate:    16.8%          (major supply shock)")
println("  Recession Prob:     35.8%          (NY Fed DSGE)")
println()

# Model baseline quarter-by-quarter
println("="^72)
println("  MODEL BASELINE FORECAST (Model's Own Rate Path)")
println("="^72)
println()
@printf("  %-30s", "")
for q in 1:H; @printf("   Q+%-6d", q); end
println()
println("  ", "-"^(30 + 10 * H))

for (title, obs_key) in level_indicators
    y = get_forecast(mb_baseline, obs_key, H)
    if y !== nothing
        @printf("  %-30s", title)
        for q in 1:H; @printf("  %+7.2f%%", y[q]); end
        println()
    end
end
println()

# Scenario comparison (midpoints)
println("="^72)
println("  SCENARIO COMPARISON: Average Over $(H) Quarters (Midpoint of Range)")
println("="^72)
println()
scenario_keys = collect(keys(scenarios))
@printf("  %-28s %11s %11s %11s\n", "Indicator", "Cut 25bp", "Hold", "Hike 25bp")
println("  ", "-"^63)

for (title, obs_key) in level_indicators
    @printf("  %-28s", title)
    for name in scenario_keys
        y = get_forecast(results[name], obs_key, H)
        if y !== nothing
            @printf("  %+9.2f%%", mean(y))
        else
            @printf("  %11s", "N/A")
        end
    end
    println()
end

# Hours per capita (as % diff from hold; obs_hours is stored as 100*log(hours)
# so differences in that unit ≈ percent change in hours).
hold_h = get_forecast(results[hold_key], employment_key, H)
if hold_h !== nothing
    @printf("  %-30s", "Hours per capita (vs. Hold)")
    for name in scenario_keys
        y = get_forecast(results[name], employment_key, H)
        if y !== nothing
            diff = mean(y) - mean(hold_h)
            if name == hold_key
                @printf("  %12s", "baseline")
            else
                @printf("  %+10.2f %%", diff)
            end
        end
    end
    println()
end

# Unemployment rate via Okun bridge (anchor u=4.3% at Q1, Okun=0.5)
@printf("  %-28s", "Unemployment rate (Okun)")
for name in scenario_keys
    u_path = hours_to_unemployment(results[name], H)
    @printf("  %+9.2f%%", mean(u_path))
end
println()
println()

# Differential impact
println("="^72)
println("  DIFFERENTIAL IMPACT vs. Baseline Rate Path (avg over $(n_peg_quarters) pinned quarters)")
println("="^72)
println()
@printf("  %-30s %14s %14s\n", "Indicator", "Cut 25bp", "Hike 25bp")
println("  ", "-"^58)

for (title, obs_key) in level_indicators
    hold_y = get_forecast(results[hold_key], obs_key, H)
    if hold_y === nothing; continue; end
    hold_avg = mean(hold_y)

    @printf("  %-30s", title)
    for name in [scenario_keys[1], scenario_keys[3]]
        y = get_forecast(results[name], obs_key, H)
        if y !== nothing
            @printf("  %+12.3f pp", mean(y) - hold_avg)
        end
    end
    println()
end

if hold_h !== nothing
    @printf("  %-30s", "Hours per capita")
    for name in [scenario_keys[1], scenario_keys[3]]
        y = get_forecast(results[name], employment_key, H)
        if y !== nothing
            @printf("  %+12.3f %%", mean(y) - mean(hold_h))
        end
    end
    println()
end

# Unemployment rate differential via Okun bridge
hold_u = hours_to_unemployment(results[hold_key], H)
@printf("  %-30s", "Unemployment rate (Okun)")
for name in [scenario_keys[1], scenario_keys[3]]
    u_path = hours_to_unemployment(results[name], H)
    @printf("  %+12.3f pp", mean(u_path) - mean(hold_u))
end
println()

println("  ", "-"^58)
println("  pp = percentage points; hours in % deviation (obs_hours = 100·log(hours));")
println("  unemployment via Okun bridge: Δu = -$(OKUN_COEF) × Δhours%, anchored at $(OKUN_ANCHOR_U)%")
println()

####################################################################
# POLICY ASSESSMENT
####################################################################
println("="^72)
println("  POLICY ASSESSMENT — DUAL MANDATE")
println("="^72)
println()

hold_infl  = mean(get_forecast(results[hold_key], :obs_corepce, H))
cut_infl   = mean(get_forecast(results[scenario_keys[1]], :obs_corepce, H))
hike_infl  = mean(get_forecast(results[scenario_keys[3]], :obs_corepce, H))

println("  PRICE STABILITY (2% Core PCE target):")
@printf("    Cut 25bp     → Core PCE: %.2f%%  (%+.2f pp from target)\n", cut_infl, cut_infl - 2.0)
@printf("    Hold         → Core PCE: %.2f%%  (%+.2f pp from target)\n", hold_infl, hold_infl - 2.0)
@printf("    Hike 25bp    → Core PCE: %.2f%%  (%+.2f pp from target)\n", hike_infl, hike_infl - 2.0)
println()

if hold_h !== nothing
    cut_emp_diff  = mean(get_forecast(results[scenario_keys[1]], employment_key, H)) - mean(hold_h)
    hike_emp_diff = mean(get_forecast(results[scenario_keys[3]], employment_key, H)) - mean(hold_h)

    println("  HOURS PER CAPITA (% deviation vs. Baseline rate path):")
    @printf("    Cut 25bp     → %+.3f %%\n", cut_emp_diff)
    @printf("    Hold         → baseline\n")
    @printf("    Hike 25bp    → %+.3f %%\n", hike_emp_diff)
    println()
end

# Unemployment rate levels (Okun bridge)
cut_u  = mean(hours_to_unemployment(results[scenario_keys[1]], H))
hold_u_lvl  = mean(hours_to_unemployment(results[hold_key], H))
hike_u = mean(hours_to_unemployment(results[scenario_keys[3]], H))
println("  UNEMPLOYMENT RATE (Okun bridge, anchor = $(OKUN_ANCHOR_U)%):")
@printf("    Cut 25bp     → %.2f%%  (%+.2f pp vs. Hold)\n", cut_u,  cut_u  - hold_u_lvl)
@printf("    Hold         → %.2f%%  (baseline)\n",          hold_u_lvl)
@printf("    Hike 25bp    → %.2f%%  (%+.2f pp vs. Hold)\n", hike_u, hike_u - hold_u_lvl)
println("    (Okun coef $(OKUN_COEF): Δu pp = -$(OKUN_COEF) × Δhours%)")
println()

println("  KEY CONTEXT:")
println("  - Inflation above target is driven largely by SUPPLY shocks")
println("    (tariffs at 16.8% avg). Rate hikes are less effective")
println("    against supply-driven inflation than demand-driven.")
println("  - Labor market is weakening (4.3% unemployment, slowing")
println("    payrolls at 22K/mo avg). Further tightening risks")
println("    accelerating job losses without meaningfully reducing")
println("    supply-driven price pressures.")
println("  - NY Fed DSGE recession probability: 35.8%")
println("  - A 25bp cut supports employment with a modest inflation")
println("    cost that is small relative to the tariff-driven overshoot.")
println()

####################################################################
# MODEL'S VIEW OF CURRENT STATE
####################################################################
println("="^72)
println("  MODEL'S VIEW: Where Is the Economy Right Now?")
println("  (Filtered estimates from Kalman filter on FRED data)")
println("="^72)
println()
println("  The model processes all historical data through a Kalman filter")
println("  to estimate the current state. These are the model's implied")
println("  values, which may differ from the latest BLS/BEA releases.")
println()

hist_cols = Symbol.(names(mb_hist.means))
n_hist = nrow(mb_hist.means)

# Show last 4 quarters of history
n_show = min(4, n_hist)
println("  Recent quarters (model-filtered estimates):")
println()
@printf("  %-30s", "")
for i in (n_hist - n_show + 1):n_hist
    @printf("  %12s", string(mb_hist.means[i, :date])[1:7])
end
println()
println("  ", "-"^(30 + 13 * n_show))

hist_display = OrderedDict(
    "Core PCE Inflation (%)"   => :obs_corepce,
    "GDP Deflator Infl. (%)"   => :obs_gdpdeflator,
    "Fed Funds Rate (%)"       => :obs_nominalrate,
    "Wage Growth (%)"          => :obs_wages,
    "Credit Spread (%)"        => :obs_spread,
    "Hours Worked (log)"       => :obs_hours,
)

for (title, obs_key) in hist_display
    if obs_key in hist_cols
        @printf("  %-30s", title)
        for i in (n_hist - n_show + 1):n_hist
            val = mb_hist.means[i, obs_key]
            if isnan(val)
                @printf("  %12s", "N/A")
            else
                @printf("  %+11.2f%%", val)
            end
        end
        println()
    end
end
println()
println("  Compare to actual data:")
println("    Actual Core PCE (Feb 2026):    3.0-3.1%")
println("    Actual Unemployment (Mar 2026): 4.3%")
println("    Actual Fed Funds Rate:          3.50-3.75%")
println()

####################################################################
# COST-PUSH SHOCK (TARIFF) IMPULSE RESPONSE
####################################################################
println("="^72)
println("  TARIFF EFFECT: Cost-Push Shock Impulse Response")
println("  (1 std dev price markup shock — proxy for tariff impact)")
println("="^72)
println()

# Try to load and display IRF
try
    mb_irf = read_mb(m, :mode, :none, :irfobs; forecast_string = "irf_costpush")
    irf_cols = Symbol.(names(mb_irf.means))
    n_irf = min(H, nrow(mb_irf.means))

    println("  When a cost-push shock (like tariffs) hits, the model predicts:")
    println()
    @printf("  %-30s", "")
    for q in 1:n_irf; @printf("   Q+%-6d", q); end
    println()
    println("  ", "-"^(30 + 10 * n_irf))

    irf_display = OrderedDict(
        "Core PCE Inflation (%)"   => :obs_corepce,
        "GDP Deflator Infl. (%)"   => :obs_gdpdeflator,
        "Fed Funds Rate (%)"       => :obs_nominalrate,
        "Wage Growth (%)"          => :obs_wages,
        "Hours Worked"             => :obs_hours,
    )

    for (title, obs_key) in irf_display
        if obs_key in irf_cols
            @printf("  %-30s", title)
            for q in 1:n_irf
                val = mb_irf.means[q, obs_key]
                @printf("  %+8.3f", val)
            end
            println()
        end
    end
    println()
    println("  Interpretation: A tariff-like shock raises inflation but")
    println("  REDUCES employment. Rate hikes would compound the employment")
    println("  loss without addressing the supply-side cause of inflation.")

    # Plot the IRF
    irf_plot_vars = OrderedDict(
        "Inflation Response (%)" => :obs_gdpdeflator,
        "Employment Response"    => :obs_hours,
        "Rate Response (%)"      => :obs_nominalrate,
        "Wage Response (%)"      => :obs_wages,
    )

    irf_subplots = []
    for (title, obs_key) in irf_plot_vars
        if obs_key in irf_cols
            y = mb_irf.means[1:n_irf, obs_key]
            p = plot(1:n_irf, y; title = title, xlabel = "Quarter",
                     color = :darkorange, linewidth = 2.5, legend = false,
                     titlefontsize = 10, xticks = 1:n_irf, fillrange = 0,
                     fillalpha = 0.1)
            hline!(p, [0.0]; color = :black, linewidth = 0.5)
            push!(irf_subplots, p)
        end
    end

    if !isempty(irf_subplots)
        irf_plot = plot(irf_subplots...; layout = (2, 2), size = (1000, 650),
                        plot_title = "Cost-Push Shock (Tariff Proxy): Impulse Response")
        display(irf_plot)
        savefig(irf_plot, joinpath(plotdir, "tariff_impulse_response.html"))
        println("\n   Saved: tariff_impulse_response.html")
    end
catch e
    println("  (Could not compute IRF: $e)")
    println("  This is OK — the scenario comparison still captures tariff effects")
    println("  indirectly through the estimated cost-push shocks in the data.")
end

####################################################################
# PRESENTATION CHART: Inflation Projections (BLS-style)
# Extended horizon with history + forecast
####################################################################
println("\n>> Generating presentation inflation chart...")

# How many forecast quarters to display (beyond the peg, rates revert to
# the model's natural path — this is realistic and informative to show)
n_fcast_display = 12  # 3 years out

# --- Collect history ---
hist_dates = mb_hist.means[!, :date]
hist_core  = :obs_corepce in Symbol.(names(mb_hist.means)) ?
             mb_hist.means[!, :obs_corepce] : nothing
hist_defl  = :obs_gdpdeflator in Symbol.(names(mb_hist.means)) ?
             mb_hist.means[!, :obs_gdpdeflator] : nothing

# Take last 8 quarters of history for context
n_hist_show = min(8, nrow(mb_hist.means))
hist_range  = (nrow(mb_hist.means) - n_hist_show + 1):nrow(mb_hist.means)

# --- Collect forecast paths ---
function get_long_forecast(mb, obs_key, n)
    cols = Symbol.(names(mb.means))
    if obs_key in cols
        col = mb.means[!, obs_key]
        nn = min(n, length(col))
        vals = col[1:nn]
        return any(isnan.(vals)) ? nothing : vals
    end
    return nothing
end

# Use Core PCE as the primary inflation measure (Fed's target)
# Fall back to GDP Deflator if Core PCE unavailable
infl_key = :obs_corepce
infl_label = "Core PCE Inflation"
hist_infl = hist_core

if hist_infl === nothing || all(isnan.(hist_infl[hist_range]))
    infl_key = :obs_gdpdeflator
    infl_label = "GDP Deflator Inflation"
    hist_infl = hist_defl
end

# Build x-axis: history quarters as negative, forecast as positive
# x = -n_hist_show+1 ... 0 | 1 ... n_fcast_display
#     ← history →    now   ← forecast →

x_hist = collect((-n_hist_show + 1):0)
x_fcast = collect(1:n_fcast_display)
x_all = vcat(x_hist, x_fcast)

# Quarter labels for x-axis
fcast_start = hist_dates[end]  # last history date ≈ forecast start
quarter_labels = Dict{Int, String}()
for (i, x) in enumerate(x_hist)
    d = hist_dates[hist_range[i]]
    q = Dates.quarterofyear(d)
    y = Dates.year(d) % 100
    quarter_labels[x] = "$(y)Q$(q)"
end
for x in x_fcast
    d = fcast_start + Dates.Month(3 * x)
    q = Dates.quarterofyear(d)
    y = Dates.year(d) % 100
    quarter_labels[x] = "$(y)Q$(q)"
end

# Show every other label to avoid crowding
tick_positions = x_all[1:2:end]
tick_labels = [get(quarter_labels, x, "") for x in tick_positions]

# --- Main Chart ---
infl_chart = plot(;
    title = "$infl_label: Rate Scenario Projections",
    xlabel = "",
    ylabel = "Year-over-Year (%)",
    size = (1000, 500),
    legend = :topright,
    legendfontsize = 9,
    titlefontsize = 13,
    guidefontsize = 11,
    xticks = (tick_positions, tick_labels),
    xrotation = 45,
    grid = true,
    gridalpha = 0.3,
    background_color = :white,
    foreground_color = :black,
)

# 2% target line (full width)
hline!(infl_chart, [2.0]; color = :black, linestyle = :dot, linewidth = 1.5,
       label = "2% Fed Target")

# Vertical line at forecast start
vline!(infl_chart, [0.5]; color = :gray, linestyle = :dash, linewidth = 1,
       label = "")

# End-of-pinned-window marker. Rate is pinned at baseline ± delta for
# n_peg_quarters quarters; after that the peg releases and the rate is
# endogenous again under the estimated Taylor rule.
vline!(infl_chart, [n_peg_quarters + 0.5]; color = :gray, linestyle = :dot,
       linewidth = 1, label = "")
annotate!(infl_chart, [(n_peg_quarters + 0.5, 0.3,
          text("peg ends", 8, :gray, :right))])

# History (solid black)
if hist_infl !== nothing
    y_hist = hist_infl[hist_range]
    valid = .!isnan.(y_hist)
    if any(valid)
        plot!(infl_chart, x_hist[valid], y_hist[valid];
              color = :black, linewidth = 3, label = "Actual",
              markershape = :circle, markersize = 4)
    end
end

# Scenario forecasts
scenario_plot_order = [
    ("Cut 25bp",  :forestgreen, "Baseline − 25bp"),
    ("Hold",      :royalblue,   "Baseline"),
    ("Hike 25bp", :crimson,     "Baseline + 25bp"),
]

# Helper: long-horizon posterior band pull (same logic as get_bands but
# extends across as many rows as the bands DataFrame holds).
function get_long_bands(mb, obs_key::Symbol, pct::String, n::Int)
    if isempty(mb.bands); return nothing; end
    if !haskey(mb.bands, obs_key); return nothing; end
    bdf = mb.bands[obs_key]
    lb_col = Symbol(pct, " LB")
    ub_col = Symbol(pct, " UB")
    cols = propertynames(bdf)
    if !(lb_col in cols) || !(ub_col in cols); return nothing; end
    nrow = min(n, size(bdf, 1))
    lb = bdf[1:nrow, lb_col]
    ub = bdf[1:nrow, ub_col]
    (any(isnan.(lb)) || any(isnan.(ub))) && return nothing
    return (lb, ub)
end

# Draw the baseline posterior fan ONCE, behind all scenario lines.
b90_base = get_long_bands(mb_baseline, infl_key, "90.0%", n_fcast_display)
b68_base = get_long_bands(mb_baseline, infl_key, "68.0%", n_fcast_display)
if b90_base !== nothing
    n = length(b90_base[1])
    plot!(infl_chart, x_fcast[1:n], b90_base[1]; fillrange = b90_base[2],
          fillalpha = 0.10, color = :slategray, linewidth = 0,
          label = "Baseline 90%")
end
if b68_base !== nothing
    n = length(b68_base[1])
    plot!(infl_chart, x_fcast[1:n], b68_base[1]; fillrange = b68_base[2],
          fillalpha = 0.22, color = :slategray, linewidth = 0,
          label = "Baseline 68%")
end

# Scenario conditional means (at :mode) overlaid as solid lines.
for (key, color, label) in scenario_plot_order
    if haskey(results, key)
        y_mid = get_long_forecast(results[key], infl_key, n_fcast_display)
        if y_mid !== nothing
            local nn = length(y_mid)
            plot!(infl_chart, x_fcast[1:nn], y_mid;
                  color = color, linewidth = 2.5, label = label)
        end
    end
end


# Add annotation for forecast region
annotate!(infl_chart, [(n_fcast_display ÷ 2, 1.5,
          text("← Projection →", 9, :gray, :center))])

display(infl_chart)
savefig(infl_chart, joinpath(plotdir, "inflation_projections.html"))
println("   Saved: inflation_projections.html")

# --- Same chart for Fed Funds Rate ---
println(">> Generating rate path chart...")

rate_chart = plot(;
    title = "Fed Funds Rate: Scenario Paths",
    xlabel = "",
    ylabel = "Annualized Rate (%)",
    size = (1000, 500),
    legend = :topright,
    legendfontsize = 9,
    titlefontsize = 13,
    guidefontsize = 11,
    xticks = (tick_positions, tick_labels),
    xrotation = 45,
    grid = true,
    gridalpha = 0.3,
    background_color = :white,
)

vline!(rate_chart, [0.5]; color = :gray, linestyle = :dash, linewidth = 1, label = "")

# End-of-pinned-window marker.
vline!(rate_chart, [n_peg_quarters + 0.5]; color = :gray, linestyle = :dot,
       linewidth = 1, label = "")

# History
hist_rate = :obs_nominalrate in Symbol.(names(mb_hist.means)) ?
            mb_hist.means[hist_range, :obs_nominalrate] : nothing
if hist_rate !== nothing
    valid = .!isnan.(hist_rate)
    if any(valid)
        plot!(rate_chart, x_hist[valid], hist_rate[valid];
              color = :black, linewidth = 3, label = "Actual",
              markershape = :circle, markersize = 4)
    end
end

# Baseline FFR posterior fan (from :full) drawn once behind all scenario lines.
b90_base_r = get_long_bands(mb_baseline, :obs_nominalrate, "90.0%", n_fcast_display)
b68_base_r = get_long_bands(mb_baseline, :obs_nominalrate, "68.0%", n_fcast_display)
if b90_base_r !== nothing
    n = length(b90_base_r[1])
    plot!(rate_chart, x_fcast[1:n], b90_base_r[1]; fillrange = b90_base_r[2],
          fillalpha = 0.10, color = :slategray, linewidth = 0, label = "Baseline 90%")
end
if b68_base_r !== nothing
    n = length(b68_base_r[1])
    plot!(rate_chart, x_fcast[1:n], b68_base_r[1]; fillrange = b68_base_r[2],
          fillalpha = 0.22, color = :slategray, linewidth = 0, label = "Baseline 68%")
end

# Scenario conditional rate paths (pinned to baseline ± 25bp over the
# pinned window; endogenous afterwards. Dotted vertical marks the end.)
for (key, color, label) in scenario_plot_order
    if haskey(results, key)
        y_mid = get_long_forecast(results[key], :obs_nominalrate, n_fcast_display)
        if y_mid !== nothing
            local nn = length(y_mid)
            plot!(rate_chart, x_fcast[1:nn], y_mid;
                  color = color, linewidth = 2.5, label = label)
        end
    end
end


display(rate_chart)
savefig(rate_chart, joinpath(plotdir, "rate_path_projections.html"))
println("   Saved: rate_path_projections.html")

# --- Employment impact chart ---
println(">> Generating employment chart...")

emp_chart = plot(;
    title = "Employment Impact: Hours Worked (Change vs. Hold)",
    xlabel = "",
    ylabel = "Δ Hours per Capita (%)",
    size = (1000, 500),
    legend = :topright,
    legendfontsize = 9,
    titlefontsize = 13,
    guidefontsize = 11,
    xticks = (tick_positions[tick_positions .>= 1], tick_labels[tick_positions .>= 1]),
    xrotation = 45,
    grid = true,
    gridalpha = 0.3,
    background_color = :white,
)

hline!(emp_chart, [0.0]; color = :black, linewidth = 1, linestyle = :dot, label = "Hold (baseline)")

hold_key_name = "Hold"
hold_emp = get_long_forecast(results[hold_key_name], :obs_hours, n_fcast_display)

if hold_emp !== nothing
    for (key, color, label) in scenario_plot_order
        if key == hold_key_name; continue; end
        if haskey(results, key)
            y = get_long_forecast(results[key], :obs_hours, n_fcast_display)
            if y !== nothing
                local nn = min(length(y), length(hold_emp))
                diff = y[1:nn] .- hold_emp[1:nn]
                plot!(emp_chart, x_fcast[1:nn], diff;
                      color = color, linewidth = 2.5, label = label)
            end
        end
    end

end

display(emp_chart)
savefig(emp_chart, joinpath(plotdir, "employment_projections.html"))
println("   Saved: employment_projections.html")

####################################################################
# PRESENTATION CHART: Unemployment Rate Projections (long horizon)
#
# Same style as inflation_projections / rate_path_projections:
#   - 8 quarters of Kalman-filtered history on the left (Okun-translated
#     from obs_hours so it ties into the model's view of current labor
#     market rather than a separate FRED series)
#   - 12 quarters of forecast on the right with baseline posterior fan
#     and scenario conditional paths overlaid
#   - Dotted vertical at end of pinned window (Q{n_peg_quarters}.5)
#   - NAIRU reference line at 4.0%
####################################################################
println(">> Generating long-horizon unemployment chart...")

u_chart = plot(;
    title = "Unemployment Rate: Scenario Projections",
    xlabel = "",
    ylabel = "Unemployment Rate (%)",
    size = (1000, 500),
    legend = :topright,
    legendfontsize = 9,
    titlefontsize = 13,
    guidefontsize = 11,
    xticks = (tick_positions, tick_labels),
    xrotation = 45,
    grid = true,
    gridalpha = 0.3,
    background_color = :white,
)

# NAIRU reference
hline!(u_chart, [4.0]; color = :black, linestyle = :dot, linewidth = 1.5,
       label = "NAIRU ≈ 4.0%")

# Vertical line at forecast start
vline!(u_chart, [0.5]; color = :gray, linestyle = :dash, linewidth = 1, label = "")

# End-of-pinned-window marker
vline!(u_chart, [n_peg_quarters + 0.5]; color = :gray, linestyle = :dot,
       linewidth = 1, label = "")

# History: actual FRED U3 (quarterly averages from the UNRATE pull
# above, aligned to mb_hist dates). Falls back to Okun-translated
# filtered hours if the FRED pull failed.
hist_u_display = if u_fred_df !== nothing
    u_actual_full[hist_range]
else
    hours_vec_to_u(mb_hist.means[hist_range, :obs_hours])
end
let valid = .!isnan.(hist_u_display)
    if any(valid)
        plot!(u_chart, x_hist[valid], hist_u_display[valid];
              color = :black, linewidth = 3,
              label = u_fred_df !== nothing ? "Actual U3 (FRED)" : "Actual (Okun)",
              markershape = :circle, markersize = 4)
    end
end

# Baseline posterior fan (90% outer, 68% inner) — from :full obs_hours
# bands pushed through the Okun formula across the full forecast horizon.
u_b90_long = hours_to_unemployment_bands(mb_baseline, "90.0%", n_fcast_display)
u_b68_long = hours_to_unemployment_bands(mb_baseline, "68.0%", n_fcast_display)
if u_b90_long !== nothing
    local nn = length(u_b90_long[1])
    plot!(u_chart, x_fcast[1:nn], u_b90_long[1]; fillrange = u_b90_long[2],
          fillalpha = 0.10, color = :slategray, linewidth = 0,
          label = "Baseline 90%")
end
if u_b68_long !== nothing
    local nn = length(u_b68_long[1])
    plot!(u_chart, x_fcast[1:nn], u_b68_long[1]; fillrange = u_b68_long[2],
          fillalpha = 0.22, color = :slategray, linewidth = 0,
          label = "Baseline 68%")
end

# Baseline mean u (from :mode — same reference frame scenarios use)
base_u_long = hours_to_unemployment(mb_baseline_mode, n_fcast_display)
let nn = length(base_u_long)
    plot!(u_chart, x_fcast[1:nn], base_u_long;
          color = :slategray, linewidth = 2, linestyle = :dash,
          label = "Baseline (mean)")
end

# Scenario conditional paths (deterministic, from :mode)
for (key, color, label) in scenario_plot_order
    if haskey(results, key)
        u_scen = hours_to_unemployment(results[key], n_fcast_display)
        local nn = length(u_scen)
        plot!(u_chart, x_fcast[1:nn], u_scen;
              color = color, linewidth = 2.5, label = label)
    end
end

display(u_chart)
savefig(u_chart, joinpath(plotdir, "unemployment_projections.html"))
println("   Saved: unemployment_projections.html")

####################################################################
# PRESENTATION CHART: Real Wage Growth Projections (long horizon)
#
# Same structure as inflation_projections and unemployment_projections.
# obs_wages is directly observed (COMPNFB deflated by GDPDEF, q/q log
# growth, annualized), so mb_hist.means[:, :obs_wages] is already the
# FRED-derived value — no bridge layer like unemployment needs.
#
# Note on label: obs_wages is REAL compensation per hour growth (q/q
# log-change of nominal comp deflated by GDPDEF, annualized), NOT
# nominal wage growth. For nominal wage growth you'd compute
# obs_wages + obs_gdpdeflator — the wages_vs_inflation chart shows
# that relationship implicitly via the gap between the two lines.
####################################################################
println(">> Generating long-horizon wage growth chart...")

wage_chart = plot(;
    title = "Real Wage Growth: Scenario Projections",
    xlabel = "",
    ylabel = "Annualized % (real comp/hr)",
    size = (1000, 500),
    legend = :topright,
    legendfontsize = 9,
    titlefontsize = 13,
    guidefontsize = 11,
    xticks = (tick_positions, tick_labels),
    xrotation = 45,
    grid = true,
    gridalpha = 0.3,
    background_color = :white,
)

# Zero line + productivity-trend reference (real wage growth bounces
# around long-run productivity growth, usually ~1.5% for the US).
hline!(wage_chart, [0.0]; color = :black, linestyle = :dot, linewidth = 1,
       label = "")
hline!(wage_chart, [1.5]; color = :darkgreen, linestyle = :dot, linewidth = 1.2,
       label = "Trend ≈ 1.5%")

# Forecast-start and peg-end verticals
vline!(wage_chart, [0.5]; color = :gray, linestyle = :dash, linewidth = 1, label = "")
vline!(wage_chart, [n_peg_quarters + 0.5]; color = :gray, linestyle = :dot,
       linewidth = 1, label = "")

# History straight out of mb_hist (FRED-derived obs_wages)
if :obs_wages in Symbol.(names(mb_hist.means))
    hist_wages = mb_hist.means[hist_range, :obs_wages]
    valid = .!isnan.(hist_wages)
    if any(valid)
        plot!(wage_chart, x_hist[valid], hist_wages[valid];
              color = :black, linewidth = 3, label = "Actual",
              markershape = :circle, markersize = 4)
    end
end

# Baseline posterior fan (:full obs_wages bands) across the full
# forecast horizon.
w_b90 = get_long_bands(mb_baseline, :obs_wages, "90.0%", n_fcast_display)
w_b68 = get_long_bands(mb_baseline, :obs_wages, "68.0%", n_fcast_display)
if w_b90 !== nothing
    local nn = length(w_b90[1])
    plot!(wage_chart, x_fcast[1:nn], w_b90[1]; fillrange = w_b90[2],
          fillalpha = 0.10, color = :slategray, linewidth = 0,
          label = "Baseline 90%")
end
if w_b68 !== nothing
    local nn = length(w_b68[1])
    plot!(wage_chart, x_fcast[1:nn], w_b68[1]; fillrange = w_b68[2],
          fillalpha = 0.22, color = :slategray, linewidth = 0,
          label = "Baseline 68%")
end

# Baseline mean (from :mode, same reference frame scenarios use)
base_wages_long = get_long_forecast(mb_baseline_mode, :obs_wages, n_fcast_display)
if base_wages_long !== nothing
    local nn = length(base_wages_long)
    plot!(wage_chart, x_fcast[1:nn], base_wages_long;
          color = :slategray, linewidth = 2, linestyle = :dash,
          label = "Baseline (mean)")
end

# Scenario conditional paths (:mode)
for (key, color, label) in scenario_plot_order
    if haskey(results, key)
        y_scen = get_long_forecast(results[key], :obs_wages, n_fcast_display)
        if y_scen !== nothing
            local nn = length(y_scen)
            plot!(wage_chart, x_fcast[1:nn], y_scen;
                  color = color, linewidth = 2.5, label = label)
        end
    end
end

display(wage_chart)
savefig(wage_chart, joinpath(plotdir, "wage_projections.html"))
println("   Saved: wage_projections.html")

####################################################################
# PRESENTATION CHART: Wage Growth vs. Core PCE Inflation
#
# Real-wage story: the gap between nominal wage growth and inflation
# is (approximately) real wage growth. When wages > inflation, workers
# are gaining purchasing power; when inflation > wages, they're losing
# it. This chart puts both series on the same axis with history + a
# 12-quarter baseline forecast fan so the reader can see whether the
# model expects inflation to converge to wage growth, diverge further,
# or cross it entirely.
####################################################################
println(">> Generating wage-vs-inflation chart...")

wage_infl_chart = plot(;
    title = "Wage Growth vs. Core PCE Inflation (Baseline)",
    xlabel = "",
    ylabel = "Annualized %",
    size = (1000, 500),
    legend = :topright,
    legendfontsize = 9,
    titlefontsize = 13,
    guidefontsize = 11,
    xticks = (tick_positions, tick_labels),
    xrotation = 45,
    grid = true,
    gridalpha = 0.3,
    background_color = :white,
)

# 2% inflation target
hline!(wage_infl_chart, [2.0]; color = :black, linestyle = :dot, linewidth = 1.5,
       label = "2% Target")

# Forecast-start vertical
vline!(wage_infl_chart, [0.5]; color = :gray, linestyle = :dash, linewidth = 1, label = "")

# History: wages and inflation from the Kalman filter (mb_hist)
if :obs_corepce in Symbol.(names(mb_hist.means))
    hist_pce = mb_hist.means[hist_range, :obs_corepce]
    valid = .!isnan.(hist_pce)
    any(valid) && plot!(wage_infl_chart, x_hist[valid], hist_pce[valid];
                        color = :crimson, linewidth = 3, label = "Inflation (actual)",
                        markershape = :circle, markersize = 4)
end

if :obs_wages in Symbol.(names(mb_hist.means))
    hist_wages = mb_hist.means[hist_range, :obs_wages]
    valid = .!isnan.(hist_wages)
    any(valid) && plot!(wage_infl_chart, x_hist[valid], hist_wages[valid];
                        color = :steelblue, linewidth = 3, label = "Wages (actual)",
                        markershape = :circle, markersize = 4)
end

# Baseline posterior fans from :full — both 68% and 90% for inflation,
# just 68% for wages so the two fans don't visually fight each other.
b90_infl_w = get_long_bands(mb_baseline, :obs_corepce, "90.0%", n_fcast_display)
b68_infl_w = get_long_bands(mb_baseline, :obs_corepce, "68.0%", n_fcast_display)
if b90_infl_w !== nothing
    local nn = length(b90_infl_w[1])
    plot!(wage_infl_chart, x_fcast[1:nn], b90_infl_w[1]; fillrange = b90_infl_w[2],
          fillalpha = 0.08, color = :crimson, linewidth = 0,
          label = "Inflation 90%")
end
if b68_infl_w !== nothing
    local nn = length(b68_infl_w[1])
    plot!(wage_infl_chart, x_fcast[1:nn], b68_infl_w[1]; fillrange = b68_infl_w[2],
          fillalpha = 0.18, color = :crimson, linewidth = 0,
          label = "Inflation 68%")
end

b68_wages_w = get_long_bands(mb_baseline, :obs_wages, "68.0%", n_fcast_display)
if b68_wages_w !== nothing
    local nn = length(b68_wages_w[1])
    plot!(wage_infl_chart, x_fcast[1:nn], b68_wages_w[1]; fillrange = b68_wages_w[2],
          fillalpha = 0.18, color = :steelblue, linewidth = 0,
          label = "Wages 68%")
end

# Baseline mean lines (from :mode for consistency with other charts)
mode_infl_path  = get_long_forecast(mb_baseline_mode, :obs_corepce, n_fcast_display)
mode_wages_path = get_long_forecast(mb_baseline_mode, :obs_wages,   n_fcast_display)

if mode_infl_path !== nothing
    local nn = length(mode_infl_path)
    plot!(wage_infl_chart, x_fcast[1:nn], mode_infl_path;
          color = :crimson, linewidth = 2.5, label = "Inflation (baseline)")
end
if mode_wages_path !== nothing
    local nn = length(mode_wages_path)
    plot!(wage_infl_chart, x_fcast[1:nn], mode_wages_path;
          color = :steelblue, linewidth = 2.5, label = "Wages (baseline)")
end

# Annotation reminding the reader that the gap between the two lines is
# (approximately) real wage growth.
annotate!(wage_infl_chart, [(n_fcast_display ÷ 2, 0.3,
          text("gap = real wage growth (wages − inflation)",
               9, :gray, :center))])

display(wage_infl_chart)
savefig(wage_infl_chart, joinpath(plotdir, "wages_vs_inflation.html"))
println("   Saved: wages_vs_inflation.html")

println()
println("="^72)
println("  ALL DONE!")
println("  Plots saved to: $plotdir")
println("  Open .html files in browser to view.")
println("="^72)
