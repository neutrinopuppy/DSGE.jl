"""
```
transform_data(m::AbstractDSGEModel, levels::DataFrame; cond_type::Symbol = :none,
    verbose::Symbol = :low)
```

Transform data loaded in levels and order columns appropriately for the DSGE
model. Returns DataFrame of transformed data.

The DataFrame `levels` is output from `load_data_levels`. The series in levels are
transformed as specified in `m.observable_mappings`.

- To prepare for per-capita transformations, population data are filtered using
  `hpfilter`. The series in `levels` to use as the population series is given by
  the `population_mnemonic` setting. If `use_population_forecast(m)`, a
  population forecast is appended to the recorded population levels before the
  filtering. Both filtered and unfiltered population levels and growth rates are
  added to the `levels` data frame.
- The transformations are applied for each series using the `levels` DataFrame
  as input.

Conditional data (identified by `cond_type in [:semi, :full]`) are handled
slightly differently: If `use_population_forecast(m)`, we drop the first period
of the population forecast because we treat the first forecast period
`date_forecast_start(m)` as if it were data. We also only apply transformations
for the observables given in `cond_full_names(m)` or `cond_semi_names(m)`.
"""
function transform_data(m::AbstractDSGEModel, levels::DataFrame;
                        cond_type::Symbol = :none, verbose::Symbol = :low)

    n_obs, _ = size(levels)

    # Step 1: HP filter (including population forecasts, if they're being used)
    population_mnemonic = parse_population_mnemonic(m)[1]
    if !isnull(population_mnemonic)
        population_forecast_levels = if use_population_forecast(m)
            read_population_forecast(m; verbose = verbose)
        else
            DataFrame()
        end

        population_data, _ = transform_population_data(levels, population_forecast_levels,
                                                       get(population_mnemonic);
                                                       verbose = verbose,
                                                       use_hpfilter = hpfilter_population(m))

        levels = if isdefined(DataFrames, :leftjoin) # left joins using `join` is deprecated in DataFrames v0.21 (and higher)
            leftjoin(levels, population_data, on = :date)
        else
            join(levels, population_data, on = :date, kind = :left)
        end
        name_maps = [s => t for (s,t) = zip([:filtered_population_recorded, :dlfiltered_population_recorded, :dlpopulation_recorded],
                [:filtered_population, :filtered_population_growth, :unfiltered_population_growth])]
        rename!(levels, name_maps)
    end

    # Step 2: apply transformations to each series
    transformed = DataFrame()
    transformed[!, :date] = levels[:, :date]

    data_transforms = collect_data_transforms(m)

    for series in keys(data_transforms)
        println(verbose, :high, "Transforming series $series...")
        f = data_transforms[series]
        transformed[!, series] = f(levels)
    end

    sort!(transformed, [:date])

    return transformed
end

function collect_data_transforms(m; direction=:fwd)

    data_transforms = OrderedDict{Symbol,Function}()

    # Parse vector of observable mappings into data_transforms dictionary
    for obs in keys(m.observable_mappings)
        data_transforms[obs] = getfield(m.observable_mappings[obs], Symbol(string(direction) * "_transform"))
    end

    data_transforms
end

"""
```
transform_population_data(population_data, population_forecast,
    population_mnemonic; verbose = :low)
```

Load, HP-filter, and compute growth rates from population data in
levels. Optionally do the same for forecasts.

### Inputs

- `population_data`: pre-loaded DataFrame of historical population data
  containing the columns `:date` and `population_mnemonic`. Assumes this is
  sorted by date.
- `population_forecast`: pre-loaded `DataFrame` of population forecast
  containing the columns `:date` and `population_mnemonic`
- `population_mnemonic`: column name for population series in `population_data`
  and `population_forecast`

### Keyword Arguments

- `verbose`: one of `:none`, `:low`, or `:high`
- `use_hpfilter`: whether to HP filter population data and forecast. See `Output` below.
- `pad_forecast_start::Bool`: Whether you want to re-size
the population_forecast such that the first index is one quarter ahead of the last index
of population_data. Only set to false if you have manually constructed population_forecast
to artificially start a quarter earlier, so as to avoid having an unnecessary missing first entry.

### Output

Two dictionaries containing the following keys:

- `population_data_out`:
  + `:filtered_population_recorded`: HP-filtered historical population series (levels)
  + `:dlfiltered_population_recorded`: HP-filtered historical population series (growth rates)
  + `:dlpopulation_recorded`: Non-filtered historical population series (growth rates)

- `population_forecast_out`:
  + `:filtered_population_forecast`: HP-filtered population forecast series (levels)
  + `:dlfiltered_population_forecast`: HP-filtered population forecast series (growth rates)
  + `:dlpopulation_forecast`: Non-filtered historical population series (growth rates)

If `population_forecast_file` is not provided, the r\"*forecast\" fields will be
empty. If `use_hpfilter = false`, then the r\"*filtered*\" fields will be
empty.
"""
function transform_population_data(population_data::DataFrame, population_forecast::DataFrame,
                                   population_mnemonic::Symbol; verbose = :low,
                                   use_hpfilter::Bool = true,
                                   pad_forecast_start::Bool = false)

    # Unfiltered population data
    population_recorded = population_data[:, [:date, population_mnemonic]]

    # Make sure first period of unfiltered population forecast is the first forecast quarter
    if !isempty(population_forecast) && !pad_forecast_start
        last_recorded_date = population_recorded[end, :date]
        if population_forecast[1, :date] <= last_recorded_date
            last_recorded_ind   = findall(population_forecast[!,:date] .== last_recorded_date)[1]
            population_forecast = population_forecast[(last_recorded_ind+1):end, :]
        end
        @assert subtract_quarters(population_forecast[1, :date], last_recorded_date) == 1
    end

    # population_all: full unfiltered series (including forecast)
    population_all = if isempty(population_forecast)
        population_recorded[!,population_mnemonic]
    else
        pop_all = vcat(population_recorded, population_forecast)
        pop_all[!,population_mnemonic]
    end

    # HP filter
    if use_hpfilter
        population_all = convert(Array{Union{Float64, Missing}}, population_all)
        filtered_population, _ = hpfilter(population_all, 1600)
    end

    # Output dictionary for population data
    population_data_out = DataFrame()
    population_data_out[!,:date] = convert(Array{Date}, population_recorded[!,:date])
    population_data_out[!,:dlpopulation_recorded] = difflog(population_recorded[!,population_mnemonic])

    n_population_forecast_obs = size(population_forecast,1)

    if use_hpfilter
        filt_pop_recorded = filtered_population[1:end-n_population_forecast_obs]
        population_data_out[!,:filtered_population_recorded] = filt_pop_recorded
        population_data_out[!,:dlfiltered_population_recorded] = difflog(filt_pop_recorded)
    end

    # Output dictionary for population forecast
    population_forecast_out = DataFrame()
    if n_population_forecast_obs > 0
        population_forecast_out[!,:date] = convert(Array{Date}, population_forecast[!,:date])
        population_forecast_out[!,:dlpopulation_forecast] = difflog(population_forecast[!,population_mnemonic])

        if use_hpfilter
            filt_pop_fcast = filtered_population[end-n_population_forecast_obs:end]
            population_forecast_out[!,:filtered_population_forecast]   = filt_pop_fcast[2:end]
            population_forecast_out[!,:dlfiltered_population_forecast] = difflog(filt_pop_fcast)[2:end]
        end
    end

    return population_data_out, population_forecast_out
end

"""
```
function transform_spd_data(spd_df::DataFrame; column::Symbol = :MODAL_MEDIAN,
    use_last_survey::Bool = true, use_last_meeting::Bool = true)
```
Transform the raw SPD Modal Path data into a format matching that of OIS.
Two different forecasts of the same quarter taken at different times in the same quarter are averaged.
FFR forecasts of different FOMC meetings in the same quarter are aggregated using the implied daily FFR.
But if data given for quarter rather than meeting, use that value for whole quarter.

### Inputs

- `spd_df`: Modal forecasts dataset from the NY Fed's SPD.

### Keyword Arguments

- `column`: Which column's values to return (median, 25th, or 75th percentile)
- `use_last_survey`: whether to use the last survey in a quarter.
    If false, both surveys in a quarter are averaged.
- `use_last_meeting`: whether to use the last meeting in a quarter for the forecast.
    If false, the forecasted FFR comes from the implied daily FFR.
- `remove_zlb`: whether to remove instances when the median forecast is at the ZLB.
"""
function transform_spd_data(df::DataFrame; column::Symbol = :MODAL_MEDIAN,
                            use_last_survey::Bool = true, use_last_meeting::Bool = true,
                            remove_zlb::Bool = true)
    # Copy to ensure we don't change df
    spd_df = copy(df)

    # Get Quarter forecast is made in
    str_len = length.(spd_df[!,:SURVEY_MTG_DT])
    start_qtr = SubString.(spd_df[!,:SURVEY_MTG_DT], 1, str_len .- 2) .* "20" .*
        SubString.(spd_df[!,:SURVEY_MTG_DT], str_len .- 1, str_len)
    start_qtr = Date.(start_qtr, "d-u-y")
    insertcols!(spd_df, :survey_date => start_qtr)
    insertcols!(spd_df, :date => DSGE.quartertodate.(string.(year.(start_qtr)) .* "Q" .* string.(DSGE.datetoquarter.(start_qtr))))

    # Take avg of modal forecasts made within a quarter
    if use_last_survey
        sort!(spd_df, [:survey_date])
        gd = groupby(spd_df, :date)
        last_surveys = combine(gd, :survey_date => (x -> x[end]), renamecols = false)

        new_df = subset(spd_df, :survey_date => ByRow((x -> x in last_surveys[!,:survey_date])))
    else
        gd = groupby(spd_df, [:date, :PREDICTION_HORIZON_DATE])
        new_df = combine(gd, column => mean, renamecols = false)
    end

    # Get how many qtrs ahead a forecast is
    predicted_qtr = Vector{Date}(undef,nrow(new_df))
    for i in 1:nrow(new_df)
        # Get quarter of forecast
        if occursin("/", new_df[i,:PREDICTION_HORIZON_DATE])
            datei = Date(new_df[i,:PREDICTION_HORIZON_DATE],"m/d/y")
            predicted_qtr[i] = DSGE.quartertodate(string(year(datei)) * "Q" * string(DSGE.datetoquarter(datei)))
        elseif occursin("q", new_df[i,:PREDICTION_HORIZON_DATE])
            predicted_qtr[i] = DSGE.quartertodate(new_df[i,:PREDICTION_HORIZON_DATE])
        elseif occursin("h1", new_df[i,:PREDICTION_HORIZON_DATE])
            predicted_qtr[i] = DSGE.quartertodate(SubString(new_df[i,:PREDICTION_HORIZON_DATE],1,4) * "Q2")
        elseif occursin("h2", new_df[i,:PREDICTION_HORIZON_DATE])
            predicted_qtr[i] = DSGE.quartertodate(SubString(new_df[i,:PREDICTION_HORIZON_DATE],1,4) * "Q4")
        else
            predicted_qtr[i] = DSGE.quartertodate(new_df[i,:PREDICTION_HORIZON_DATE] * "Q4")
        end
    end
    # Get no. of qtrs ahead
    insertcols!(new_df, :predicted_qtr => predicted_qtr)
    insertcols!(new_df, :qtrs_ahead => DSGE.subtract_quarters.(predicted_qtr, new_df[!,:date]))
    ## insertcols!(new_df, :prev_meeting => vcat(new_df[1,:PREDICTION_HORIZON_DATE], new_df[1:end-1,:PREDICTION_HORIZON_DATE])) ## Only used when 2 qtrs_ahead rows for qtr, note 1st row never used
    insertcols!(new_df, :final_vals => new_df[!,column])

    sort!(new_df, [:date, :qtrs_ahead])
    insertcols!(new_df, :prev_val => vcat(new_df[1,column], new_df[1:end-1,column]))

    # Get last FOMC meeting date in qtrs when needed
    insertcols!(new_df, :last_meeting => new_df[!,:predicted_qtr])
    find_date_qtr = groupby(new_df, :predicted_qtr)
    for i in 1:length(find_date_qtr)
        date_given = findall(x -> occursin.("/", x), find_date_qtr[i][!,:PREDICTION_HORIZON_DATE])
        find_date_qtr[i][!,:last_meeting] .= if !isempty(date_given)
            maximum(Date.(find_date_qtr[i][date_given,:PREDICTION_HORIZON_DATE], "m/d/y"))
        else
            ## Use 1st day of qtr b/c the value is used for whole qtr.
            DSGE.iterate_quarters(find_date_qtr[i][1,:predicted_qtr],-1) + Dates.Day(1)
        end
    end

    # Combine forecasts that are the same i qtrs ahead from same start qtr
    ## Aggregate using implied daily FFR if !use_last_meeting
    if use_last_meeting
        new_df_inds = findall(i -> !occursin("/", new_df[i, :PREDICTION_HORIZON_DATE]) || new_df[i,:last_meeting] == Date(new_df[i,:PREDICTION_HORIZON_DATE], "m/d/y"), 1:nrow(new_df))
        new_df2 = new_df[new_df_inds,:]
    else
    gd2 = groupby(new_df, [:date, :qtrs_ahead])
    for i in 1:length(gd2)
        if nrow(gd2[i]) > 1 && all(occursin.("/", gd2[i][!,:PREDICTION_HORIZON_DATE]))
            horizon_dates = Date.(gd2[i][!,:PREDICTION_HORIZON_DATE], "m/d/y")

            fomc_diffs = (horizon_dates[2:end] .- horizon_dates[1:end-1]) ./ Dates.Day(1)#(Date.(gd2[i][2:end,:PREDICTION_HORIZON_DATE], "m/d/y") .- Date.(gd2[i][1:end-1,:PREDICTION_HORIZON_DATE], "m/d/y")) ./ Dates.Day(1)

            gd2[i][!,:final_vals] .= (sum(fomc_diffs .* gd2[i][1:end-1,column]) +
                                      (gd2[i][end,:predicted_qtr] - horizon_dates[end]) / Dates.Day(1) * gd2[i][end,column] +
                                      (horizon_dates[1] - DSGE.iterate_quarters(gd2[i][1,:predicted_qtr],-1)) / Dates.Day(1) * gd2[i][1,:prev_val]) /
                                      ((gd2[i][1,:predicted_qtr] - DSGE.iterate_quarters(gd2[i][1,:predicted_qtr], -1)) / Dates.Day(1))

        elseif nrow(gd2[i]) > 1 && any(occursin.("/", gd2[i][!,:PREDICTION_HORIZON_DATE]))
            if any(occursin.("/", gd2[i][!,:PREDICTION_HORIZON_DATE]))
                ## Find date corresponding to dates expressed as q or h
                no_date = findall(x -> !occursin.("/", x), gd2[i][!,:PREDICTION_HORIZON_DATE])
                gd2[i][no_date, :PREDICTION_HORIZON_DATE] = string.(month.(gd2[i][no_date,:last_meeting])) .* "/" .*
                    string.(day.(gd2[i][no_date,:last_meeting])) .* "/" .* string.(year.(gd2[i][no_date,:last_meeting]))

                ## Average those of same date
                subgd = combine(groupby(gd2[i], :PREDICTION_HORIZON_DATE), column => mean,
                                :predicted_qtr => (x -> x[1]), :prev_val => (x -> x[1]))

                ## Get daily aggregated value
                horizon_dates = Date.(subgd[!,:PREDICTION_HORIZON_DATE], "m/d/y")
                fomc_diffs = (horizon_dates[2:end] .- horizon_dates[1:end-1]) ./ Dates.Day(1)

                final_val = (sum(fomc_diffs .* subgd[1:end-1,Symbol(column,:_mean)]) +
                             (subgd[end,:predicted_qtr_function] - horizon_dates[end]) / Dates.Day(1) * subgd[end,Symbol(column,:_mean)] +
                             (horizon_dates[1] - DSGE.iterate_quarters(subgd[1,:predicted_qtr_function],-1)) / Dates.Day(1) * subgd[1,:prev_val_function]) /
                             ((subgd[1,:predicted_qtr_function] - DSGE.iterate_quarters(subgd[1,:predicted_qtr_function], -1)) / Dates.Day(1))

                gd2[i][!,:final_vals] .= final_val
            else
                gd2[i][!,:final_vals] .= mean(gd2[i][!,:final_vals])
            end
        # else: Data given for qtr so no change needed
        end
    end
    new_df2 = combine(gd2, :final_vals => mean, renamecols = false)
    end

    # Widen dataframe so there's a separate column for each qtr ahead.
    new_df2 = unstack(new_df2, :date, :qtrs_ahead, :final_vals)
    select!(new_df2, Not("0"))
    new_df2[:,2:end] .*= 25.0 ## Convert from annualized to quarterly*100

    # Sort and rename columns
    col_inds = sortperm(parse.(Int,names(new_df2)[2:end]))
    new_df2 = new_df2[:,vcat(1,col_inds .+ 1)]
    rename!(new_df2, vcat(names(new_df2)[1], "exp_ant" .* names(new_df2)[2:end]))

    # Remove periods when median forecast at ZLB if required

    return new_df2
end

"""
```
function spd_data_bands(spd_df::DataFrame)
```
Transform the raw SPD Modal Path data into a format matching that of OIS.
Returns 3 datasets, one each for 25%, 50%, and 75%.
"""
function spd_data_bands(spd_df::DataFrame)
    df_med = transform_spd_data(spd_df, column = :MODAL_MEDIAN)
    df_25 = transform_spd_data(spd_df, column = :MODAL_25TH)
    df_75 = transform_spd_data(spd_df, column = :MODAL_75TH)

    return df_med, df_25, df_75
end
