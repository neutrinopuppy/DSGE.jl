# BROAD STROKES
# 1. preprocess terminal & k_max periods back of zlb from every available liftoff policy
# 2. find l_t, k_t, for each alternative policy (full credibility), using the preprocessed
#    matrices above
# 3. mix (and no need to find l&k) :)

# NEW SETTINGS
# :l_max, :k_max
# :zlb_period_dicts => array of altpolicies, each corresponding to an array of
# zlbPeriodEntries
# :imperfect_credibility_weights => mapping of t to weights--if no weights in a given period,
# use the first zlb_period_dict's mats. only solve mats for altpols if there are weights!!

# NOTES
# the zlb_period setup is silly. I am devising a better version, will have it pushed in
# the morning.
# I still haven't really thought through the filtering setup.


# replacement for regime_eqcond_info? replace with a set of these per altpol?
# but we'd still need weights somewhere
# so, a new array to hold reg => weights...?
mutable struct zlbPeriodEntry
    start_date::Date
    end_date::Date # this would be empty unless we have a zlb of fixed length
    terminal_policy::AltPolicy
end

# terrible name. @me, do better
mutable struct altPolicyZLBDict
    zlb_periods::Array{zlbPeriodEntry}
    l_ts::Array{Int32}
    k_ts::Array{Int32}
    actual_policy_dict::Array{Symbol} # dict of t -> policy (during zlb, this would still
    # be liftoff policy)
end

# ^^ write constructors for these


# NOTE: @shlok if you end up working with this before I get it into readable form, I sincerely apologize

# NOTE: ALL OF THIS IS SET UP WITH L = 0. to add l, we'd need to add an l dimension to each
# entry in preprocessed_mats

# all pseudocode

function ocb_compute_system(m::AbstractDSGEModel; verbose::Symbol = :none)

    # Preprocess for each possible terminal condition
    # As Shlok notes in his doc, may be more effective to store these as we go. shall start with the simpler version
    # (preprocessing everything) and leave that to later
    l_max = get_setting(m, :l_max)
    k_max = get_setting(m, :k_max)

    preprocessed_mats = Dict()

    zlb_period_dicts = get_setting(m, :zlb_period_dicts)
    # Dict of terminal policy => [(end => (TTT, RRR, CCC)), (end-1 => (TTT, RRR, CCC)), ...]
    # should just be storing Ts?
    for zlb_period_dict in zlb_period_dicts
        for zlb_period in zlb_period_dict.zlb_periods
            # this all is set up as if we had perfect cred
            altkey = zlb_period.terminal_policy.key
            if !haskey(preprocessed_mats, altkey)
                preprocessed_mats[altkey] = Dict()
                # keys into this subdict are the length of the zlb prior to the entry matrices
                preprocessed_mats[altkey][0] = [zlb_period.terminal_policy.solve(m)]
                # this is a stupid way to do this. putting something on paper now, will fix it
                # later
                # Get equilibrium condition matrices for zlb
                Γ0_zlb, Γ1_zlb, C_zlb, Ψ_zlb, Π_zlb = zlb_rule_eqcond(m)
                # Get equilibrium condition matrices for terminal policy
                Γ0_fin, Γ1_fin, C_fin, Ψ_fin, Π_fin = zlb_period.terminal_policy.solve(m)
                # great code, me. peak efficiency. killin it. /s
                Γ0s = Array{Array{Float64}}(undef, k_max+1)
                Γ1s = Array{Array{Float64}}(undef, k_max+1)
                Cs = Array{Array{Float64}}(undef, k_max+1)
                Ψs = Array{Array{Float64}}(undef, k_max+1)
                Πs = Array{Array{Float64}}(undef, k_max+1)
                for k in 1:k_max
                    Γ0s[k] = Γ0_zlb
                    Γ1s[k] = Γ1_zlb
                    Cs[k]  = C_zlb
                    Ψs[k]  = Ψ_zlb
                    Πs[k]  = Π_zlb
                end
                # man, this is grim
                Γ0s[k] = Γ0_fin
                Γ1s[k] = Γ1_fin
                Cs[k]  = C_fin
                Ψs[k]  = Ψ_fin
                Πs[k]  = Π_fin

                # run recursion
                Tcals, Rcals, Ccals = gensys2(m, Γ0s, Γ1s, Cs, Ψs, Πs, preprocessed_mats[altkey][0][1], preprocessed_mats[altkey][0][2], preprocessed_mats[altkey][0][3], k_max) # this might be k_max + 1
                for k in 1:k_max
                    preprocessed_mats[altkey][k] = [Tcals[k_max+1-k], Rcals[k_max+1-k], Ccals[k_max+1-k]]
                end
            end
        end
    end


    # Use our modified binary search to run through possible ls and ks (currently, just
    # running through k, assuming l = 0) for all altpols
    # though our binary search mod is based on forecast path, which we'll no longer have
    # so this turns into a pure binary search
    # we need to solve for l_t, k_t for each alternative policy

    # this is stupid. why am I doing this for each zlb period (facepalm)
    # the whole zlb_regime setup as it currently exists is silly
    for zlb_regime_dict in zlb_period_dicts
        for t in 1:n # however many total regimes we have. there has to be a way to cut this
            # down
            # guess 0
            # guess k_max/2
            # binary search!! the simple, friendly version. huzzah.
            # since we aren't picking our initial guess off of a forecast path, I don't think
            # it really makes sense to use our +-a couple setup from current endo

            # we'd have to filter up the first zlb period start using talyor transitions
            # then get the transitions up to the next zlb period start, then filter again?
            # that seems pretty time intensive

            # ok, assuming we have a state for the current time period, s_t:
            k = 0
            h = k_max
            l = 0
            while true
                T, R, C = preprocessed_mats[zlb_regime_dict.actual_policy_dict[t]][k]
                # s_t1 = C + T*s_t
                # indexing
                if (nom > 0 & nom_t1 < 0) | (nom > 0 & k==0)
                    break
                elseif nom > 0 & nom_t1 > 0
                    h = k
                elseif nom < 0
                    l = k
                end
                k = floor((h-l)/2)+l
            end
            # figure out and implement the check

            # ^^ above finds l_t, k_t for the given perfcred policy
            zlb_regime_dict.l_ts[t] = 0
            zlb_regime_dict.k_ts[t] = k #whatever we found above
        end
    end

    # then mash them together!
    TTTs = Array{Array{Float64}}(undef, n)
    RRRs = Array{Array{Float64}}(undef, n)
    CCCs = Array{Array{Float64}}(undef, n)
    # this combination is based heavily on gensys_uncertain_altpol, see
    # src/solve/gensys_uncertain_altpol.jl:183
    for t in 1:n
        if haskey(get_setting(m, :imperfect_credibility_weights), t)
            weights = get_setting(m, :imperfect_credibility_seights)[t]

            # first, sort out the gamma tils:
            # hmmmm. this is probably wrong. why are we using one of the altpols over the
            # other, anyways?
            if zlb_period_dicts[1].l_ts[t] = 0 && zlb_period_dicts[1].k_ts[t] > 0
                #then we are in a zlb regime
                Γ0, Γ1, C, Ψ, Π = zlb_rule_eqcond(m)
                # then we find what policy we're currently under. it occurs to me that the zlb
                # period setup I have currently written up assumes that we change policies only
                # after a zlb, which is obviously problematic.
            else
                current_policy = AltPol(:talyor) # whatever the format for this is
                for i = 1:length(zlb_period_dicts[1].zlb_periods)
                    # this is not how regime_dates works. we'd need to add a setting/change this
                    # one to fit
                    if zlb_period_dicts[1].zlb_periods[i].start_date <= get_setting(m, :regime_dates)[i]
                        current_policy = zlb_period_dicts[1].zlb_periods[i].terminal_policy
                    else
                        break
                    end
                end
                Γ0, Γ1, C, Ψ, Π = current_policy.eqcond(m)
            end

            Γ0_til, Γ1_til, Γ2_til, C_til, Ψ_til = gensys_to_predictable_form(Γ0, Γ1, C, Ψ, Π)

            # pick out the liftoff policies we're mixing
            actual_policies = Array{Symbol}(undef, length(weights))
            for i = 1:length(weights)
                actual_policies[i] = zlb_period_dicts[i].actual_policy_dict[t]
            end

            # ok, now take the combination (using the relevant preprocessed matrices
            # indexed by each pols l_ts&k_ts for the t in question
            inds = 1:n_states(m)
            T̅, C̅ = (prob_vec[1] == 0.) ? (zeros(size(Γ0_til)), zeros(size(C))) :
            (prob_vec[1] .* (@view preprocessed_mats[actual_policies[1]][zlb_period_dicts[1].k_ts[t]][1][inds, inds]), prob_vec[1] .* (@view preprocessed_mats[actual_policies[1]][zlb_period_dicts[1].k_ts[t]][2][inds]))

            has_pos_prob = findall(x -> x > 0., (@view weights[2:end]))

            for i in has_pos_prob
                T̅ .+= weights[i + 1] * (@view preprocessed_mats[actual_policies[i]][zlb_period_dicts[i].k_ts[t]][1][inds, inds])
                C̅ .+= weights[i + 1] * (@view preprocessed_mats[actual_policies[i]][zlb_period_dicts[i].k_ts[t]][2][inds])
            end

            Lmat = (Γ2_til * T̅ + Γ0_til)
            TTTs[t] = Lmat \ Γ1_til
            RRRs[t] = Lmat \ Ψ_til
            CCCs[t] = Lmat \ (C_til - Γ2_til * C̅)
        else

            actual_policy = zlb_period_dicts[1].actual_policy_dict[t] # this is liftoff pol in zlb
            TTTs[t], RRRs[t], CCCs[t] = preprocessed_mats[actual_policy][zlb_period_dicts[1].k_ts[t]]
        end
    end

    # sort out measurement eqs

    # Return array of transition matrices (... these could probably change before covid regimes now too,
    # which is fun)
    return TTTs, RRRs, CCCs

end

