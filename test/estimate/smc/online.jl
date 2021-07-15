using DSGE, ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test, Suppressor

path = dirname(@__FILE__)
writing_output = false
if VERSION < v"1.5"
    ver = "111"
elseif VERSION < v"1.6"
    ver = "150"
else
    ver = "160"
end

m = AnSchorfheide()

save = normpath(joinpath(dirname(@__FILE__), "save"))
m <= Setting(:saveroot, save)

data = h5read(joinpath(path, "reference/smc.h5"), "data")

m <= Setting(:n_particles, 400)
m <= Setting(:n_Φ, 100)
m <= Setting(:λ, 2.0)
m <= Setting(:n_smc_blocks, 1)
m <= Setting(:use_parallel_workers, true)
m <= Setting(:step_size_smc, 0.5)
m <= Setting(:n_mh_steps_smc, 1)
m <= Setting(:resampler_smc, :polyalgo)
m <= Setting(:target_accept, 0.25)

m <= Setting(:mixture_proportion, 0.9)
m <= Setting(:adaptive_tempering_target_smc, false)
m <= Setting(:resampling_threshold, 0.5)
m <= Setting(:smc_iteration, 0)
m <= Setting(:use_chand_recursion, true)


verbose = :low
use_chand_recursion = true

# Estimate with full sample
m = deepcopy(m)
m <= Setting(:n_particles, 1000, true, "npart", "")
m <= Setting(:data_vintage, "210714")

@everywhere Random.seed!(42)

# estimate model with full data

savepath_full = rawpath(m, "estimate", "smc_cloud.jld2")

#DSGE.smc2(m, data; verbose = verbose)

full_file   = load(rawpath(m, "estimate", "smc_cloud.jld2"))
full_cloud  = full_file["cloud"]


# Estimate with 1st half of sample
m_old = deepcopy(m)
m_old <= Setting(:n_particles, 1000, true, "npart", "")
m_old <= Setting(:data_vintage, "000000")

savepath_old = rawpath(m_old, "estimate", "smc_cloud.jld2")
loadpath_old = rawpath(m_old, "estimate", "smc_cloud.jld2")

println("Estimating Initial AnSchorfheide Model... (approx. 2 minutes)")

@suppress begin
#    DSGE.smc2(m_old, data[:, 1:Int(floor(end/2))]; verbose = verbose)
end

println("Initial estimation done!")


m_new = deepcopy(m)

# Estimate with 2nd half of sample
m_new <= Setting(:data_vintage, "200218")
m_new <= Setting(:tempered_update_prior_weight, 0.5)
m_new <= Setting(:tempered_update, true)
old_vint = "000000"
new_vint = "200218"

savepath_new = rawpath(m_new, "estimate", "smc_could.jld2")

loadpath = rawpath(m_old, "estimate", "smc_cloud.jld2")
loadpath = replace(loadpath, r"vint=[0-9]{6}" => "vint=" * old_vint)

old_cloud = load(loadpath, "cloud")

m_new <= Setting(:previous_data_vintage, old_vint)
println("Beginning online estimation")
#@suppress begin
#    DSGE.smc2(m_new, data; verbose = verbose, old_data = data[:,1:Int(floor(end/2))],old_cloud = old_cloud,
#              tempered_update_prior_weight = 0.5)
#end
println("Finished online estimation")


loadpath_new = replace(loadpath, r"vint=[0-9]{6}" => "vint=" * new_vint)
online_cloud = load(loadpath_new, "cloud")
