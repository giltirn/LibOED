mutable struct InferenceResult
    dist_properties::NamedArray{Any,2} #[chain idx][func name] => statistic : results of applying each of the "dist_properties" functions to each chain separately
    avg_dist_properties::NamedArray{Any,1} #[func_name] => avg statistic  : results of averaging each of the "dist_properties" over all chains

    param_dist_properties::NamedArray{Any,3} #[chain idx][param name][func name] => statistic :  results of applying each of the "param_dist_properties" functions to each chain separately
    avg_param_dist_properties::NamedArray{Any,2} #[param name][func_name] => avg statistic  :  results of averaging each of the "param_dist_properties" over all chains

    chains::Union{Nothing,Vector{Chains}} #optional: output the chains themselves if enabled
    base_seed::Int64 #the base seed used by the internal RNGs. The seed for a given chain is base_seed + chain_idx
end

strf(f) = String(nameof(f))

function InferenceResult(dist_properties::Array{F,1}, param_dist_properties::Array{G,1}, param_list::Vector{String}, n::Integer, output_chains::Bool, base_seed::Int64) where {F<:Function,G<:Function}
    fcolnames = [strf(f) for f in dist_properties]
    pcolnames = [strf(f) for f in param_dist_properties]

    o_dist_properties = NamedArray(Any,0,0)
    o_avg_dist_properties = NamedArray(Any,0)
    o_param_dist_properties = NamedArray(Any,0,0,0)
    o_avg_param_dist_properties = NamedArray(Any,0,0)
    
    if(length(fcolnames)>0)
        o_dist_properties = NamedArray(Any, n, length(fcolnames))
        setdimnames!(o_dist_properties, "Sample", 1)
        setdimnames!(o_dist_properties, "Statistic", 2)
        setnames!(o_dist_properties, fcolnames, 2)
        
        o_avg_dist_properties = NamedArray(Any, length(fcolnames))
        setdimnames!(o_avg_dist_properties, "Statistic", 1)
        setnames!(o_avg_dist_properties, fcolnames, 1)
    end

    if(length(pcolnames)>0)
        o_param_dist_properties = NamedArray(Any, n, length(param_list), length(pcolnames))
        setdimnames!(o_param_dist_properties, "Sample", 1)
        setdimnames!(o_param_dist_properties, "Param", 2)
        setnames!(o_param_dist_properties, param_list, 2)
        setdimnames!(o_param_dist_properties, "Statistic", 3)
        setnames!(o_param_dist_properties, pcolnames, 3)
        
        o_avg_param_dist_properties = NamedArray(Any, length(param_list), length(pcolnames))
        setdimnames!(o_avg_param_dist_properties, "Param", 1)
        setnames!(o_avg_param_dist_properties, param_list, 1)
        setdimnames!(o_avg_param_dist_properties, "Statistic", 2)
        setnames!(o_avg_param_dist_properties, pcolnames, 2)
    end
        
    return InferenceResult(o_dist_properties, o_avg_dist_properties, o_param_dist_properties, o_avg_param_dist_properties, 
                           output_chains ? Vector{Chains}(undef,n) : nothing,
                           base_seed)
end

function replace_param_names!(inf::InferenceResult, dict::AbstractDict)
    npcols = size(inf.param_dist_properties,3)
    if(npcols > 0)
        pnames = names(inf.param_dist_properties,2)
        for i in 1:length(pnames)
            if(haskey(dict,pnames[i])); pnames[i] = dict[pnames[i]]; end
        end
        setnames!(inf.param_dist_properties, pnames, 2)
        setnames!(inf.avg_param_dist_properties, pnames, 1)
    end
    if(inf.chains != nothing)
        for i in 1:length(inf.chains)
            inf.chains[i] = replacenames(inf.chains[i], dict)
        end
    end
end


#Run n chains on the worker, where n is length(y_samples)
function worker_func(dist, driver, y_samples, chain_idx_offset, chain_length, base_seed, mcmc_sampler, dist_properties, param_dist_properties, output_chains::Bool)
    n = length(y_samples)

    thr_dist_prop = Dict{String, Array{Any,1}}( strf(f) => Array{Any,1}(undef, n) for f in dist_properties )

    #problem; there is no easy way to obtain the names of the parameters from the model that works for both array-type parameters and regular parameters
    #these can only easily be inferred from the Chains objects AFAICT
    #as such we cannot fully initialize the map forcing us to put the chain index on the outer index
    thr_pdist_prop = [ Dict{String,Dict{String,Any}}( strf(f) => Dict{String,Any}() for f in param_dist_properties ) for c in 1:n ]

    chains = output_chains ? Vector{Any}(undef, n) : nothing
    
    @threads for c in 1:n
        y = y_samples[c]
        rng=Xoshiro(base_seed + chain_idx_offset + c)
        chain = sample(rng, dist(y,driver), mcmc_sampler, chain_length)

        #Properties of model return distribution
        if(length(dist_properties) > 0)        
            Q = generated_quantities(dist(y,driver), Turing.MCMCChains.get_sections(chain, :parameters))
            for f in dist_properties
                thr_dist_prop[strf(f)][c] = f(Q)
            end
        end

        #Properties of parameter distributions
        if(length(param_dist_properties) > 0)
            S = summarize(chain, param_dist_properties...)
            cparams = chain.name_map.parameters

            println("Chain param list: ",chain.name_map.parameters)
            for f in param_dist_properties
                for p in cparams
                    thr_pdist_prop[c][strf(f)][String(p)] = getindex(S,p,nameof(f))
                end
            end           
        end        

        #Optional returning of full chain
        if(output_chains); chains[c] = chain; end
        
    end

    #reorder thr_pdist_prop
    thr_pdist_prop_reord = Dict{String,Dict{String, Array{Any,1}} }()
    if(length(param_dist_properties) > 0)
        pkeys = keys(thr_pdist_prop[1][strf(param_dist_properties[1])])
        thr_pdist_prop_reord = Dict{String,Dict{String, Array{Any,1}} }( strf(f) => Dict{String,Array{Any,1}}( p => Array{Any,1}(undef, n) for p in pkeys) for f in param_dist_properties )
        for f in param_dist_properties
            fname = strf(f)
            for p in pkeys        
                for c in 1:n
                    thr_pdist_prop_reord[fname][p][c] = thr_pdist_prop[c][fname][p]
                end
            end
        end
    end
    
    return (thr_dist_prop, thr_pdist_prop_reord, chains)   
end

    
#Divide work over procs
function divide_work(n, nproc)
    nw_base = div(n,nproc)
    nrem = n - nw_base * nproc
    nw::Array{Int64} = fill(nw_base,nproc)
    for i in 1:nrem
        nw[nproc-i+1] += 1  #add from end 
    end
    return nw
end

const global_base_seed = Ref{Int64}(1234)

#dist: two-argument function that takes the y-sample and the driver; this is expected to wrap a Turing model
#dist_properties: a list of functions that are applied for each chain to the distribution of model return values (obtained via generated_quantities)
#base_seed: a RNG is seeded for each chain as base_seed + chain_idx
function simulate_inference(dist, driver, N_samp, y_sampler; chain_length=1000, mcmc_sampler=NUTS(0.65), dist_properties::Union{Nothing,Array{F,1}}=[var], param_dist_properties::Union{Nothing,Array{G,1}} = nothing, output_chains::Bool=false, base_seed::Int64=global_base_seed[] )  where {F<:Function,G<:Function}
    if(dist_properties === nothing); dist_properties = Array{Function,1}(); end
    if(param_dist_properties === nothing); param_dist_properties = Array{Function,1}(); end
    
    if( !output_chains && length(dist_properties) == 0 && length(param_dist_properties) == 0); error("Must provide at least one function if not outputing raw chains"); end

    nproc = nprocs()
    work = divide_work(N_samp, nproc)

    println("Dividing ",N_samp, " samples over ",nproc, " processors")
    println("Work distribution: ")
    for i in 1:nproc
        print(i,":",work[i]," ")
        if i % 5 == 0
            print("\n")
        end
    end
    if nproc % 5 != 0; println(""); end

    results = Vector{Any}(undef, nproc)
    
    if(nproc > 1)    
        wf =Array{Any,1}(undef, nproc) #handles for async processes
        
        #Spawn work on main process last!
        off = work[1]
        for i in 2:nproc
            if work[i] > 0
                y_samples = Array{Any}(undef, work[i]) #generate the y_samples for the process here so they can be automatically copied to the remote process
                for c in 1:work[i]
                    y_samples[c] = y_sampler(off + c)
                end
                wf[i] = @spawnat i worker_func(dist, driver, y_samples, off, chain_length, base_seed,
                                               mcmc_sampler, dist_properties, param_dist_properties, output_chains)
                off += work[i]
            end
        end
        if work[1] > 0 #main process
            y_samples = Array{Any}(undef, work[1])
            for c in 1:work[1]
                y_samples[c] = y_sampler(c)
            end
            wf[1] = @spawnat 1 worker_func(dist, driver, y_samples, 0, chain_length, base_seed,
                                           mcmc_sampler, dist_properties, param_dist_properties, output_chains)
        end

        #Fetch results after main process has finished
        for i in 1:nproc
            if work[i] > 0
                r = fetch(wf[i])
                if isa(r,Exception)
                    error(r)
                end
                results[i] = r
            end
        end
        
    else
        if(work[1] != N_samp); error("Work division error for single process"); end
        y_samples = [ y_sampler(c) for c in 1:N_samp ]
        results[1] = worker_func(dist,driver, y_samples, 0, chain_length, base_seed,
                                 mcmc_sampler, dist_properties, param_dist_properties, output_chains)
    end

    #Get the list of params so we can initialize output (only needed if using param_dist_properties)
    param_list = Vector{String}()
    if(length(param_dist_properties) > 0)
        if(work[1] == 0); error("Expect first proc to have work"); end
        param_list = [k for k in keys(results[1][2][strf(param_dist_properties[1])]) ]::Vector{String}
        println("Model parameters: ", param_list)
    end

    #Initialize output
    out = InferenceResult(dist_properties, param_dist_properties, param_list, N_samp, output_chains, base_seed)
    
    #Extract and combine data   
    dfkeys = [strf(f) for f in dist_properties]
    pfkeys = [strf(f) for f in param_dist_properties]
    off = 0
    for i in 1:nproc
        if work[i] > 0

            #Extract dist_properties data
            if(length(dist_properties) > 0)
                for fkey in dfkeys              
                    rf = results[i][1][fkey]
                
                    if length(rf) != work[i]
                        error("Unexpected amount of work (dist_properties)!")
                    end
                
                    out.dist_properties[off+1:off+work[i], fkey] = rf[:]
                end
            end

            #Extract param_dist_properties data
            if(length(param_dist_properties) > 0)
                for fkey in pfkeys
                    for p in param_list                    
                        rf = results[i][2][fkey][p]
                
                        if length(rf) != work[i]
                            error("Unexpected amount of work (param_dist_properties)!")
                        end
                
                        out.param_dist_properties[off+1:off+work[i],p,fkey] = rf[:]
                    end
                end
            end

            #Extract chains
            if(output_chains)
                rf = results[i][3]
                if length(rf) != work[i]
                    error("Unexpected amount of work (chains)!")
                end
                out.chains[off+1:off+work[i]] = rf[:]
            end
            
            off += work[i]
        end
    end

    #Averages for dist_properties
    for fkey in dfkeys
        out.avg_dist_properties[fkey] = mean(out.dist_properties[:,fkey])
    end

    #Averages for param_dist_properties
    for fkey in pfkeys
        for p in param_list
            out.avg_param_dist_properties[p,fkey] = mean(out.param_dist_properties[:,p,fkey])
        end
    end

    #Increment global_base_seed to ensure next call uses different seed for all chains
    global_base_seed[] += N_samp
    
    return out
end

#y_samples: 2-d array with samples in columns
function simulate_inference(dist, driver, y_samples::AbstractMatrix{T}; chain_length=1000, mcmc_sampler=NUTS(0.65), dist_properties=[var], param_dist_properties=nothing, output_chains::Bool=false, base_seed::Int64=global_base_seed[] ) where T<:Number
    y_sampler(i) = y_samples[:,i]
    N_samp = size(y_samples,2) #number of columns
    simulate_inference(dist, driver, N_samp, y_sampler; chain_length=chain_length, mcmc_sampler=mcmc_sampler, dist_properties=dist_properties, param_dist_properties=param_dist_properties, output_chains=output_chains, base_seed=base_seed)
end
