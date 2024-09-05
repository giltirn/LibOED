struct InferenceResult
    dist_properties::NamedArray{Any,2} #[chain idx][func name] => statistic : results of applying each of the "dist_properties" functions to each chain separately
    avg_dist_properties::NamedArray{Any,1} #[func_name] => avg statistic  : results of averaging each of the "dist_properties" over all chains

    param_dist_properties::NamedArray{Any,3} #[chain idx][param name][func name] => statistic :  results of applying each of the "param_dist_properties" functions to each chain separately
    avg_param_dist_properties::NamedArray{Any,2} #[param name][func_name] => avg statistic  :  results of averaging each of the "param_dist_properties" over all chains

    chains::Union{Nothing,Vector{Chains}} #optional: output the chains themselves if enabled
    base_seed::Int64 #the base seed used by the internal RNGs. The seed for a given chain is base_seed + chain_idx
end

function getFuncName(f::Tuple{AbstractString,Function})
    return f[1]
end
function getFuncName(f::Function)
    return String(nameof(f))
end

function getFunc(f::Tuple{AbstractString,Function})
    return f[2]
end
function getFunc(f::Function)
    return f
end


function InferenceResult(dist_properties_funcs, param_dist_properties_funcs, param_list::Vector{String}, n::Integer, output_chains::Bool, base_seed::Int64)
    fcolnames = [getFuncName(f) for f in dist_properties_funcs]
    pcolnames = [getFuncName(f) for f in param_dist_properties_funcs]

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
        println("Original parameters: ",pnames)
        for i in 1:length(pnames)
            if(haskey(dict,pnames[i])); pnames[i] = String(dict[pnames[i]]); end
        end
        println("New parameters: ", pnames)
        setnames!(inf.param_dist_properties, pnames, 2)
        setnames!(inf.avg_param_dist_properties, pnames, 1)
    end
    if(inf.chains != nothing)
        for i in 1:length(inf.chains)
            if(i==1); println("Original Chain parameters: ", inf.chains[i].name_map.parameters); end
            inf.chains[i] = replacenames(inf.chains[i], dict)
            if(i==1); println("New Chain parameters: ", inf.chains[i].name_map.parameters); end
        end
    end
end


#Run n chains on the worker, where n is length(y_samples)
function worker_func(dist, driver, y_samples, chain_idx_offset, chain_length, base_seed, mcmc_sampler, dist_properties, param_dist_properties, output_chains::Bool)
    n = length(y_samples)

    thr_dist_prop = nothing
    if(length(dist_properties)>0)
        thr_dist_prop = NamedArray(Any, n, length(dist_properties)) #[chain idx, func idx]
        setnames!(thr_dist_prop, [getFuncName(f) for f in dist_properties], 2)
    end

    thr_pdist_prop = nothing
    if(length(param_dist_properties)>0)
        #problem; there is no easy way to obtain the names of the parameters from the model that works for both array-type parameters and regular parameters
        #these can only easily be inferred from the Chains objects AFAICT
        #as such we cannot fully initialize a multidimensional array forcing us to put the chain index on the outer index
        thr_pdist_prop = Vector{ NamedArray{Any,2} }(undef, n) #[chain idx][param idx, func idx]
    end

    chains = output_chains ? Vector{Any}(undef, n) : nothing
    
    @threads for c in 1:n
        y = y_samples[c]
        rng=Xoshiro(base_seed + chain_idx_offset + c)
        chain = sample(rng, dist(y,driver), mcmc_sampler, chain_length)

        #Properties of model return distribution
        if(length(dist_properties) > 0)        
            Q = generated_quantities(dist(y,driver), Turing.MCMCChains.get_sections(chain, :parameters))
           
            for f in dist_properties
                thr_dist_prop[c,getFuncName(f)] = getFunc(f)(Q)
            end
        end

        #Properties of parameter distributions
        if(length(param_dist_properties) > 0)
            fnames = [ getFuncName(f) for f in param_dist_properties ]
            funcs = [ getFunc(f) for f in param_dist_properties ]
            
            S = summarize(chain, funcs...; func_names=[Symbol(fn) for fn in fnames])
            cparams = chain.name_map.parameters
            thr_pdist_prop[c] = NamedArray(Any, length(chain.name_map.parameters), length(param_dist_properties) )
            setnames!(thr_pdist_prop[c], [String(f) for f in chain.name_map.parameters], 1)
            setnames!(thr_pdist_prop[c], fnames, 2)
            
            for f in param_dist_properties
                for p in cparams
                    thr_pdist_prop[c][String(p),getFuncName(f)] = getindex(S,p,Symbol(getFuncName(f)))
                end
            end           
        end        

        #Optional returning of full chain
        if(output_chains); chains[c] = chain; end
        
    end

   
    #reorder thr_pdist_prop
    thr_pdist_prop_reord = nothing
    if(length(param_dist_properties) > 0)
        #check all chains agree on params list
        pkeys=names(thr_pdist_prop[1],1)
        for c in 2:n
            if names(thr_pdist_prop[c],1) != pkeys; error("Name mismatch, got ", names(thr_pdist_prop[c],1), " expect ", pkeys); end
        end
        thr_pdist_prop_reord = NamedArray(Any, n, length(pkeys), length(param_dist_properties))
        setnames!(thr_pdist_prop_reord, pkeys, 2)
        setnames!(thr_pdist_prop_reord, [getFuncName(f) for f in param_dist_properties], 3)

        for c in 1:n
            thr_pdist_prop_reord[c, :, :] = thr_pdist_prop[c][:,:]
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
function simulate_inference(dist, driver, N_samp, y_sampler; chain_length=1000, mcmc_sampler=NUTS(0.65), dist_properties=[var], param_dist_properties = nothing, output_chains::Bool=false, base_seed::Int64=global_base_seed[] )
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
        param_list = names(results[1][2],2)
        println("Model parameters: ", param_list)
    end

    #Initialize output
    out = InferenceResult(dist_properties, param_dist_properties, param_list, N_samp, output_chains, base_seed)
    
    #Extract and combine data   
    dfkeys = [getFuncName(f) for f in dist_properties]
    pfkeys = [getFuncName(f) for f in param_dist_properties]
    off = 0
    for i in 1:nproc
        if work[i] > 0
            #Extract dist_properties data
            if(length(dist_properties) > 0)
                rf = results[i][1]
                if(size(rf,1) != work[i]); error("Unexpected amount of work (dist_properties)!"); end
                if(size(rf,2) != length(dist_properties)); error("Unexpected amount of funcs (dist_properties)!"); end
                if(names(out.dist_properties,2) != names(rf,2)); error("Function names mismatch (dist_properties)!"); end
                out.dist_properties[off+1:off+work[i], :] = rf[:,:]
            end

            #Extract param_dist_properties data
            if(length(param_dist_properties) > 0)
                rf = results[i][2]
                if(size(rf,1) != work[i]); error("Unexpected amount of work (param_dist_properties)!"); end
                if(size(rf,2) != length(param_list)); error("Unexpected amount of params (param_dist_properties)!"); end
                if(size(rf,3) != length(param_dist_properties)); error("Unexpected amount of funcs (param_dist_properties)!"); end
                if(names(out.param_dist_properties,2) != names(rf,2)); error("Parameter names mismatch (param_dist_properties)!"); end
                if(names(out.param_dist_properties,3) != names(rf,3)); error("Function names mismatch (param_dist_properties)!"); end
                out.param_dist_properties[off+1:off+work[i], :, :] = rf[:,:,:]
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
        #Depending on the generated_quantities type for the Turing model, the concept of mean may be undefined; in which case we set the entry to nothing
        try
            v = Vector(out.dist_properties[:,fkey])
            out.avg_dist_properties[fkey] = mean(v)
            #out.avg_dist_properties[fkey] = mean(out.dist_properties[:,fkey]) #sometimes fails MethodError: no method matching length(::Colon)
        catch
            out.avg_dist_properties[fkey] = nothing
        end
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
