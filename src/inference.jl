struct InferenceResult
    dist_properties::NamedArray{Any,2} #[chain idx][func name] => statistic : results of applying each of the "dist_properties" functions to each chain separately
    avg_dist_properties::NamedArray{Any,1} #[func_name] => avg statistic  : results of averaging each of the "dist_properties" over all chains

    param_dist_properties::NamedArray{Any,3} #[chain idx][param name][func name] => statistic :  results of applying each of the "param_dist_properties" functions to each chain separately
    avg_param_dist_properties::NamedArray{Any,2} #[param name][func_name] => avg statistic  :  results of averaging each of the "param_dist_properties" over all chains

    posterior_derived_quants::NamedArray{Any, 2} #[chain idx][func name] => derived quantity : results of applying an arbitrary operation to the chain itself, eg sampling from it and computing a derived quantity
    
    chains::Union{Nothing,Vector{Chains}} #optional: output the chains themselves if enabled
    base_seed::Int64 #the base seed used by the internal RNGs. The seed for a given chain is base_seed + chain_idx
end

function getFuncName(f::Tuple{Symbol,Function})
    return f[1]
end
function getFuncName(f::Function)
    return nameof(f)
end

function getFunc(f::Tuple{Symbol,Function})
    return f[2]
end
function getFunc(f::Function)
    return f
end


function InferenceResult_(dist_properties_funcs, param_dist_properties_funcs, posterior_funcs, param_list::Vector{Symbol}, n::Integer, output_chains::Bool, base_seed::Int64)
    fcolnames = [getFuncName(f) for f in dist_properties_funcs]
    pcolnames = [getFuncName(f) for f in param_dist_properties_funcs]
    postcolnames = [getFuncName(f) for f in posterior_funcs]

    chainidxlabels = [string(i) for i in 1:n]
    
    o_dist_properties = NamedArray(Any,0,0)
    o_avg_dist_properties = NamedArray(Any,0)
    o_param_dist_properties = NamedArray(Any,0,0,0)
    o_avg_param_dist_properties = NamedArray(Any,0,0)
    o_posterior_derived_quants = NamedArray(Any,0,0)
    
    if(length(fcolnames)>0)
        o_dist_properties = NamedArray( Array{Any,2}(undef, n, length(fcolnames)); names=(chainidxlabels, fcolnames), dimnames=(:Sample,:Statistic) )
        o_avg_dist_properties = NamedArray( Vector{Any}(undef, length(fcolnames)); names=(fcolnames,), dimnames=(:Statistic,) )
    end

    if(length(pcolnames)>0)
        o_param_dist_properties = NamedArray( Array{Any,3}(undef, n, length(param_list), length(pcolnames)); names=(chainidxlabels, param_list, pcolnames), dimnames=(:Sample,:Param,:Statistic) )
        o_avg_param_dist_properties = NamedArray( Array{Any,2}(undef, length(param_list), length(pcolnames)); names=(param_list, pcolnames), dimnames=(:Param,:Statistic) )
    end

    if(length(postcolnames)>0)
        o_posterior_derived_quants = NamedArray( Array{Any,2}(undef, n, length(postcolnames)); names=(chainidxlabels, postcolnames), dimnames=(:Sample,:Quantity) ) 
    end
        
    return InferenceResult(o_dist_properties, o_avg_dist_properties, o_param_dist_properties, o_avg_param_dist_properties, o_posterior_derived_quants,
                           output_chains ? Vector{Chains}(undef,n) : nothing,
                           base_seed)
end

function replace_param_names!(inf::InferenceResult, dict::AbstractDict)
    npcols = size(inf.param_dist_properties,3)
    if(npcols > 0)
        pnames = names(inf.param_dist_properties,2)
        println("Original parameters: ",pnames)
        for i in 1:length(pnames)
            if(haskey(dict,pnames[i])); pnames[i] = dict[pnames[i]]; end
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
function worker_func(dist, inputs, chain_idx_offset, chain_length, base_seed, mcmc_sampler, dist_properties, param_dist_properties, posterior_operations, output_chains::Bool, initialize_MAP::Bool, thread_block_size::Int64)
    n = length(inputs)
    rnames = [string(i) for i in 1:n]    
    
    thr_dist_prop = length(dist_properties)>0 ?
        NamedArray(Array{Any,2}(undef, n, length(dist_properties)); names=(rnames,[getFuncName(f) for f in dist_properties]) ) : nothing   #[chain idx, func name]

    #problem; there is no easy way to obtain the names of the parameters from the model that works for both array-type parameters and regular parameters
    #these can only easily be inferred from the Chains objects AFAICT
    #as such we cannot fully initialize a multidimensional array forcing us to put the chain index on the outer index
    thr_pdist_prop = length(param_dist_properties)>0 ? Vector{ NamedArray{Any,2} }(undef, n) : nothing  #[chain idx][param idx, func name]

    thr_post_op = length(posterior_operations)>0 ?
        NamedArray(Array{Any,2}(undef, n, length(posterior_operations)); names=(rnames,[getFuncName(f) for f in posterior_operations])) : nothing   #[chain idx, func name]
    
    chains = output_chains ? Vector{Any}(undef, n) : nothing

    if(thread_block_size <= 0 || thread_block_size > Threads.nthreads())
        thread_block_size = Threads.nthreads()
    end
    
    blocks = div(n + thread_block_size - 1, thread_block_size)
    off = 0
    for b in 1:blocks
        bs = min(thread_block_size, n-off)
        
        @threads for cc in 1:bs
            c = cc + off
            
            y = inputs[c]
            rng=Xoshiro(base_seed + chain_idx_offset + c)
            turing_model = dist(y)
            
            if(initialize_MAP)
                println("Generating maximum a priori initialization for worker chain ",c)
                map_estimate = maximum_a_posteriori(turing_model)
                println("Sampling worker chain ",c)
                chain = sample(rng, turing_model, mcmc_sampler, chain_length; initial_params=map_estimate.values.array)
            else
                println("Sampling worker chain ",c)
                chain = sample(rng, turing_model, mcmc_sampler, chain_length)
            end

            #Properties of model return distribution
            if(length(dist_properties) > 0)        
                Q = generated_quantities(turing_model, Turing.MCMCChains.get_sections(chain, :parameters))
                
                for f in dist_properties
                    thr_dist_prop[c,getFuncName(f)] = getFunc(f)(Q)
                end
            end

            #Properties of parameter distributions
            if(length(param_dist_properties) > 0)
                fnames = [ getFuncName(f) for f in param_dist_properties ]
                funcs = [ getFunc(f) for f in param_dist_properties ]
                
                S = summarize(chain, funcs...; func_names=fnames)
                cparams = chain.name_map.parameters
                thr_pdist_prop[c] = NamedArray( Array{Any}(undef, length(chain.name_map.parameters), length(param_dist_properties) ); names=(chain.name_map.parameters, fnames) )
                
                for f in param_dist_properties
                    for p in cparams
                        thr_pdist_prop[c][p,getFuncName(f)] = getindex(S,p,getFuncName(f))
                    end
                end           
            end

            #Operations performed on the posterior chain
            if(length(posterior_operations) > 0)
                for f in posterior_operations
                    thr_post_op[c,getFuncName(f)] = getFunc(f)(chain, rng)
                end
            end        

            #Optional returning of full chain
            if(output_chains); chains[c] = chain; end
            
        end #thread loop
        off += bs
    end
   
    #reorder thr_pdist_prop
    thr_pdist_prop_reord = nothing
    if(length(param_dist_properties) > 0)
        #check all chains agree on params list
        if(length(thr_pdist_prop) == 0)
            throw(string("thr_pdist_prop size is zero but param_dist_properties has length ",length(param_dist_properties)))
        end
        if(thr_pdist_prop === nothing)
            throw(string("thr_pdist_prop is nothing"))
        end
            
        pkeys=names(thr_pdist_prop[1],1)
        for c in 2:n
            if names(thr_pdist_prop[c],1) != pkeys; error("Name mismatch, got ", names(thr_pdist_prop[c],1), " expect ", pkeys); end
        end        
        fkeys = [getFuncName(f) for f in param_dist_properties]
        
        thr_pdist_prop_reord = NamedArray( Array{Any,3}(undef, n, length(pkeys), length(param_dist_properties)); names=( [string(i) for i in 1:n], pkeys, fkeys ) )
        for c in 1:n
            thr_pdist_prop_reord[c, :, :] = thr_pdist_prop[c][:,:]
        end        
    end
    
    return (thr_dist_prop, thr_pdist_prop_reord, thr_post_op, chains)   
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

#dist: single-argument function that takes the Turing model inputs (e.g. controls and observations); this is expected to wrap a Turing model
#N_chain : the number of chains
#input_sampler: single-argument function returning the Turing model inputs for a given chain index
#dist_properties: a list of functions that are applied for each chain to the distribution of model return values (obtained via generated_quantities)
#param_dist_properties: a list of functions that are applied over the posterior samples of each parameter
#posterior_operations : a list of functions that are applied to the output posterior chains. Function signature should be Function(::Chains, ::AbstractRNG)
#base_seed: a RNG is seeded for each chain as base_seed + chain_idx
#initialize_MAP: use a maximum a priori sampler to initialize the model parameters
#thread_block_size: limit the number of threads used in parallel on a given processor, up to the maximum threads allocated at startup. A value <=0 (default) will use all available threads.
#max_procs : limit the number of processors used up the maximum. A value <=0 (default) will use all available processors
function simulate_inference(dist, N_chain, input_sampler;
                            chain_length=1000, mcmc_sampler=NUTS(0.65),
                            dist_properties=[var], param_dist_properties = nothing, posterior_operations = nothing,
                            output_chains::Bool=false, base_seed::Int64=global_base_seed[], initialize_MAP::Bool=false, thread_block_size::Int64=-1,
                            max_procs::Int64=-1)
    
    if(dist_properties === nothing); dist_properties = Array{Function,1}(); end
    if(param_dist_properties === nothing); param_dist_properties = Array{Function,1}(); end
    if(posterior_operations === nothing); posterior_operations = Array{Function,1}(); end
    
    if( !output_chains && length(dist_properties) == 0 && length(param_dist_properties) == 0 && length(posterior_operations) == 0);
        error("Must provide at least one function if not outputing raw chains");
    end

    nproc = nprocs()
    if(max_procs > 0)
        nproc = max_procs
    end
    
    work = divide_work(N_chain, nproc)

    println("Dividing ",N_chain, " chains over ",nproc, " processors")
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
                inputs = Array{Any}(undef, work[i]) #generate the y_samples for the process here so they can be automatically copied to the remote process
                for c in 1:work[i]
                    inputs[c] = input_sampler(off + c)
                end
                wf[i] = @spawnat i worker_func(dist, inputs, off, chain_length, base_seed,
                                               mcmc_sampler, dist_properties, param_dist_properties, posterior_operations, output_chains, initialize_MAP, thread_block_size)
                off += work[i]
            end
        end
        if work[1] > 0 #main process
            inputs = Array{Any}(undef, work[1])
            for c in 1:work[1]
                inputs[c] = input_sampler(c)
            end
            wf[1] = @spawnat 1 worker_func(dist, inputs, 0, chain_length, base_seed,
                                           mcmc_sampler, dist_properties, param_dist_properties, posterior_operations, output_chains, initialize_MAP, thread_block_size)
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
        if(work[1] != N_chain); error("Work division error for single process"); end
        inputs = [ input_sampler(c) for c in 1:N_chain ]
        results[1] = worker_func(dist, inputs, 0, chain_length, base_seed,
                                 mcmc_sampler, dist_properties, param_dist_properties, posterior_operations, output_chains, initialize_MAP, thread_block_size)
    end

    #Get the list of params so we can initialize output (only needed if using param_dist_properties)
    param_list = Vector{Symbol}()
    if(length(param_dist_properties) > 0)
        if(work[1] == 0); error("Expect first proc to have work"); end
        param_list = names(results[1][2],2)
        println("Model parameters: ", param_list)
    end

    #Initialize output
    out = InferenceResult_(dist_properties, param_dist_properties, posterior_operations, param_list, N_chain, output_chains, base_seed)
    
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

            #Extract posterior_derived_quants
            if(length(posterior_operations) > 0)
                rf = results[i][3]
                if(size(rf,1) != work[i]); error("Unexpected amount of work (posterior_derived_quants)!"); end
                if(size(rf,2) != length(posterior_operations)); error("Unexpected amount of funcs (posterior_derived_quants)!"); end
                if(names(out.posterior_derived_quants,2) != names(rf,2)); error("Function names mismatch (posterior_derived_quants)!"); end
                out.posterior_derived_quants[off+1:off+work[i], :, :] = rf[:,:]
            end
            
            #Extract chains
            if(output_chains)
                rf = results[i][4]
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
    global_base_seed[] += N_chain
    
    return out
end

#dist: two-argument function taking  1) the sample observation and 2) the driver (same over all samples)
#y_samples: 2-d array with samples in columns
function simulate_inference(dist, driver, y_samples::AbstractMatrix{T}; chain_length=1000, mcmc_sampler=NUTS(0.65), dist_properties=[var], param_dist_properties=nothing, posterior_operations=nothing, output_chains::Bool=false, base_seed::Int64=global_base_seed[], thread_block_size::Int64=-1 ) where T<:Number
    y_sampler(i) = (y_samples[:,i], driver)
    dwrp(inputs) = dist(inputs...)

    N_samp = size(y_samples,2) #number of columns
    simulate_inference(dwrp, N_samp, y_sampler; chain_length=chain_length, mcmc_sampler=mcmc_sampler, dist_properties=dist_properties, param_dist_properties=param_dist_properties, posterior_operations=posterior_operations, output_chains=output_chains, base_seed=base_seed, thread_block_size=thread_block_size)
end
