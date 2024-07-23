mutable struct InferenceResult
    dist_properties::Dict{String,Array{Any,1}} #results of applying each of the "dist_properties" functions to each chain separately
    avg_dist_properties::Dict{String,Any} #results of averaging each of the "dist_properties" over all chains

    param_dist_properties::Dict{String,Dict{String,Array{Any,1}} } #results of applying each of the "param_dist_properties" functions to each chain separately
    avg_param_dist_properties::Dict{String,Dict{String,Any} } #results of averaging each of the "param_dist_properties" over all chains
end

strf(f) = String(nameof(f))

InferenceResult(dist_properties::Array{F,1}, param_dist_properties::Array{G,1}, param_list::Vector{String}, n::Integer) where {F<:Function,G<:Function} = InferenceResult(
    Dict{String,Array{Any,1}}( strf(f) => Array{Any,1}(undef,n) for f in dist_properties ) , #dist_properties
    Dict{String,Any}( strf(f) => nothing for f in dist_properties ) ,                          #avg_dist_properties
##
    Dict{String,Dict{String, Array{Any,1}}  }( strf(f) => Dict{String, Array{Any,1}}( p => Array{Any,1}(undef,n) for p in param_list )
                                               for f in param_dist_properties ),        #param_dist_properties
    Dict{String,Dict{String, Any} }( strf(f) => Dict{String, Any}( p => nothing for p in param_list )
                                     for f in param_dist_properties )                    #avg_param_dist_properties
)

#Run n chains on the worker, where n is length(y_samples)
function worker_func(dist, driver, y_samples, chain_length, mcmc_sampler, dist_properties, param_dist_properties, param_list::Vector{String})
    n = length(y_samples)
    out = (  Dict{String, Array{Any,1}}( strf(f) => Array{Any,1}(undef, n) for f in dist_properties ),
             Dict{String,Dict{String, Array{Any,1}} }( strf(f) => Dict{String,Array{Any,1}}( p => Array{Any,1}(undef, n) for p in param_list) for f in param_dist_properties )
             )
    @threads for c in 1:n
        y = y_samples[c]
        chain = sample(dist(y,driver), mcmc_sampler, chain_length)

        #Properties of model return distribution
        if(length(dist_properties) > 0)        
            Q = generated_quantities(dist(y,driver), Turing.MCMCChains.get_sections(chain, :parameters))
            for f in dist_properties
                out[1][strf(f)][c] = f(Q)
            end
        end

        #Properties of parameter distributions
        if(length(param_dist_properties) > 0)
            S = summarize(chain, param_dist_properties...)
            for f in param_dist_properties
                for p in param_list
                    out[2][strf(f)][p][c] = getindex(S,Symbol(p),Symbol(strf(f)))
                end
            end           
        end        
        
    end
    return out
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


#dist: two-argument function that takes the y-sample and the driver; this is expected to wrap a Turing model
#dist_properties: a list of functions that are applied for each chain to the distribution of model return values (obtained via generated_quantities)
function simulate_inference(dist, driver, N_samp, y_sampler; chain_length=1000, mcmc_sampler=NUTS(0.65), dist_properties::Array{F,1}=[var], param_dist_properties::Array{G,1} = Array{Function,1}() )  where {F<:Function,G<:Function}
    if(length(dist_properties) == 0 && length(param_dist_properties) == 0); error("Must provide at least one function"); end

    #Get list of parameters
    param_list::Vector{String} = [ String(s) for s in DynamicPPL.syms(DynamicPPL.VarInfo(dist(y_sampler(1),driver))) ]
    
    out = InferenceResult(dist_properties, param_dist_properties, param_list, N_samp)

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
                wf[i] = @spawnat i worker_func(dist, driver, y_samples, chain_length, mcmc_sampler, dist_properties, param_dist_properties, param_list)
                off += work[i]
            end
        end
        if work[1] > 0 #main process
            y_samples = Array{Any}(undef, work[1])
            for c in 1:work[1]
                y_samples[c] = y_sampler(c)
            end
            wf[1] = @spawnat 1 worker_func(dist, driver, y_samples, chain_length, mcmc_sampler, dist_properties, param_dist_properties, param_list)
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
        results[1] = worker_func(dist,driver, y_samples, chain_length, mcmc_sampler, dist_properties, param_dist_properties, param_list)
    end
        
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
                
                    out.dist_properties[fkey][off+1:off+work[i]] = rf[:]
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
                
                        out.param_dist_properties[fkey][p][off+1:off+work[i]] = rf[:]
                    end
                end
            end
                
            off += work[i]
        end
    end

    #Averages for dist_properties
    for fkey in dfkeys
        out.avg_dist_properties[fkey] = mean(out.dist_properties[fkey])
    end

    #Averages for param_dist_properties
    for fkey in pfkeys
        for p in param_list
            out.avg_param_dist_properties[fkey][p] = mean(out.param_dist_properties[fkey][p])
        end
    end
    
    return out
end

#y_samples: 2-d array with samples in columns
function simulate_inference(dist, driver, y_samples::AbstractMatrix{T}; chain_length=1000, mcmc_sampler=NUTS(0.65), dist_properties::Array{F,1}=[var], param_dist_properties::Array{G,1} = Array{Function,1}() ) where {T <: Number, F <: Function, G <: Function}
    y_sampler(i) = y_samples[:,i]
    N_samp = size(y_samples,2) #number of columns
    simulate_inference(dist, driver, N_samp, y_sampler; chain_length=chain_length, mcmc_sampler=mcmc_sampler, dist_properties=dist_properties, param_dist_properties=param_dist_properties)
end
