mutable struct InferenceResult
    dist_properties::Dict{String,Array{Any,1}} #results of applying each of the "dist_properties" functions to each chain separately
    avg_dist_properties::Dict{String,Any} #results of averaging each of the "dist_properties" over all chains
end

strf(f) = String(nameof(f))

InferenceResult(dist_properties::Array{F,1}, n::Integer) where F<:Function = InferenceResult(
    Dict{String,Array{Any,1}}( strf(f) => Array{Any,1}(undef,n) for f in dist_properties ),
    Dict{String,Any}( strf(f) => nothing for f in dist_properties ) )

#Run n chains on the worker, where n is length(y_samples)
function worker_func(dist, driver, y_samples, chain_length, mcmc_sampler, dist_properties)
    n = length(y_samples)
    out = Dict{String, Array{Any,1}}( strf(f) => Array{Any,1}(undef, n) for f in dist_properties )
    @threads for c in 1:n
        y = y_samples[c]
        chain = sample(dist(y,driver), mcmc_sampler, chain_length)
        Q = generated_quantities(dist(y,driver), Turing.MCMCChains.get_sections(chain, :parameters))
        for f in dist_properties
            out[strf(f)][c] = f(Q)
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
function simulate_inference(dist, driver, N_samp, y_sampler; chain_length=1000, mcmc_sampler=NUTS(0.65), dist_properties::Array{F,1}=[var]) where F<:Function
    #each y sample is different random params + random noise
    #p(y) = ∫ p(y,theta) dtheta = ∫p(y|theta) p(theta) dtheta 
    #draw from p(theta)
    #then from that draw y
    out = InferenceResult(dist_properties, N_samp)

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
    
    wf =Array{Any,1}(undef, nproc) #handles for async processes
    
    #Spawn work on main process last!
    off = work[1]
    for i in 2:nproc
        if work[i] > 0
            y_samples = Array{Any}(undef, work[i]) #generate the y_samples for the process here so they can be automatically copied to the remote process
            for c in 1:work[i]
                y_samples[c] = y_sampler(off + c)
            end
            wf[i] = @spawnat i worker_func(dist, driver, y_samples, chain_length, mcmc_sampler, dist_properties)
            off += work[i]
        end
    end
    if work[1] > 0 #main process
        y_samples = Array{Any}(undef, work[1])
        for c in 1:work[1]
            y_samples[c] = y_sampler(c)
        end
        wf[1] = @spawnat 1 worker_func(dist, driver, y_samples, chain_length, mcmc_sampler, dist_properties)
    end

    #Fetch results after main process has finished
    fkeys = [strf(f) for f in dist_properties]
    off = 0
    for i in 1:nprocs()
        if work[i] > 0
            r = fetch(wf[i])
            if isa(r,Exception)
                error(r)
            end
                        
            for fkey in fkeys
                rf = r[fkey]
                
                if length(rf) != work[i]
                    error("Unexpected amount of work!")
                end
                
                out.dist_properties[fkey][off+1:off+work[i]] = rf[:]
            end
                
            off += work[i]
        end
    end
    #show(stdout,"text/plain", out.dist_properties)
    #println("")

    for fkey in fkeys
        if length(out.dist_properties[fkey]) != N_samp
            error("Output size does not match N_samp for key ",fkey)
        end
        out.avg_dist_properties[fkey] = mean(out.dist_properties[fkey])
    end
    
    return out
end

#y_samples: 2-d array with samples in columns
function simulate_inference(dist, driver, y_samples::AbstractMatrix{T}; chain_length=1000, mcmc_sampler=NUTS(0.65), dist_properties::Array{F,1}=[var]) where {T <: Number, F <: Function}
    y_sampler(i) = y_samples[:,i]
    N_samp = size(y_samples,2) #number of columns
    simulate_inference(dist, driver, N_samp, y_sampler; chain_length=chain_length, mcmc_sampler=mcmc_sampler, dist_properties=dist_properties)
end
