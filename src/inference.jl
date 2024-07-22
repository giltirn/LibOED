mutable struct InferenceResult
    𝕍Q::Array{Any}
    𝔼𝕍Q::Any
end

InferenceResult(n::Integer) = InferenceResult(Array{Any}(undef,n), undef)

#Run n chains on the worker, where n is length(y_samples)
function worker_func(dist, driver, y_samples, chain_length, mcmc_sampler, dist_property)
    n = length(y_samples)
    out = Array{Any,1}(undef, n)
    @threads for c in 1:n
        y = y_samples[c]
        #println("Y for worker sample ",c," of ",n)
        #show(stdout,"text/plain",y)
        #println("")
        chain = sample(dist(y,driver), mcmc_sampler, chain_length)
        #println("Chain information")
        #show(stdout,"text/plain",chain)
        #println("")
        Q = generated_quantities(dist(y,driver), Turing.MCMCChains.get_sections(chain, :parameters))
        #println("Results of generated_quantities")
        #show(stdout,"text/plain",Q)
        #println("")       
        out[c] = dist_property(Q)
        #println("Distribution property on generated_quantities")
        #show(stdout,"text/plain",out[c])
        #println("")
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
function simulate_inference(dist, driver, N_samp, y_sampler; chain_length=1000, mcmc_sampler=NUTS(0.65), dist_property=var)
    #each y sample is different random params + random noise
    #p(y) = ∫ p(y,theta) dtheta = ∫p(y|theta) p(theta) dtheta 
    #draw from p(theta)
    #then from that draw y
    out = InferenceResult(N_samp)

    nproc = nprocs()
    work = divide_work(N_samp, nproc)
    wf = Array{Any}(undef, nproc)

    #Spawn work on main process last!
    off = work[1]
    for i in 2:nproc
        if work[i] > 0
            y_samples = Array{Any}(undef, work[i]) #generate the y_samples for the process here so they can be automatically copied to the remote process
            for c in 1:work[i]
                y_samples[c] = y_sampler(off + c)
            end
            wf[i] = @spawnat i worker_func(dist, driver, y_samples, chain_length, mcmc_sampler, dist_property)
            off += work[i]
        end
    end
    if work[1] > 0 #main process
        y_samples = Array{Any}(undef, work[1])
        for c in 1:work[1]
            y_samples[c] = y_sampler(c)
        end
        wf[1] = @spawnat 1 worker_func(dist, driver, y_samples, chain_length, mcmc_sampler, dist_property)
    end

    #Fetch results after main process has finished
    off = 0
    for i in 1:nprocs()
        if work[i] > 0
            r = fetch(wf[i])
            if isa(r,Exception)
                error(r)
            end
                        
            if length(r) != work[i]
                error("Unexpected amount of work!")
            end
            out.𝕍Q[off+1:off+work[i]] = r[:]
            off += work[i]
        end
    end
    show(stdout,"text/plain", out.𝕍Q)
    println("")
    if length(out.𝕍Q) != N_samp
        error("Output size does not match N_samp")
    end

    # y_samples = Array{Any}(undef, N_samp)
    # for i in 1:N_samp
    #     y_samples[i] = y_sampler(i)
    # end    
    # out.𝕍Q = worker_func(dist, driver, y_samples, chain_length, mcmc_sampler, dist_property)

    out.𝔼𝕍Q = mean(out.𝕍Q)
    
    return out
end

#y_samples: 2-d array with samples in columns
function simulate_inference(dist, driver, y_samples::AbstractMatrix{T}; chain_length=1000, mcmc_sampler=NUTS(0.65), dist_property=var) where T <: Number
    y_sampler(i) = y_samples[:,i]
    N_samp = size(y_samples,2) #number of columns
    simulate_inference(dist, driver, N_samp, y_sampler; chain_length=chain_length, mcmc_sampler=mcmc_sampler, dist_property=dist_property)
end
