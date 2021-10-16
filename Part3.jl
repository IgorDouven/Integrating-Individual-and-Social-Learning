# code used for the 'grand finale' reported toward the end of Section 7

using PlotlyJS
using DataFrames
using KernelDensity
using SimpleANOVA
using HypothesisTests
using EffectSizes
using GLM
import ColorSchemes.viridis

PlotlyJS.templates.default = "gridon"

using Distributed
addprocs(10)

@everywhere begin
    using StatsBase
    using Random
    using Distributions
    using SharedArrays
end

@everywhere begin
    abstract type Agent end

    mutable struct Carnapian <: Agent
        ϵ::Float64
        α::Float64
        λ::Float64
        op::Float64
        sc::Float64
    end

    mutable struct MetaLearner <: Agent
        op::Float64
        sc::Float64
    end

    mutable struct Meritocrat <: Agent
        α::Float64
        λ::Float64
        op::Float64
        sc::Float64
    end

    mutable struct MixedLearner <: Agent
        λ::Float64
        θ::Float64
        op::Float64
        sc::Float64
    end
end

@everywhere opinions(pop::Array{Agent,1}) = Float64[ pop[i].op for i in eachindex(pop) ]

@everywhere function peers(pop::Array{Agent,1})
   pc = filter(i -> typeof(i) == Carnapian, pop)
   return Bool[ abs(pc[j].op - pop[i].op) ≤ pc[j].ϵ for i in eachindex(pop), j in eachindex(pc) ] # the peers for agent j are in the j-th column
end

@everywhere function generate_data(bias::Float64, numb_toss::Int, numb_agents::Int)
    sim_dat = Array{Int64,2}(undef, numb_agents, numb_toss)
    @inbounds for i in 1:numb_agents
        sim_dat[i, :] = rand(Bernoulli(bias), numb_toss)
    end
    return sim_dat
end

@everywhere function update!(pop::Array{Agent,1}, data::Array{Int64,2}, t::Int)
    prs = peers(pop)
    ops = opinions(pop)
    soc = sum(ops .* prs, dims=1) ./ sum(prs, dims=1)
    dt = sum(@view(data[:, 1:t]), dims=2)
    @inbounds for i in eachindex(pop)
        pop[i].sc += 1 - (data[i, t] - pop[i].op)^2 # make Brier scores bonus points, which is easier for the `weights` function
        wgts = [ max(pop[j].sc - pop[i].sc, eps()) for j in eachindex(pop) ]
        if typeof(pop[i]) == Carnapian
            pop[i].op = pop[i].α*((dt[i] + (pop[i].λ/2))/(t + pop[i].λ)) + (1 - pop[i].α)*soc[i]
        elseif typeof(pop[i]) == Meritocrat
            pop[i].op = pop[i].α*((dt[i] + (pop[i].λ/2))/(t + pop[i].λ)) + (1 - pop[i].α)*mean([ pop[j].op for j in eachindex(pop) ], weights([ pop[j].sc for j in eachindex(pop) ]))
        elseif typeof(pop[i]) == MetaLearner
            pop[i].op = mean([ pop[j].op for j in eachindex(pop) ], weights(wgts))
        else
            if rand() > pop[i].θ
                pop[i].op = (dt[i] + (pop[i].λ/2))/(t + pop[i].λ)
            else
                pop[i].op = mean([ pop[j].op for j in eachindex(pop) ], weights([ pop[j].sc for j in eachindex(pop) ]))
            end            
        end
    end
end

@everywhere function run_model(ϵ_inf::Float64, ϵ_sup::Float64, # upper and lower bound for ϵ
                               α_inf::Float64, α_sup::Float64, # same for α
                               λ_inf::Float64, λ_sup::Float64, # same for λ
                               θ_inf::Float64, θ_sup::Float64, # same for θ
                               bias::Float64, # bias of coin
                               numb_carnap::Int, numb_meta::Int, numb_merit::Int, numb_mixed::Int, # numbers of different types of agents
                               numb_updates::Int) # number of updates
    numb_agents = numb_carnap + numb_meta + numb_merit + numb_mixed
    data = generate_data(bias, numb_updates, numb_agents)
    rnd_eps() = ϵ_inf < ϵ_sup ? rand(Uniform(ϵ_inf, ϵ_sup)) : ϵ_inf
    rnd_alp() = α_inf < α_sup ? rand(Uniform(α_inf, α_sup)) : α_inf
    rnd_lmb() = λ_inf < λ_sup ? rand(Uniform(λ_inf, λ_sup)) : λ_inf
    rnd_tht() = θ_inf < θ_sup ? rand(Uniform(θ_inf, θ_sup)) : θ_inf
    pca = [ Carnapian(rnd_eps(), rnd_alp(), rnd_lmb(), rand(), .0) for _ in 1:numb_carnap ]
    pml = [ MetaLearner(rand(), .0) for _ in 1:numb_meta ]
    pmc = [ Meritocrat(rnd_alp(), rnd_lmb(), rand(), .0) for _ in 1:numb_merit ]
    pmx = [ MixedLearner(rnd_lmb(), rnd_tht(), rand(), .0) for _ in 1:numb_mixed ] 
    pp = vcat(pca, pml, pmc, pmx)
    res = zeros(numb_agents, numb_updates + 1)
    res[:, 1] = opinions(pp)
    @inbounds for i in 1:numb_updates
        update!(pp, data, i)
        res[:, i + 1] = opinions(pp)
    end
    return res
end

function populate()
    pca = [ Carnapian(rand(), rand(), rand(Uniform(0, 20)), rand(), .0) for _ in 1:50 ]
    pml = [ MetaLearner(rand(), .0) for _ in 1:50 ]
    pmc = [ Meritocrat(rand(), rand(Uniform(0, 20)), rand(), .0) for _ in 1:50 ]
    pmx = [ MixedLearner(rand(Uniform(0, 20)), rand(), rand(), .0) for _ in 1:50 ]
    return vcat(pca, pml, pmc, pmx)
end

function run_model_evo(pop::Array{Agent,1}; numb_updates=100, numb_test=25)
    ssc = SharedArray{Float64,2}(200, numb_test)
    @inbounds @sync @distributed for t in 1:numb_test
        τ = rand()
        data = generate_data(τ, numb_updates, 200)
        res = zeros(200, numb_updates + 1)
        res[:, 1] = opinions(pop)
        @inbounds for i in 1:numb_updates
            update!(pop, data, i)
            res[:, i + 1] = opinions(pop)
        end
        scrs = (res .- τ).^2
        ssc[:, t] = sum(scrs, dims=2)
    end    
    return mean(ssc, dims=2)
end

out = run_model_evo(populate())

function evolutionary_algorithm(max_numb_gens::Int)
    store_pops = Array{Agent,1}[]
    push!(store_pops, populate())
    g = 1
    while g <= max_numb_gens
        out = run_model_evo(store_pops[end])
        bst = shuffle(getindex.(findall(i->i<=median(out), out), 1))[1:100]
        parent_pop = store_pops[end][bst]
        f_c = filter(i->typeof(i) == Carnapian, parent_pop)
        f_m = filter(i->typeof(i) == MetaLearner, parent_pop)
        f_t = filter(i->typeof(i) == Meritocrat, parent_pop)
        f_x = filter(i->typeof(i) == MixedLearner, parent_pop)
        child_pop = Agent[]
        if !isempty(f_c)
            for _ in 1:length(f_c)
                a1, a2 = sample(f_c, 2)
                push!(child_pop, Carnapian((a1.ϵ + a2.ϵ)/2, (a1.α + a2.α)/2, (a1.λ + a2.λ)/2, rand(), .0))
            end
        end
        if !isempty(f_m)
            for _ in 1:length(f_m)
                push!(child_pop, MetaLearner(rand(), .0))
            end
        end
        if !isempty(f_t)
            for _ in 1:length(f_t)
                a1, a2 = sample(f_t, 2)
                push!(child_pop, Meritocrat((a1.α + a2.α)/2, (a1.λ + a2.λ)/2, rand(), .0))
            end
        end
        if !isempty(f_x)
            for _ in 1:length(f_x)
                a1, a2 = sample(f_x, 2)
                push!(child_pop, MixedLearner((a1.λ + a2.λ)/2, (a1.θ + a2.θ)/2, rand(), .0))
            end
        end
        new_pop = vcat(parent_pop, child_pop)
        f_cc = filter(i->typeof(i) == Carnapian, new_pop)
        f_mm = filter(i->typeof(i) == MetaLearner, new_pop)
        f_mt = filter(i->typeof(i) == Meritocrat, new_pop)
        f_xx = filter(i->typeof(i) == MixedLearner, new_pop)
        new_pop = vcat(f_cc, f_mm, f_mt, f_xx)
        push!(store_pops, new_pop)
        g += 1
        if (isempty(f_c) && isempty(f_m)) && isempty(f_t) || 
                (isempty(f_c) && isempty(f_m) && isempty(f_x)) || 
                (isempty(f_c) && isempty(f_t) && isempty(f_x)) || 
                (isempty(f_m) && isempty(f_t) && isempty(f_x))
            break
        end 
    end
    return store_pops
end

function ea_run(numb_gen::Int)
    res = evolutionary_algorithm(numb_gen)
    a = [ length(filter(i->typeof(i) == Carnapian, res[j])) for j in 1:length(res) ]
    b = [ length(filter(i->typeof(i) == MetaLearner, res[j])) for j in 1:length(res) ]
    c = [ length(filter(i->typeof(i) == Meritocrat, res[j])) for j in 1:length(res) ]
    d = [ length(filter(i->typeof(i) == MixedLearner, res[j])) for j in 1:length(res) ]
    return hcat(a, b, c, d)
end

full_mixed_res = [ ea_run(100) for _ in 1:100 ]

# for how many generations did the simulations run?
ngen = div.(length.(full_mixed_res), 4)
countmap(ngen)

p = plot(histogram(x=ngen, xbins_size=5, marker_color=:midnightblue, opacity=.7), 
         Layout(width=535, height=380, xaxis_title="Number of generations until convergence", yaxis_title="Count"))

dat = full_mixed_res[2]

trace1 = bar(;x=1:101, y=dat[:, 1], name="Carnapians", marker_color=:midnightblue, opacity=.7)
trace2 = bar(;x=1:101, y=dat[:, 2], name="Meta-inductivists", marker_color=:indianred, opacity=.7)
trace3 = bar(;x=1:101, y=dat[:, 3], name="Meritocrats", marker_color=:mediumseagreen, opacity=.7)
trace4 = bar(;x=1:101, y=dat[:, 4], name="Mixed learners", marker_color=:thistle, opacity=.7)
layout = Layout(; barmode="stack", width=660, height=425, xaxis_title="Generation", yaxis_title="Number of agents")
bp = plot([trace4, trace3, trace2, trace1], layout)

# average number of agents of given type present in last generations
mean([ full_mixed_res[i][end, :] for i in 1:100 ], dims=1)
std([ full_mixed_res[i][end, :] for i in 1:100 ]) 

fmr = [ full_mixed_res[i][:, 2] for i in 1:100 ]

z = zeros(Int, 100, 101)

for j in 1:100
    for i in 1:length(fmr[j])
        z[j, i] = fmr[j][i]
    end
end

fmr_m = mean(z, dims=1)
fmr_s = std(z, dims=1)

trace1 = scatter(x=1:30, y=dropdims(fmr_m .- (fmr_s/10), dims=1)[1:30], mode="lines", showlegend=false, line_width=0)
trace2 = scatter(x=1:30, y=dropdims(fmr_m, dims=1)[1:30], showlegend=false, line_color=:midnightblue, fill="tonexty", fillcolor="rgba(25,25,112, 0.7)", mode="lines")
trace3 = scatter(x=1:30, y=dropdims(fmr_m .+ (fmr_s/10), dims=1)[1:30], showlegend=false, fill="tonexty", fillcolor="rgba(25,25,112, 0.7)", mode="lines", line_width=0)
data = [trace1, trace2, trace3]
layout = Layout(font=attr(size=12), width=715, height=475, xaxis=attr(title="Generation"), yaxis=attr(title="Number of meta-inductivists", range=[0, 80]))
p = plot(data, layout)
