# code used for the simulations reported in Sections 3-6

using PlotlyJS
using DataFrames
using KernelDensity
using SimpleANOVA
using HypothesisTests
using EffectSizes
using GLM
import ColorSchemes.viridis

using Distributed
addprocs(...) # for parallel processing, however many cores you may want to make available for that

@everywhere begin
    using StatsBase
    using Random
    using Distributions
    using SharedArrays
end

PlotlyJS.templates.default = "gridon"

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
end

@everywhere opinions(pop::Array{Agent,1}) = Float64[ pop[i].op for i in eachindex(pop) ]

@everywhere function peers(pop::Array{Agent,1})
   pc = filter(i -> typeof(i) == Carnapian, pop)
   return Bool[ abs(pc[j].op - pop[i].op) ≤ pc[j].ϵ for i in eachindex(pop), j in eachindex(pc) ] # the peers for agent j are in the j-th column (we calculate peers also for meta-learners, although we won't do anything with those data)
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
        pop[i].sc += 1 - (data[i, t] - pop[i].op)^2 # Brier loss
        wgts = [ max(pop[j].sc - pop[i].sc, eps()) for j in eachindex(pop) if typeof(pop[j]) == Carnapian] # regrets
        if typeof(pop[i]) == Carnapian
            pop[i].op = pop[i].α*((dt[i] + (pop[i].λ/2))/(t + pop[i].λ)) + (1 - pop[i].α)*soc[i]
        elseif typeof(pop[i]) == Meritocrat
            pop[i].op = pop[i].α*((dt[i] + (pop[i].λ/2))/(t + pop[i].λ)) + (1 - pop[i].α)*mean([ pop[j].op for j in eachindex(pop) if typeof(pop[j]) == Carnapian ], weights([ pop[j].sc for j in eachindex(pop) if typeof(pop[j]) == Carnapian ]))
        else
            pop[i].op = sum(wgts) == 0 ? pop[last(findmin([ pop[j].sc for j in eachindex(pop) if typeof(pop[j]) == Carnapian ]))].op : mean([ pop[j].op for j in eachindex(pop) if typeof(pop[j]) == Carnapian ], weights(wgts))
        end
    end
end

@everywhere function run_model(ϵ_inf::Float64, ϵ_sup::Float64, # upper and lower bound for ϵ
                               α_inf::Float64, α_sup::Float64, # same for α
                               λ_inf::Float64, λ_sup::Float64, # same for λ
                               bias::Float64, # bias of coin
                               numb_carnap::Int, numb_meta::Int, numb_mixed::Int, # numbers of different types of agents
                               numb_updates::Int) # number of updates
    numb_agents = numb_carnap + numb_meta + numb_mixed
    data = generate_data(bias, numb_updates, numb_agents)
    rnd_eps() = ϵ_inf < ϵ_sup ? rand(Uniform(ϵ_inf, ϵ_sup)) : ϵ_inf
    rnd_alp() = α_inf < α_sup ? rand(Uniform(α_inf, α_sup)) : α_inf
    rnd_lmb() = λ_inf < λ_sup ? rand(Uniform(λ_inf, λ_sup)) : λ_inf
    pc = [ Carnapian(rnd_eps(), rnd_alp(), rnd_lmb(), rand(), .0) for _ in 1:numb_carnap ]
    pm = [ MetaLearner(rand(), .0) for _ in 1:numb_meta ]
    px = [ Meritocrat(rnd_alp(), rnd_lmb(), rand(), .0) for _ in 1:numb_mixed ]
    pp = vcat(pc, pm, px)
    res = zeros(numb_agents, numb_updates + 1)
    res[:, 1] = opinions(pp)
    @inbounds for i in 1:numb_updates
        update!(pp, data, i)
        res[:, i + 1] = opinions(pp)
    end
    return res
end

const n_carnap = 50
const n_meta = 10
const n_mixed = 10
@time res = run_model(.1, .1, .1, .1, 2., 2., .7, n_carnap, n_meta, n_mixed, 10000)

traces1 = [ scatter(;x=1:size(res, 2), y=res[i, :], mode="lines", showlegend=false, line_color=:midnightblue) 
            for i in 1:n_carnap ]
traces2 = [ scatter(;x=1:size(res, 2), y=res[i, :], mode="lines", showlegend=false, line_color=:indianred) 
            for i in n_carnap + 1:n_carnap + n_meta ]
traces3 = [ scatter(;x=1:size(res, 2), y=res[i, :], mode="lines", showlegend=false, line_color=:mediumseagreen) 
            for i in n_carnap + n_meta + 1:n_carnap + n_meta + n_mixed ]
traces4 = scatter(;x=1:size(res, 2), y=res[1, :], mode="lines", line_color=:midnightblue, name="Carnapians")
traces5 = scatter(;x=1:size(res, 2), y=res[n_carnap + 1, :], mode="lines", line_color=:indianred, name="Meta-inductivists")
traces6 = scatter(;x=1:size(res, 2), y=res[n_carnap + n_meta + 1, :], mode="lines", line_color=:mediumseagreen, name="Meritocrats")
data = vcat(traces1, traces2, traces3, traces4, traces5, traces6)
layout = Layout(font=attr(size=14), width=715, height=475, 
                xaxis=attr(showgrid=false, title="Update", type="log", dtick=1, exponentformat="power"), 
                yaxis=attr(showgrid=true, title="Estimate", range=[0, 1]), margin=attr(l=-1, r=-1, t=-1, b=-1),
                title="\u03b1<sub>i</sub> = 0.1; \u03b5<sub>i</sub> = 0.1; \u03bb<sub>i</sub> = 2", shapes=[hline(.7, line_color="grey", line_dash="dashdot")],
                annotations=[(x=4.001, y=.77, text="\u03c4", showarrow=false, font=Dict(size=>16))])
p = plot(data, layout)

@everywhere function run_single(numb_carn::Int, numb_meta::Int, numb_mixed::Int)
    tau = rand()
    res = run_model(.0, 1., .0, 1., .0, 20., tau, numb_carn, numb_meta, numb_mixed, 100)
    scrs = (res .- tau).^2
    ssc = sum(scrs, dims=2)
    return (mean(ssc[1:numb_carn]), mean(ssc[numb_carn + 1:numb_carn + numb_meta]), mean(ssc[numb_carn + numb_meta + 1:numb_carn + numb_meta + numb_mixed]))
end

@time mbs = pmap(_->run_single(50, 50, 0), 1:100)

data = hcat(first.(mbs), getindex.(mbs, 2), last.(mbs))
anova(data)
# pairwise comparisons (1: Carnapians; 2: meta-inductivists; 3: meritocrats)
EqualVarianceTTest(data[:, 1], data[:, 2])
EqualVarianceTTest(data[:, 1], data[:, 3])
EqualVarianceTTest(data[:, 2], data[:, 3])

# meta-inductivists
EqualVarianceTTest(data[:, 1], data[:, 2])
CohenD(data[:, 1], data[:, 2]) 
mean_and_std.([first.(mbs), getindex.(mbs, 2), last.(mbs)]) # Carnapians: 1.11 (± 0.62); meta-inductivists: 0.38 (± 0.20)
# meritocrats
EqualVarianceTTest(data[:, 1], data[:, 3]) 
CohenD(data[:, 1], data[:, 3]) 
mean_and_std.([first.(mbs), getindex.(mbs, 2), last.(mbs)])

trace1 = box(;y=first.(mbs), marker_color=:midnightblue, name="Carnapians")
trace2 = box(;y=last.(mbs), marker_color=:mediumseagreen, name="Meritocrats")
trace3 = box(;y=getindex.(mbs, 2), marker_color=:indianred, name="Meta-inductivists")
layout = Layout(font=attr(size=16), width=800, height=580, showlegend=false, yaxis=attr(title="Total Brier loss", range=[-0.025, 3.5]),
                title="\u03b1<sub>i</sub> ~ U(0, 1); \u03b5<sub>i</sub> ~ U(0, 1); \u03bb<sub>i</sub> ~ U(0, 20)")
p = plot([trace1, trace2, trace3], layout)

@everywhere function run_single_fixed(p::Float64, numb_meta::Int, numb_mixed::Int)
    tau = rand()
    res = run_model(p, p, .1, .1, 2., 2., tau, 50, numb_meta, numb_mixed, 100)
    scrs = (res .- tau).^2
    ssc = sum(scrs, dims=2)
    return (mean(ssc[1:50]), mean(ssc[51:50 + numb_meta]), mean(ssc[51 + numb_meta:50 + numb_meta + numb_mixed]))
end

run_single_fixed_sim(x, y) = pmap(_->run_single_fixed(x, y, 0), 1:100)

acc_res = [ run_single_fixed_sim(x, y) for x in .0:.01:.5, y in 0:50 ]

mns = [ mean(first.(acc_res[i, j])) for i in 1:51, j in 1:51 ]

trace = heatmap(x=collect(0:0.1:5), y=collect(0:50), z=rotl90(mns), zmin=.5, zmax=3., 
    colorscale="Viridis",
    colorbar=attr(;thickness=18, len=0.8, lenmode="fraction", outlinewidth=0, title="Average<br>Brier loss", titleside="top", titlefont=attr(;size=13)))
layout = Layout(margin=attr(t=20, r=75, l=75, b=60), autosize=false, width=560, height=490, font_size=13, xaxis=attr(title="\u03bb", zeroline=false, showgrid=false), yaxis=attr(title="Meritocrats", zeroline=false, showgrid=false),
    annotations=[(x=2.5, y=53, text="\u03b1<sub>i</sub> ~ U(0, 1); \u03b5<sub>i</sub> ~ U(0, 1)", showarrow=false, font=Dict(size=>17))])
p = Plot(trace, layout)

## evolutionary computing

# start population

const numb_carnapians = 50
const numb_metainductivists = 50

function populate()
    pc = [ Carnapian(rand(), rand(), rand(Uniform(0, 20)), rand(), .0) for _ in 1:numb_carnapians ]
    pm = [ MetaLearner(rand(), .0) for _ in 1:numb_metainductivists ]
    return vcat(pc, pm)
end

function run_model_evo(pop::Array{Agent,1}; numb_updates=100, numb_test=25)
    ssc = SharedArray{Float64,2}(numb_carnapians + numb_metainductivists, numb_test)
    @inbounds @sync @distributed for t in 1:numb_test
        τ = rand()
        data = generate_data(τ, numb_updates, numb_carnapians + numb_metainductivists)
        res = zeros(numb_carnapians + numb_metainductivists, numb_updates + 1)
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
        bst = shuffle(getindex.(findall(i->i<=median(out), out), 1))[1:div(numb_carnapians + numb_metainductivists, 2)]
        parent_pop = store_pops[end][bst]
        f_c = filter(i -> typeof(i) == Carnapian, parent_pop)
        f_m = filter(i -> typeof(i) == MetaLearner, parent_pop)
        child_pop = Agent[]
        if isempty(f_c)
            break
        else
            for i in 1:length(f_c)
                a1, a2 = sample(f_c, 2)
                push!(child_pop, Carnapian((a1.ϵ + a2.ϵ)/2, (a1.α + a2.α)/2, (a1.λ + a2.λ)/2, rand(), .0))
            end
        end
        if !isempty(f_m)
            for i in 1:length(f_m)
                push!(child_pop, MetaLearner(rand(), .0))
            end
        end
        new_pop = vcat(parent_pop, child_pop)
        f_cc = filter(i -> typeof(i) == Carnapian, new_pop)
        f_mm = filter(i -> typeof(i) == MetaLearner, new_pop)
        new_pop = vcat(f_cc, f_mm)
        push!(store_pops, new_pop)
        g += 1
    end
    return store_pops
end

ea_res = [ map(x -> length(filter(i -> typeof(i) == Carnapian, x)), evolutionary_algorithm(50)) for _ in 1:100 ]

length.(ea_res[length.(ea_res) .< 51]) |> length
length.(ea_res[length.(ea_res) .< 51]) |> mean_and_std

last.(ea_res[length.(ea_res) .== 51]) |> mean_and_std

# repeat the foregoing but with meritocrats instead of meta-inductivists

function populate()
    pc = [ Carnapian(rand(), rand(), rand(Uniform(0, 20)), rand(), .0) for _ in 1:50 ]
    px = [ Meritocrat(rand(), rand(Uniform(0, 20)), rand(), .0) for _ in 1:50 ]
    return vcat(pc, px)
end

function run_model_evo(pop::Array{Agent,1}; numb_updates=100, numb_test=25)
    ssc = SharedArray{Float64,2}(100, numb_test)
    @inbounds @sync @distributed for t in 1:numb_test
        τ = rand()
        data = generate_data(τ, numb_updates, 100)
        res = zeros(100, numb_updates + 1)
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
        bst = shuffle(getindex.(findall(i->i<=median(out), out), 1))[1:50]
        parent_pop = store_pops[end][bst]
        f_c = filter(i -> typeof(i) == Carnapian, parent_pop)
        f_x = filter(i -> typeof(i) == Meritocrat, parent_pop)
        child_pop = Agent[]
        if isempty(f_c)
            break
        else
            for i in 1:length(f_c)
                a1, a2 = sample(f_c, 2)
                push!(child_pop, Carnapian((a1.ϵ + a2.ϵ)/2, (a1.α + a2.α)/2, (a1.λ + a2.λ)/2, rand(), .0))
            end
        end
        if !isempty(f_x)
            for i in 1:length(f_x)
                a1, a2 = sample(f_x, 2)
                push!(child_pop, Meritocrat((a1.α + a2.α)/2, (a1.λ + a2.λ)/2, rand(), .0))
            end
        end
        new_pop = vcat(parent_pop, child_pop)
        f_cc = filter(i -> typeof(i) == Carnapian, new_pop)
        f_xx = filter(i -> typeof(i) == Meritocrat, new_pop)
        new_pop = vcat(f_cc, f_xx)
        push!(store_pops, new_pop)
        g += 1
    end
    return store_pops
end

out = [ map(x -> length(filter(i -> typeof(i) == Carnapian, x)), evolutionary_algorithm(25)) for _ in 1:100 ]

length.(out) |> countmap

[ out[i][end] for i in 1:100 ] |> countmap

# evolutionary computing: start with 50 agents of each type, then select on the basis of lowest total brier score, averaged over 10 runs of 1000 updates, make new population, and repeat

# start population
function populate()
    pc = [ Carnapian(rand(), rand(), rand(Uniform(0, 20)), rand(), .0) for _ in 1:50 ]
    pm = [ MetaLearner(rand(), .0) for _ in 1:50 ]
    px = [ Meritocrat(rand(), rand(Uniform(0, 20)), rand(), .0) for _ in 1:50 ]
    return vcat(pc, pm, px)
end

function run_model_evo(pop::Array{Agent,1}; numb_updates=100, numb_test=25)
    ssc = SharedArray{Float64,2}(150, numb_test)
    @inbounds @sync @distributed for t in 1:numb_test
        τ = rand()
        data = generate_data(τ, numb_updates, 150)
        res = zeros(150, numb_updates + 1)
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
        bst = shuffle(getindex.(findall(i->i<=median(out), out), 1))[1:75]
        parent_pop = store_pops[end][bst]
        f_c = filter(i -> typeof(i) == Carnapian, parent_pop)
        f_m = filter(i -> typeof(i) == MetaLearner, parent_pop)
        f_x = filter(i -> typeof(i) == Meritocrat, parent_pop)
        child_pop = Agent[]
        if isempty(f_c)
            break
        else
            for i in 1:length(f_c)
                a1, a2 = sample(f_c, 2)
                push!(child_pop, Carnapian((a1.ϵ + a2.ϵ)/2, (a1.α + a2.α)/2, (a1.λ + a2.λ)/2, rand(), .0))
            end
        end
        if !isempty(f_m)
            for i in 1:length(f_m)
                push!(child_pop, MetaLearner(rand(), .0))
            end
        end
        if !isempty(f_x)
            for i in 1:length(f_x)
                a1, a2 = sample(f_x, 2)
                push!(child_pop, Meritocrat((a1.α + a2.α)/2, (a1.λ + a2.λ)/2, rand(), .0))
            end
        end
        new_pop = vcat(parent_pop, child_pop)
        f_cc = filter(i -> typeof(i) == Carnapian, new_pop)
        f_mm = filter(i -> typeof(i) == MetaLearner, new_pop)
        f_xx = filter(i -> typeof(i) == Meritocrat, new_pop)
        new_pop = vcat(f_cc, f_mm, f_xx)
        push!(store_pops, new_pop)
        g += 1
    end
    return store_pops
end

ea_res = [ evolutionary_algorithm(50) for _ in 1:100 ]

ag_count(n) = length(filter(i->typeof(i)==Carnapian, ea_res[n][end])), length(filter(i->typeof(i)==MetaLearner, ea_res[n][end])), length(filter(i->typeof(i)==Meritocrat, ea_res[n][end]))

agc = [ ag_count(i) for i in 1:length(ea_res) ]

getindex.(agc, 2) |> mean_and_std
last.(agc) |> mean_and_std

lea = length.(ea_res)

mean_and_std(lea)

ag_count2(n) = length(filter(i->typeof(i)==Carnapian, ea_res[n][2])), length(filter(i->typeof(i)==MetaLearner, ea_res[n][2])), length(filter(i->typeof(i)==Meritocrat, ea_res[n][2]))

agc2 = [ ag_count2(i) for i in 1:length(ea_res) ]

getindex.(agc2, 2) |> mean_and_std
last.(agc2) |> mean_and_std

# variation with hidden identities (social updating always on the basis of all agents)
@everywhere function update_var!(pop::Array{Agent,1}, data::Array{Int64,2}, t::Int)
    prs = peers(pop)
    ops = opinions(pop)
    soc = sum(ops .* prs, dims=1) ./ sum(prs, dims=1)
    dt = sum(@view(data[:, 1:t]), dims=2)
    @inbounds for i in eachindex(pop)
        pop[i].sc += 1 - (data[i, t] - pop[i].op)^2
        wgts = [ max(pop[j].sc - pop[i].sc, eps()) for j in eachindex(pop) ]
        if typeof(pop[i]) == Carnapian
            pop[i].op = pop[i].α*((dt[i] + (pop[i].λ/2))/(t + pop[i].λ)) + (1 - pop[i].α)*soc[i]
        elseif typeof(pop[i]) == Meritocrat
            pop[i].op = pop[i].α*((dt[i] + (pop[i].λ/2))/(t + pop[i].λ)) + (1 - pop[i].α)*mean([ pop[j].op for j in eachindex(pop) ], weights([ pop[j].sc for j in eachindex(pop) ]))
        else
            pop[i].op = mean([ pop[j].op for j in eachindex(pop) ], weights(wgts))
        end
    end
end

@everywhere function run_model_var(ϵ_inf::Float64, ϵ_sup::Float64, # upper and lower bound for ϵ
                                   α_inf::Float64, α_sup::Float64, # same for α
                                   λ_inf::Float64, λ_sup::Float64, # same for λ
                                   bias::Float64, # bias of coin
                                   numb_carnap::Int, numb_meta::Int, numb_mixed::Int, # numbers of different types of agents
                                   numb_updates::Int) # number of updates
    numb_agents = numb_carnap + numb_meta + numb_mixed
    data = generate_data(bias, numb_updates, numb_agents)
    rnd_eps() = ϵ_inf < ϵ_sup ? rand(Uniform(ϵ_inf, ϵ_sup)) : ϵ_inf
    rnd_alp() = α_inf < α_sup ? rand(Uniform(α_inf, α_sup)) : α_inf
    rnd_lmb() = λ_inf < λ_sup ? rand(Uniform(λ_inf, λ_sup)) : λ_inf
    pc = [ Carnapian(rnd_eps(), rnd_alp(), rnd_lmb(), rand(), .0) for _ in 1:numb_carnap ]
    pm = [ MetaLearner(rand(), .0) for _ in 1:numb_meta ]
    px = [ Meritocrat(rnd_alp(), rnd_lmb(), rand(), .0) for _ in 1:numb_mixed ]
    pp = vcat(pc, pm, px)
    res = zeros(numb_agents, numb_updates + 1)
    res[:, 1] = opinions(pp)
    @inbounds for i in 1:numb_updates
        update_var!(pp, data, i)
        res[:, i + 1] = opinions(pp)
    end
    return res
end

const n_carnap = 50
const n_meta = 10
const n_mixed = 10
@time res = run_model_var(.1, .1, .1, .1, 2., 2., .7, n_carnap, n_meta, n_mixed, 10000)

traces1 = [ scatter(;x=1:size(res, 2), y=res[i, :], mode="lines", showlegend=false, line_color=:midnightblue) for i in 1:n_carnap ]
traces2 = [ scatter(;x=1:size(res, 2), y=res[i, :], mode="lines", showlegend=false, line_color=:indianred) for i in n_carnap + 1:n_carnap + n_meta ]
traces3 = [ scatter(;x=1:size(res, 2), y=res[i, :], mode="lines", showlegend=false, line_color=:mediumseagreen) for i in n_carnap + n_meta + 1:n_carnap + n_meta + n_mixed ]
traces4 = scatter(;x=1:size(res, 2), y=res[1, :], mode="lines", line_color=:midnightblue, name="Carnapians")
traces5 = scatter(;x=1:size(res, 2), y=res[n_carnap + 1, :], mode="lines", line_color=:indianred, name="Meta-inductivists")
traces6 = scatter(;x=1:size(res, 2), y=res[n_carnap + n_meta + 1, :], mode="lines", line_color=:mediumseagreen, name="Meritocrats")
data = vcat(traces1, traces2, traces3, traces4, traces5, traces6)
#data = vcat(traces2, traces3, traces5, traces6)
layout = Layout(font=attr(size=14), width=715, height=475, xaxis=attr(title="Update", type="log", showgrid=false, dtick=1, exponentformat="power"), yaxis=attr(title="Estimate", range=[0, 1]), margin=attr(l=-1, r=-1, t=-1, b=-1),
                title="\u03b1<sub>i</sub> = 0.1; \u03b5<sub>i</sub> = 0.1; \u03bb<sub>i</sub> = 2", shapes=[hline(.7, line_color="grey", line_dash="dashdot")],
                annotations=[(x=4.001, y=.77, text="\u03c4", showarrow=false, font=Dict(size=>16))])
p = plot(data, layout)

@everywhere function run_single_var(numb_carn::Int, numb_meta::Int, numb_mixed::Int)
    tau = rand()
    res = run_model_var(.0, 1., .0, 1., .0, 20., tau, numb_carn, numb_meta, numb_mixed, 100)
    scrs = (res .- tau).^2
    ssc = sum(scrs, dims=2)
    return (mean(ssc[1:numb_carn]), mean(ssc[numb_carn + 1:numb_carn + numb_meta]), mean(ssc[numb_carn + numb_meta + 1:numb_carn + numb_meta + numb_mixed]))
end

mbs = pmap(_->run_single_var(50, 50, 50), 1:100)

trace1 = box(;y=first.(mbs), marker_color=:midnightblue, name="Carnapians")
trace2 = box(;y=last.(mbs), marker_color=:mediumseagreen, name="Meritocrats")
trace3 = box(;y=getindex.(mbs, 2), marker_color=:indianred, name="Meta-inductivists")
layout = Layout(font=attr(size=16), width=800, height=580, showlegend=false, yaxis=attr(title="Total Brier loss", range=[-0.025, 3.5]),
                title="\u03b1<sub>i</sub> = 0.1; \u03b5<sub>i</sub> = 0.1; \u03bb<sub>i</sub> = 2")
p = plot([trace1, trace2, trace3], layout)

data = hcat(first.(mbs), getindex.(mbs, 2), last.(mbs))
anova(data)

# pairwise comparisons (1: Carnapians; 2: meta-inductivists; 3: meritocrats)
EqualVarianceTTest(data[:, 1], data[:, 2])
EqualVarianceTTest(data[:, 1], data[:, 3])
EqualVarianceTTest(data[:, 2], data[:, 3])
mean_and_std.([first.(mbs), getindex.(mbs, 2), last.(mbs)])

function populate()
    pc = [ Carnapian(rand(), rand(), rand(Uniform(0, 20)), rand(), eps()) for _ in 1:50 ]
    pm = [ MetaLearner(rand(), eps()) for _ in 1:50 ]
    px = [ Meritocrat(rand(), rand(Uniform(0, 20)), rand(), eps()) for _ in 1:50 ]
    return vcat(pc, pm, px)
end

function run_model_evo_var(pop::Array{Agent,1}; numb_updates=100, numb_test=25)
    ssc = SharedArray{Float64,2}(150, numb_test)
    @inbounds @sync @distributed for t in 1:numb_test
        τ = rand()
        data = generate_data(τ, numb_updates, 150)
        res = zeros(150, numb_updates + 1)
        res[:, 1] = opinions(pop)
        @inbounds for i in 1:numb_updates
            update_var!(pop, data, i)
            res[:, i + 1] = opinions(pop)
        end
        scrs = (res .- τ).^2
        ssc[:, t] = sum(scrs, dims=2)
    end    
    return mean(ssc, dims=2)
end

out = run_model_evo_var(populate())

function evolutionary_algorithm_var(max_numb_gens::Int)
    store_pops = Array{Agent,1}[]
    push!(store_pops, populate())
    g = 1
    while g <= max_numb_gens
        out = run_model_evo_var(store_pops[end])
        bst = shuffle(getindex.(findall(i->i<=median(out), out), 1))[1:75]
        parent_pop = store_pops[end][bst]
        f_c = filter(i -> typeof(i) == Carnapian, parent_pop)
        f_m = filter(i -> typeof(i) == MetaLearner, parent_pop)
        f_x = filter(i -> typeof(i) == Meritocrat, parent_pop)
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
        if !isempty(f_x)
            for _ in 1:length(f_x)
                a1, a2 = sample(f_x, 2)
                push!(child_pop, Meritocrat((a1.α + a2.α)/2, (a1.λ + a2.λ)/2, rand(), .0))
            end
        end
        new_pop = vcat(parent_pop, child_pop)
        f_cc = filter(i -> typeof(i) == Carnapian, new_pop)
        f_mm = filter(i -> typeof(i) == MetaLearner, new_pop)
        f_xx = filter(i -> typeof(i) == Meritocrat, new_pop)
        new_pop = vcat(f_cc, f_mm, f_xx)
        push!(store_pops, new_pop)
        g += 1
        if (isempty(f_c) && isempty(f_m)) || (isempty(f_c) && isempty(f_x)) || (isempty(f_m) && isempty(f_x))
            break
        end 
    end
    return store_pops
end

function ea_run(numb_gen::Int)
    res = evolutionary_algorithm_var(numb_gen)
    a = [ length(filter(i -> typeof(i) == Carnapian, res[j])) for j in 1:length(res) ]
    b = [ length(filter(i -> typeof(i) == MetaLearner, res[j])) for j in 1:length(res) ]
    c = [ length(filter(i -> typeof(i) == Meritocrat, res[j])) for j in 1:length(res) ]
    return hcat(a, b, c)
end

full_mixed_res = [ ea_run(100) for _ in 1:100 ]

# for how many generations did the simulations run?
ngen = div.(length.(full_mixed_res), 3)
mean_and_std(ngen)
countmap(ngen)[:101]

p = plot(histogram(x=ngen, xbins_size=5, marker_color=:midnightblue, opacity=.7), 
         Layout(width=535, height=380, xaxis_title="Number of generations until convergence", yaxis_title="Count"))

dat = full_mixed_res[1]

trace1 = bar(;x=1:101, y=dat[:, 1], name="Carnapians", marker_color=:midnightblue, opacity=.7)
trace2 = bar(;x=1:101, y=dat[:, 2], name="Meta-inductivists", marker_color=:indianred, opacity=.7)
trace3 = bar(;x=1:101, y=dat[:, 3], name="Meritocrats", marker_color=:mediumseagreen, opacity=.7)
layout = Layout(; barmode="stack", width=660, height=425, xaxis_title="Generation", yaxis_title="Number of agents")
bp = plot([trace3, trace2, trace1], layout)
savefig(bp, "full_mixed_generations59.pdf"; scale=1)

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
