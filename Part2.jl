# code used for most of the simulations reported in Section 7

using HypothesisTests
using PlotlyJS
using SimpleANOVA
import ColorSchemes.viridis

PlotlyJS.templates.default = "gridon"

using Distributed
addprocs(...)

@everywhere begin
    using StatsBase
    using Random
    using Distributions
    using SharedArrays
end

@everywhere mutable struct Agent
    λ::Float64
    θ::Float64
    op::Float64
    sc::Float64
end

@everywhere opinions(pop::Array{Agent,1}) = Float64[ pop[i].op for i in eachindex(pop) ]

@everywhere function generate_data(bias::Float64, numb_toss::Int, numb_agents::Int)
    sim_dat = Array{Int64,2}(undef, numb_agents, numb_toss)
    @inbounds for i in 1:numb_agents
        sim_dat[i, :] = rand(Bernoulli(bias), numb_toss)
    end
    return sim_dat
end

@everywhere function update!(pop::Array{Agent,1}, data::Array{Int64,2}, t::Int)
    dt = sum(@view(data[:, 1:t]), dims=2)
    @inbounds for i in eachindex(pop)
        pop[i].sc += 1 - (data[i, t] - pop[i].op)^2
        if rand() > pop[i].θ
            pop[i].op = (dt[i] + (pop[i].λ/2))/(t + pop[i].λ)
        else
            pop[i].op = mean([ pop[j].op for j in eachindex(pop) ], weights([ pop[j].sc for j in eachindex(pop) ]))
        end
    end
end

@everywhere function run_model(bias::Float64, 
                               numb_agents::Int,
                               numb_updates::Int)
    data = generate_data(bias, numb_updates, numb_agents)
    pp = [ Agent(rand(Uniform(0, 20)), rand(), rand(), .0) for _ in 1:numb_agents ]
    res = zeros(numb_agents, numb_updates + 1)
    res[:, 1] = opinions(pp)
    @inbounds for i in 1:numb_updates
        update!(pp, data, i)
        res[:, i + 1] = opinions(pp)
    end
    return res
end

res = run_model(.7, 50, 10000)

traces = [ scatter(;x=1:size(res, 2), y=res[i, :], mode="lines", showlegend=false, line_color=palette[i]) for i in 1:50 ]
layout = Layout(width=715, height=475, xaxis=attr(title="Update", type="log", showgrid=false, dtick=1, exponentformat="power"), 
            yaxis_title="Estimate", margin=attr(l=-1, r=-1, t=-1, b=-1),
            title="\u03bb<sub>i</sub> ~ U(0, 20), \u03b8<sub>i</sub> ~ U(0, 1)", 
            shapes=[hline(.7, line_color="grey", line_dash="dashdot")],
                annotations=[(x=4.001, y=.77, text="\u03c4", showarrow=false, font=Dict(size=>16))])
p = plot(traces, layout)

function run_model_evo(pop::Array{Agent,1}; numb_updates=100, numb_test=25)
    ssc = SharedArray{Float64,2}(length(pop), numb_test)
    @inbounds @sync @distributed for t in 1:numb_test
        τ = rand()
        data = generate_data(τ, numb_updates, length(pop))
        res = zeros(length(pop), numb_updates + 1)
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

populate(numb_agents::Int) = [ Agent(rand(Uniform(0, 20)), rand(), rand(), .0) for _ in 1:numb_agents ]

out = run_model_evo(populate(50))

function evolutionary_algorithm(numb_gens::Int, group_size::Int)
    store_pops = Array{Array{Agent,1},1}(undef, numb_gens + 1)
    store_pops[1] = populate(group_size)
    @inbounds for g ∈ 1:numb_gens
        out = run_model_evo(store_pops[g])
        bst = getindex.(findall(i->i<=median(out), out), 1)[1:length(out) ÷ 2]
        parent_pop = store_pops[g][bst]
        child_pop = Agent[]
        @inbounds for i in 1:length(parent_pop)
            a1, a2 = sample(parent_pop, 2)
            push!(child_pop, Agent((a1.λ + a2.λ)/2, (a1.θ + a2.θ)/2, rand(), .0))
        end
        new_pop = vcat(parent_pop, child_pop)
        store_pops[g + 1] = new_pop
    end
    return store_pops
end

res = evolutionary_algorithm(20, 100)

for j in 1:21
    println([ res[j][i].θ for i in 1:100 ] |> mean)
end

for j in 1:21
    println([ res[j][i].λ for i in 1:100 ] |> mean)
end

function run_sims(group_size::Int, numb_gens::Int, numb_sims::Int)
    lambdas = Array{Float64,2}(undef, numb_gens + 1, numb_sims)
    thetas = Array{Float64,2}(undef, numb_gens + 1, numb_sims)
    for s in 1:numb_sims
        res = evolutionary_algorithm(numb_gens, group_size)
        lambdas[:, s] = [ mean([ res[j][i].λ for i in 1:group_size ]) for j in 1:numb_gens + 1 ]
        thetas[:, s] = [ mean([ res[j][i].θ for i in 1:group_size ]) for j in 1:numb_gens + 1 ]
    end
    return lambdas, thetas
end

out = run_sims(50, 20, 100)

traces = [ box(;y=out[1][i, :], name="$i", marker=attr(color=:midnightblue)) for i in 1:20 ]
layout = Layout(width=900, height=525, xaxis_title="Generation", yaxis_title="Mean \u03bb values", font_size=13, showlegend=false)
p = plot(traces, layout)
savefig(p, "lambdas.pdf"; scale=1)

traces = [ box(;y=out[2][i, :], name="$i", marker=attr(color=:midnightblue)) for i in 1:20 ]
layout = Layout(width=900, height=525, xaxis_title="Generation", yaxis_title="Mean \u03b8 values", font_size=13, showlegend=false)
p = plot(traces, layout)
savefig(p, "thetas.pdf"; scale=1)

const lmd = out[1][end, :] |> mean
const tht = out[2][end, :] |> mean

# now define a function to run a model with the best settings

@everywhere function run_model_best(bias::Float64, 
                                    numb_agents::Int,
                                    numb_updates::Int)
    data = generate_data(bias, numb_updates, numb_agents)
    pp = [ Agent(lmd, tht, rand(), .0) for _ in 1:numb_agents ]
    res = zeros(numb_agents, numb_updates + 1)
    res[:, 1] = opinions(pp)
    @inbounds for i in 1:numb_updates
        update!(pp, data, i)
        res[:, i + 1] = opinions(pp)
    end
    return res
end

@everywhere function run_model_non_meta(bias::Float64, 
                                        numb_agents::Int,
                                        numb_updates::Int)
    data = generate_data(bias, numb_updates, numb_agents)
    pp = [ Agent(rand(Uniform(0, 20)), .0, rand(), .0) for _ in 1:numb_agents ]
    res = zeros(numb_agents, numb_updates + 1)
    res[:, 1] = opinions(pp)
    @inbounds for i in 1:numb_updates
        update!(pp, data, i)
        res[:, i + 1] = opinions(pp)
    end
    return res
end

rm(bias::Float64) = (run_model(bias, 50, 1000) .- bias).^2 |> sum
rmb(bias::Float64) = (run_model_best(bias, 50, 1000) .- bias).^2 |> sum
rmnm(bias::Float64) = (run_model_non_meta(bias, 50, 1000) .- bias).^2 |> sum

function sim_rm(numb_sim::Int)
    res = Array{Float64,2}(undef, numb_sim, 6)
    for i in 1:6
        res[:, i] = [ rm((i - 1)/10) for _ in 1:numb_sim ]
    end
    return res
end

function sim_rmb(numb_sim::Int)
    res = Array{Float64,2}(undef, numb_sim, 6)
    for i in 1:6
        res[:, i] = [ rmb((i - 1)/10) for _ in 1:numb_sim ]
    end
    return res
end

function sim_rmnm(numb_sim::Int)
    res = Array{Float64,2}(undef, numb_sim, 6)
    for i in 1:6
        res[:, i] = [ rmnm((i - 1)/10) for _ in 1:numb_sim ]
    end
    return res
end

out1 = sim_rm(50)
out2 = sim_rmb(50)
out3 = sim_rmnm(50)

function bxplt()
    x0 = repeat(["0", "0.1", "0.2", "0.3", "0.4", "0.5"], inner=50)
    trace1 = box(;y=vcat([ out1[:, i] for i in 1:6 ]...),
        x=x0,
        name="Random",
        marker_color=:indianred)
    trace2 = box(;y=vcat([ out2[:, i] for i in 1:6]...),
        x=x0,
        name="Best",
        marker_color=:midnightblue)
    trace3 = box(;y=vcat([ out3[:, i] for i in 1:6 ]...),
        x=x0,
        name="Nonsocial",
        marker_color=:mediumseagreen)
    data = [trace2, trace1, trace3]
    layout = Layout(;xaxis=attr(title="Bias", zeroline=false), yaxis=attr(title="Total Brier loss", zeroline=false), width=1385, height=600,
                    boxmode="group", font_size=13)
    plot(data, layout)
end

b = bxplt()

av_fnc(i) = anova(hcat(out1[:, i], out2[:, i], out3[:, i]))

av_fnc(6)

@everywhere function run_model_mixed(bias::Float64, 
    numb_obj::Int,
    numb_meta::Int,
    numb_updates::Int)
    numb_agents = numb_obj + numb_meta
    data = generate_data(bias, numb_updates, numb_agents)
    po = [ Agent(rand(Uniform(0, 20)), .0, rand(), .0) for _ in 1:numb_obj ]
    pm = [ Agent(rand(Uniform(0, 20)), 1., rand(), .0) for _ in 1:numb_meta ]
    pp = vcat(po, pm)
    res = zeros(numb_agents, numb_updates + 1)
    res[:, 1] = opinions(pp)
    @inbounds for i in 1:numb_updates
        update!(pp, data, i)
        res[:, i + 1] = opinions(pp)
    end
    return res
end

rmx(bias::Float64) = (run_model_mixed(bias, 11, 39, 1000) .- bias).^2 |> sum

function sim_rmx(numb_sim::Int)
    res = Array{Float64,2}(undef, numb_sim, 6)
    for i in 1:6
        res[:, i] = [ rmx((i - 1)/10) for _ in 1:numb_sim ]
    end
    return res
end

out4 = sim_rmx(50)

function bxplt_mx()
    x0 = repeat(["0", "0.1", "0.2", "0.3", "0.4", "0.5"], inner=50)
    trace2 = box(;y=vcat([ out2[:, i] for i in 1:6]...),
        x=x0,
        name="Best",
        marker_color=:midnightblue)
    trace4 = box(;y=vcat([ out4[:, i] for i in 1:6 ]...),
        x=x0,
        name="Mixed",
        marker_color=:indianred)
    data = [trace2, trace4]
    layout = Layout(;xaxis=attr(title="Bias", zeroline=false), yaxis=attr(title="Total Brier loss", zeroline=false), width=1385, height=600,
                    boxmode="group", font_size=13)
    plot(data, layout)
end

b = bxplt_mx()

ttest(x) = EqualVarianceTTest(out2[:, x], out4[:, x])

ttest(6)