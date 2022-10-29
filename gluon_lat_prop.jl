#Data import and plot
using DelimitedFiles, LaTeXStrings, ThreadPools
gluon_frg_T5 = readdlm("latticeTdata/gluon_prop_trans_T_486.dat")[(2):end, :]
toPhysUnits = 1.0

p0_gluon = gluon_frg_T5[:, 1] * toPhysUnits
G_gluon = gluon_frg_T5[:, 2]
p0_gluon_sparse = p0_gluon[1:10:end]
G_gluon_sparse = G_gluon[1:10:end]
plot(p0_gluon, G_gluon; xaxis=:log, seriestype=:scatter)
plot(p0_gluon_sparse, G_gluon_sparse; xaxis=:log, seriestype=:scatter)

f_gluon = @gppp let
    f1 = GP(PropKernel(; σ=1.0, l=1 / 1.0))
    f2 = 10 * stretch(GP(SEKernel()), 10)
end

G_gluon_sparse_error = G_gluon_sparse

plot(x_gluon_f1.x, G_gluon_sparse_error; color=:red, seriestype=:scatter, xaxis=:log)

x_gluon_f2 = GPPPInput(:f2, p0_gluon_sparse);
fx_gluon_f2 = f_gluon(x_gluon_f2, 1e-5)
f_gluon_posterior_f2 = posterior(fx_gluon_f2, G_gluon_sparse_error)
x_gluon_plot = exp.(range(log(0.002), log(100); length=50));
x_gluon_posterior_f2 = GPPPInput(:f2, x_gluon_plot);

x_gluon_f1 = GPPPInput(:f1, p0_gluon_sparse);
fx_gluon_f1 = f_gluon(x_gluon_f1, 1e-1)
f_gluon_posterior_f1 = posterior(fx_gluon_f1, G_gluon_sparse_error)
x_gluon_posterior_f1 = GPPPInput(:f1, x_gluon_plot);

-logpdf(fx_gluon_f1, G_gluon_sparse_error)

function NLL_fun(σ, l)
    f_gluon = @gppp let
        f1 = GP(PropKernel(; σ=σ, l=l))
    end
    fx_gluon_f1 = f_gluon(x_gluon_f1, 1e-1)
    return -logpdf(fx_gluon_f1, G_gluon_sparse_error)
end


NLL_fun(10.0, 1.0)

NLL_sigma = exp.(range(log(0.1), log(1000); length=40));
NLL_l = exp.(range(log(0.1), log(1000); length=40));

Nll_result = zeros(length(NLL_sigma), length(NLL_l));

NLL_fun(0.01, 0.01)

@qthreads for i in 1:40
    for j in 1:40
        Nll_result[i, j] = NLL_fun(NLL_sigma[i], NLL_l[j])
    end
end

heatmap(NLL_sigma, NLL_l, Nll_result; xaxis=:log, yaxis=:log)

plot!(
    x_gluon_plot,
    mean(f_gluon_posterior_f1(x_gluon_posterior_f1));
    label="l=0.01",
    # color=:blue,
    xaxis=:log,
    xlabel=L"p_0\;[\mathrm{GeV}]",
    ylabel=L"G\left(p_0\right)\;[\mathrm{GeV^{-2}}]",
)

plot!(
    x_gluon_plot,
    f_gluon_posterior_f1(x_gluon_posterior_f1);
    ribbon_scale=1,
    label="l=1, s=1",
    # color=:blue,
    fillalpha=0.2,
    linewidth=2,
    xaxis=:log,
    xlabel=L"p_0\;[\mathrm{GeV}]",
    ylabel=L"G\left(p_0\right)\;[\mathrm{GeV^{-2}}]",
    ylims=(-0.05, 0.22),
)

plot!(
    x_gluon_plot,
    f_gluon_posterior_f1(x_gluon_posterior_f1);
    ribbon_scale=1,
    label="",
    # color=:blue,
    fillalpha=0.2,
    linewidth=2,
    xaxis=:log,
)

plot!(
    x_gluon_plot,
    f_gluon_posterior_f2(x_gluon_posterior_f2);
    ribbon_scale=1,
    label="",
    # color=:blue,
    fillalpha=0.2,
    linewidth=2,
    xaxis=:log,
    ylimit=(-0.1, 0.3),
)

#Optimize
function build_model_f2(θ::NamedTuple)
    return @gppp let
        f1 = GP(PropKernel(; σ=θ.σ, l=θ.l))
        f2 = θ.σ * stretch(GP(SEKernel()), θ.l)
    end
end
using ParameterHandling

θpar_f2 = (
    # Short length-scale and small variance.
    l=positive(1.0 / 10.0),
    σ=positive(10.0),

    # Long length-scale and larger variance.
    s_noise=positive(0.1, exp, 1e-4),
)

using ParameterHandling
using ParameterHandling: value, flatten

θ_flat_init, unflatten = flatten(θpar_f2)
unpack = value ∘ unflatten
function nlml(θ::NamedTuple)
    f = build_model_f2(θ)
    return -logpdf(f(x_gluon_f1, 1e-3), G_gluon_sparse_error)
end

using Optim
results = Optim.optimize(
    nlml ∘ unpack,
    θ_flat_init + randn(length(θ_flat_init)),
    NelderMead(),
    Optim.Options(; show_trace=true);
)

using Zygote: gradient

results = Optim.optimize(
    nlml ∘ unpack,
    θ -> gradient(nlml ∘ unpack, θ)[1],
    θ_flat_init + 0.1 * randn(length(θ_flat_init)),
    BFGS(),
    Optim.Options(; show_trace=true);
    inplace=false,
)

θ_opt = unpack(results.minimizer)

f_opt = build_model_f2(θ_opt)
f_posterior_opt = posterior(f_opt(x_gluon_f1, 1e-3), G_gluon_sparse_error);

plot!(
    x_gluon_plot,
    f_posterior_opt(x_gluon_posterior_f1);
    ribbon_scale=1,
    label="",
    # color=:blue,
    fillalpha=0.2,
    linewidth=2,
    xaxis=:log,
)
