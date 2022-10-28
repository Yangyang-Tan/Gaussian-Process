gluon_frg_T5 = readdlm("frgTdata/T_5.000/gluon_prop.dat")[(end - 49):end, :]
toPhysUnits = 0.580061

p0_gluon = gluon_frg_T5[:, 2]*toPhysUnits
G_gluon = 1 ./(gluon_frg_T5[:, 3]*toPhysUnits^2)
plot(p0_gluon, G_gluon; xaxis=:log, seriestype=:scatter)

x = GPPPInput(:f1, exp.(range(log(0.01), log(100); length=200)));
f = @gppp let
    f1 = GP(
        x -> spec_mean_fun.(x; p_grid=p0_gluon, G_grid=G_gluon, σ=10.0, l=1 / 10.0),
        SpecKernel(; p_grid=p0_gluon, σ=10.0, l=1 / 10.0),
    )
end
5000*toPhysUnits
plot(
    x.x,
    f(x);
    ribbon_scale=1,
    label="T=2.9 GeV,M",
    color=:blue,
    fillalpha=0.2,
    linewidth=2,
    xaxis=:log,
)

plot(x.x, mean(f(x)); xaxis=:log)
