gluon_lat_T121 = readdlm("latticeTdata/gluon_prop_trans_T_486.dat")
gluon_lat_T121[:, 1]
gluon_lat_T121[:, 2]
plot(gluon_lat_T121[2:end, 1], gluon_lat_T121[2:end, 2]; xaxis=:log, seriestype=:scatter)
plot(p0_gluon, G_gluon; xaxis=:log, seriestype=:scatter)

gluon_lat_T121[2:end, 1]

p0_gluon = gluon_lat_T121[2:end, 1]
G_gluon = gluon_lat_T121[2:end, 2]
x = GPPPInput(:f1, exp.(range(log(0.01), log(40); length=100)));
f = @gppp let
    f1 = GP(
        x -> spec_mean_fun.(x; p_grid=p0_gluon, G_grid=G_gluon, σ=10.0, l=1 / 10.0),
        SpecKernel(; p_grid=p0_gluon, σ=10.0, l=1 / 10.0),
    )
end


plot(
    x.x, f(x); ribbon_scale=1, label="T=121 MeV,M", color=:blue, fillalpha=0.2, linewidth=2, xaxis=:log
)

plot!(
    x.x, f(x); ribbon_scale=1, label="T=486 MeV,M", fillalpha=0.2, linewidth=2, xaxis=:log
)
