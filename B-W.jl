function Prop_BW_fun(p0; A, Γ, M)
    return A / ((p0 + Γ)^2 + M^2)
end

function Spec_BW_fun(ω; A, Γ, M)
    return 4 * A * Γ * ω / (4 * Γ^2 * ω^2 + (M^2 + Γ^2 - ω^2)^2)
end

plot(0.1:1:50, Prop_BW_fun.(0.1:1:50; A=1, Γ=1, M=1))

spec_mean_fun(
    1.0; p_grid=0.1:1:50, G_grid=Prop_BW_fun.(0.1:1:50; A=1, Γ=1, M=1), σ=10.0, l=1 / 10
)

plot(0.05:0.01:20.0, Spec_BW_fun.(0.05:0.01:20.0; A=1, Γ=1, M=1); xaxis=:log)

plot!(
    0.05:0.05:20.0,
    spec_mean_fun.(
        0.05:0.05:20.0;
        p_grid=0.01:0.2:20,
        G_grid=Prop_BW_fun.(0.01:0.2:20; A=1, Γ=1, M=1),
        σ=10.0,
        l=1 / 10,
    ),
    xaxis=:log,
)
