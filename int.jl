using Distributed, Plots, Interpolations
addprocs(["xeon6146b", "xeon6146c"])
addprocs(1)

@everywhere include("kernelfun.jl")

@time @everywhere Ctilde_RBF_fun(0.01, 0.01; σ=10.0, l=1 / 10)
@everywhere p1v1 = exp.(range(log(0.001), log(100); length=1000))
@everywhere p2v1 = exp.(range(log(0.001), log(100); length=1000))

testv2 = DArray((1000, 1000)) do inds # local indices on each processor
    # @show inds
    arr = zeros(inds) # We create an OffsetArray with the correct local indices
    Threads.@threads for i in inds[1]
        for j in inds[2]
            arr[i, j] = Ctilde_RBF_fun(p1v1[i], p2v1[j]; σ=10.0, l=1 / 10)
        end
    end
    parent(arr)
end;

testv5 = DArray((1000, 1000)) do inds # local indices on each processor
    # @show inds
    arr = zeros(inds) # We create an OffsetArray with the correct local indices
    Threads.@threads for i in inds[1]
        for j in inds[2]
            arr[i, j] = Ctilde_RBF_fun(p1v1[i], p2v1[j]; σ=1, l=1 / 10)
        end
    end
    parent(arr)
end;
testv4
using DelimitedFiles
writedlm("kck.dat", testv)
writedlm("kck2.dat", Array(testv2))
writedlm("kcksigma=1_l=0.1.dat", Array(testv5))

testv = readdlm("kck.dat")
testv2 = readdlm("kck2.dat")
testv3 = readdlm("kcksigma=1_l=1.dat")

invtestv2 = inv(testv2 + 0.005 .* Matrix(I, 1000, 1000));
invtestv2min = minimum(invtestv2)
invtestv2max = maximum(invtestv2)

heatmap(
    invtestv2;
    aspect_ratio=1,
    size=(600, 600),
    c=colormap(
        "RdBu", 2000; logscale=true, mid=-invtestv2min / (invtestv2max - invtestv2min)
    ),
    highclip=:black,
    lowclip=:white,
)

itp_invCtilde_fun = interpolate(invtestv2, BSpline(Cubic(Line(OnGrid()))))
sitp_invCtilde_fun = scale(itp_invCtilde_fun, 0.1:0.1:100, 0.1:0.1:100)

itp_Ctilde_fun = interpolate(testv2, BSpline(Cubic(Line(OnGrid()))))
sitp_Ctilde_fun_temp = scale(
    itp_Ctilde_fun,
    range(log(0.001), log(100); length=1000),
    range(log(0.001), log(100); length=1000),
)

sitp_Ctilde_fun(x, y) = sitp_Ctilde_fun_temp(log(x), log(y))
