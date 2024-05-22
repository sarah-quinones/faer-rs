using Plots
using CSV

ENV["GKSwstype"]="nul"

SIZE = (640, 400)

for ty in ["f32", "f64", "c32", "c64"]
    for (algo, name) in [
        ("cholesky", "Cholesky"),
        ("qr", "QR"),
        ("piv_qr", "QR with column pivoting"),
        ("lu", "LU with partial pivoting"),
        ("piv_lu", "LU with full pivoting"),
        ("svd", "Singular value decomposition"),
        ("thin_svd", "Thin singular value decomposition"),
        ("eigh", "Self adjoint eigenvalue decomposition"),
        ("eig", "General eigenvalue decomposition"),
    ]
        data = CSV.File("./target/mt_$(algo)_$(ty).csv", types=Float64)
        norm = data["faer"]
        p = plot(
            data["n"],
            [data["faer"]./norm data["mkl"]./norm data["openblas"]./norm],
            size=SIZE,
            xaxis=:log,
            yaxis=:log,
            title="$(name) ($(ty))",
            label=["faer" "mkl" "openblas"],
            xlabel="Matrix dimension (n)",
            ylabel="1/time (normalized)",
            ylims=(0.1, 10.0),
            yticks=([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], ["0.1", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0"])
        )
        savefig(p, "./target/mt_$(algo)_$(ty)_plot.png")
    end
end

for ty in ["f32", "f64", "c32", "c64"]
    for (algo, name) in [
        ("cholesky", "Cholesky"),
        ("qr", "QR"),
        ("piv_qr", "QR with column pivoting"),
        ("lu", "LU with partial pivoting"),
        ("piv_lu", "LU with full pivoting"),
        ("svd", "Singular value decomposition"),
        ("thin_svd", "Thin singular value decomposition"),
        ("eigh", "Self adjoint eigenvalue decomposition"),
        ("eig", "General eigenvalue decomposition"),
    ]
        data = CSV.File("./target/st_$(algo)_$(ty).csv", types=Float64)
        norm = data["faer"]
        if all(isnan.(data["nalgebra"]))
            funcs = [data["faer"]./norm data["mkl"]./norm data["openblas"]./norm]
            label = ["faer" "mkl" "openblas"]
        else
            funcs = [data["faer"]./norm data["mkl"]./norm data["openblas"]./norm data["nalgebra"]./norm]
            label = ["faer" "mkl" "openblas" "nalgebra"]
        end
        p = plot(
            data["n"],
            funcs,
            size=SIZE,
            xaxis=:log,
            yaxis=:log,
            title="$(name) ($(ty))",
            label=label,
            xlabel="Matrix dimension (n)",
            ylabel="1/time (normalized)",
            ylims=(0.1, 10.0),
            yticks=([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], ["0.1", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0"])
        )
        savefig(p, "./target/st_$(algo)_$(ty)_plot.png")
    end
end

