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
        p = plot(
            data["n"],
            [data["faer"] data["mkl"] data["openblas"]],
            size=SIZE,
            xaxis=:log,
            yaxis=:log,
            title="$(name) ($(ty))",
            label=["faer" "mkl" "openblas"],
            xlabel="Matrix dimension (n)",
            ylabel="n³ / time (seconds)",
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
        p = plot(
            data["n"],
            [data["faer"] data["mkl"] data["openblas"] data["nalgebra"]],
            size=SIZE,
            xaxis=:log,
            yaxis=:log,
            title="$(name) ($(ty))",
            label=["faer" "mkl" "openblas" "nalgebra"],
            xlabel="Matrix dimension (n)",
            ylabel="n³ / time (seconds)",
        )
        savefig(p, "./target/st_$(algo)_$(ty)_plot.png")
    end
end

