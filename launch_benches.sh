OMP_NUM_THREADS=12 ./mkl_bench --output="./target/diol_mkl_mt.json" --func-filter="$FUNC_FILTER" --arg-filter="$ARG_FILTER"
OMP_NUM_THREADS=12 ./openblas_bench --output="./target/diol_openblas_mt.json" --func-filter="$FUNC_FILTER" --arg-filter="$ARG_FILTER"
OMP_NUM_THREADS=1 ./mkl_bench --output="./target/diol_mkl_st.json" --func-filter="$FUNC_FILTER" --arg-filter="$ARG_FILTER"
OMP_NUM_THREADS=1 ./openblas_bench --output="./target/diol_openblas_st.json" --func-filter="$FUNC_FILTER" --arg-filter="$ARG_FILTER"
cargo bench --bench bench --features=nightly -- --output="./target/diol_faer_mt.json" --func-filter="faer_par_""$FUNC_FILTER" --arg-filter="$ARG_FILTER"
cargo bench --bench bench --features=nightly -- --output="./target/diol_faer_st.json" --func-filter="faer_seq_""$FUNC_FILTER" --arg-filter="$ARG_FILTER"
cargo bench --bench bench --features=nightly -- --output="./target/diol_nalgebra_st.json" --func-filter="nalgebra_""$FUNC_FILTER" --arg-filter="$ARG_FILTER"

cargo bench --bench bench_aggregate --features=nightly
