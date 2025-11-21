for cpu in 1.0 2.0 4.0 8.0; do
  echo "Running cpu_budget_gb=${cpu}..."
  python fit_I_cpu.py \
    --cpu-budget-gb $cpu \
    --output cpu_offloading_results/fit_bw_cpu_gpu_results_cpu${cpu}gb.json \
    > cpu_offloading_logs/fit_bw_cpu_gpu_cpu${cpu}gb.log 2>&1
done
