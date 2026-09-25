# Run from this directory: julia --project=. run_all_examples.jl
examples_dir = @__DIR__

for group in ("MuKumari", "SCRIBE", "MuKumariSCRIBE")
    for filename in readdir(joinpath(examples_dir, group))
        endswith(filename, ".jl") || continue
        script = joinpath(examples_dir, group, filename)
        println("\nRunning ", group, "/", filename)
        flush(stdout)
        # A fresh process releases each example's models and plotting memory.
        command = `$(Base.julia_cmd()) --project=$examples_dir --threads=1 $script`
        run(addenv(command, "OPENBLAS_NUM_THREADS" => "1",
                            "JULIA_NUM_PRECOMPILE_TASKS" => "1",
                            "GKSwstype" => "100"))
    end
end

println("\nAll examples completed. Figures are in ", joinpath(examples_dir, "res"))
