# =============================================================================
# Precompile workload
# =============================================================================
#
# Caches the ASIFT kernels and the matcher in the package image, so a fresh
# process does not recompile them: every ReTestItems worker, and every user's
# first call. Cold, the first asift_detect took 7.8 s and the first nn_match
# 1.0 s before this existed.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    img = Gray{N0f8}.(
        clamp.([0.5 + 0.35 * exp(-((i - 24)^2 + (j - 30)^2) / 18) for i in 1:64, j in 1:64], 0, 1)
    )
    d1 = rand(Float32, 20, 64)
    d2 = rand(Float32, 24, 64)
    @compile_workload begin
        for nms in (:mxm, :ellipse)
            asift_detect(img; device = KA.CPU(), nms)
        end
        nn_match(d1, d2)
        # no plan may survive into the package image
        clear_asift_plans!()
    end
end
