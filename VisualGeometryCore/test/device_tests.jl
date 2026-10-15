@testitem "device" begin
    using Test
    using VisualGeometryCore: default_device, reset_default_device!, _DEVICE_RESOLVER
    import KernelAbstractions as KA

    @testset "default_device — the family's one backend resolver" begin
        # The host is the default while no accelerator extension has registered a
        # resolver. Under ReTestItems another item on this worker may have loaded
        # CUDA (pointwise's GPU smoke), and its extension registers the GPU —
        # so the claim holds only when that extension is absent.
        if Base.get_extension(VisualGeometryCore, :VisualGeometryCoreCUDAExt) === nothing
            reset_default_device!()
            @test default_device() === KA.CPU()
        end
        # Memoized: the resolver runs once per session.
        calls = Ref(0)
        old = _DEVICE_RESOLVER[]
        try
            _DEVICE_RESOLVER[] = () -> (calls[] += 1; KA.CPU())
            reset_default_device!()
            default_device(); default_device()
            @test calls[] == 1
            # An accelerator extension registers by ASSIGNMENT; a reset re-resolves.
            _DEVICE_RESOLVER[] = () -> (calls[] += 1; KA.CPU())
            reset_default_device!()
            default_device()
            @test calls[] == 2
        finally
            _DEVICE_RESOLVER[] = old
            reset_default_device!()
        end
    end
end
