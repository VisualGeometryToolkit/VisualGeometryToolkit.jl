@testitem "require_extension: the missing extension's message, or the MethodError of a loaded one" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: require_extension
    # the Test extension is loaded in every test worker (`using Test` triggers it); a name that is no extension is not
    loaded, absent = :VisualGeometryToolkitCoreTestExt, :NoSuchExtension
    @test Base.get_extension(VgtkCore, loaded) !== nothing && Base.get_extension(VgtkCore, absent) === nothing
    missing_msg() = ArgumentError("a part needs `using SomePackage` (its package extension is not loaded)")
    built = Ref(0)
    counted() = (built[] += 1; missing_msg())

    # the check: nothing when loaded (the message never built), the caller's exception when not
    @test require_extension(counted, VgtkCore, loaded) === nothing
    @test built[] == 0
    @test_throws missing_msg() require_extension(counted, VgtkCore, absent)
    @test built[] == 1
    e = try require_extension(VgtkCore, absent) do
            ErrorException("an error of the caller's own type")
        end
    catch err
        err
    end
    @test e isa ErrorException && e.msg == "an error of the caller's own type"

    # the fallback: the message without the extension; with it, the MethodError of the arguments at the caller's world
    fallback(args...) = require_extension(missing_msg, fallback, args, VgtkCore, absent)
    @test_throws missing_msg() fallback(1, "two")
    loaded_fallback(args...) = require_extension(missing_msg, loaded_fallback, args, VgtkCore, loaded)
    # the world this caller runs in, read in the same top-level statement as the call (each statement of a test item
    # runs in the world its predecessors left)
    w, m = let w = Base.tls_world_age()
        (w, try
            loaded_fallback(1, "two")
        catch err
            err
        end)
    end
    @test m isa MethodError && m.f === loaded_fallback && m.args === (1, "two")
    @test m.world == w                # the caller's world age, so a call compiled before the load keeps Julia's hint
end
