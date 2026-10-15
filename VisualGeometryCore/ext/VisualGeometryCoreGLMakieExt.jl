module VisualGeometryCoreGLMakieExt

# =============================================================================
# GLMakie Extension for VisualGeometryCore
# =============================================================================
#
# Slim — almost everything Makie-related lives in `VisualGeometryCoreMakieExt`
# (which auto-loads as soon as any Makie backend is `using`ed).  This module
# only adds the GLMakie-backend-specific window helpers `window!` and
# `closeall`.
#
# =============================================================================

using VisualGeometryCore
using GLMakie: GLMakie
using Makie: Makie

import VisualGeometryCore: window!, closeall

"""
    window!(fig::Figure, title::String) -> GLMakie.Screen

Display a Figure in a new named GLMakie window.  Works around GLMakie's screen
attachment limitation by clearing any existing screen attachments before
displaying to a new named screen.
"""
function window!(fig::Makie.Figure, title::String)
    scene = fig.scene
    for screen in copy(scene.current_screens)
        if screen isa GLMakie.Screen
            close(screen)
        end
    end
    empty!(scene.current_screens)
    screen = GLMakie.Screen(; title = title, visible = true)
    display(screen, fig)
    return screen
end

"""
    closeall()

Close all open GLMakie screen windows.
"""
closeall() = GLMakie.closeall()

end # module VisualGeometryCoreGLMakieExt
