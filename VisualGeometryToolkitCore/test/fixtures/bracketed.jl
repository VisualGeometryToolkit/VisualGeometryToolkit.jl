# A list written as text: each element through `f`, joined by `sep`, between `open` and `close` — the one spelling of
# the record writers of the tests (chain_equivalence/core_values.jl's canonical text, fixture_literals.jl's Julia
# literals, VisualGeometryToolkitIO's hand-written JSON records). Each writer states its own grammar in its arguments.
bracketed(f, xs, sep; open = "[", close = "]") = open * join(f.(xs), sep) * close
