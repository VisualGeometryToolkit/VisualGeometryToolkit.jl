# A pool record's fields by name (pool_io_tests.jl, report_helpers_tests.jl, sample_starts_tests.jl): the keyword
# constructor's arguments, to rebuild a record with some fields replaced.
@testsetup module PoolRecordFields
    export record_fields
    record_fields(p) = NamedTuple{fieldnames(typeof(p))}(Tuple(getfield(p, f) for f in fieldnames(typeof(p))))
end
