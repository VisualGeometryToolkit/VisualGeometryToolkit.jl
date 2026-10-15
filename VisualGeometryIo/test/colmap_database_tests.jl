# =============================================================================
# colmap_database_tests.jl — the COLMAP sqlite read surface, against a
# synthetic database written with the exact COLMAP schema + blob layout
# (row-major float32/uint8/uint32, pair_id codec with the id swap, and the
# pre-3.x `inlier_matches` verified table as found in the Aachen benchmark
# database).
# =============================================================================

@testitem "colmap database" setup = [VGIoFixtures] begin
    using SQLite: SQLite
    import VisualGeometryIo as VGIo
    using VisualGeometryIo: ColmapDatabase

    const _DBI = SQLite.DBInterface

    # row-major blob of a Julia matrix
    _blob(m::AbstractMatrix) = collect(reinterpret(UInt8, vec(permutedims(m))))

    function _make_test_db(path)
        db = SQLite.DB(path)
        for sql in (
                "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)",
                "CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)",
                "CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)",
                "CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)",
                "CREATE TABLE inlier_matches (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB, config INTEGER)",
            )
            _DBI.execute(db, sql)
        end
        _DBI.execute(db, "INSERT INTO images VALUES (1, 'db/a.jpg', 1)")
        _DBI.execute(db, "INSERT INTO images VALUES (2, 'db/b.jpg', 1)")
        _DBI.execute(db, "INSERT INTO images VALUES (7, 'query/q.jpg', 2)")

        kp1 = Float32[10 20 1.5 0.1; 30 40 2.0 0.2; 50 60 2.5 0.3]   # 3×4
        kp2 = Float32[11 21 1.0 0.0; 31 41 1.0 0.0]                   # 2×4
        _DBI.execute(db, "INSERT INTO keypoints VALUES (1, 3, 4, ?)", (_blob(kp1),))
        _DBI.execute(db, "INSERT INTO keypoints VALUES (2, 2, 4, ?)", (_blob(kp2),))
        _DBI.execute(db, "INSERT INTO keypoints VALUES (7, 0, 4, NULL)")

        d1 = UInt8.(reshape(0:(3 * 128 - 1), 128, 3)' .% 256)          # 3×128
        _DBI.execute(db, "INSERT INTO descriptors VALUES (1, 3, 128, ?)", (_blob(d1),))

        # matches for the pair (1, 7): stored under min-first order (1, 7)
        m17 = UInt32[0 1; 2 0]                                        # 2×2, 0-based
        _DBI.execute(
            db, "INSERT INTO matches VALUES (?, 2, 2, ?)",
            (VGIo.pair_id(1, 7), _blob(m17))
        )
        # an EMPTY match row (rows=0) — must not count as a partner
        _DBI.execute(
            db, "INSERT INTO matches VALUES (?, 0, 2, NULL)",
            (VGIo.pair_id(2, 7),)
        )
        # verified subset in the PRE-3.x table
        _DBI.execute(
            db, "INSERT INTO inlier_matches VALUES (?, 1, 2, ?, 2)",
            (VGIo.pair_id(1, 7), _blob(UInt32[0 1;]))
        )
        _DBI.close!(db)
        return (; kp1, kp2, d1, m17)
    end

    path = joinpath(mktempdir(), "test.db")
    ref = _make_test_db(path)
    cdb = ColmapDatabase(path)

    @testset "pair-id codec" begin
        @test VGIo.pair_id(2, 4480) == 4294971774          # from aachen.db
        @test VGIo.pair_image_ids(4294971774) == (2, 4480)
        @test VGIo.pair_id(4480, 2) == VGIo.pair_id(2, 4480)  # order-insensitive
        @test VGIo.pair_image_ids(VGIo.pair_id(7, 1)) == (1, 7)
    end

    @testset "image table" begin
        @test VGIo.n_images(cdb) == 3
        @test VGIo.image_names(cdb) == ["db/a.jpg", "db/b.jpg", "query/q.jpg"]
        @test VGIo.image_id(cdb, "query/q.jpg") == 7
        @test VGIo.image_name(cdb, 2) == "db/b.jpg"
        @test_throws ArgumentError VGIo.image_id(cdb, "nope.jpg")
        @test_throws ArgumentError VGIo.image_name(cdb, 99)
    end

    @testset "keypoints / descriptors" begin
        @test VGIo.keypoints(cdb, 1) == ref.kp1
        @test VGIo.keypoints(cdb, 2) == ref.kp2
        @test size(VGIo.keypoints(cdb, 7)) == (0, 4)       # NULL blob, 0 rows
        @test VGIo.descriptors(cdb, 1) == ref.d1
        @test_throws ArgumentError VGIo.keypoints(cdb, 99)
    end

    @testset "matched_pairs" begin
        @test VGIo.matched_pairs(cdb, 7) == [1]     # rows=0 pair (2,7) excluded
        @test VGIo.matched_pairs(cdb, 1) == [7]
        @test VGIo.matched_pairs(cdb, 2) == Int[]
    end

    @testset "matches" begin
        @test VGIo.matches(cdb, 1, 7) == ref.m17
        # caller order reversed ⇒ columns swapped
        @test VGIo.matches(cdb, 7, 1) == ref.m17[:, [2, 1]]
        # absent pair ⇒ 0×2, not an error
        @test size(VGIo.matches(cdb, 1, 2)) == (0, 2)
        # verified table auto-detected as the pre-3.x name
        @test cdb.verified_table == "inlier_matches"
        @test VGIo.verified_matches(cdb, 1, 7) == UInt32[0 1;]
        @test VGIo.verified_matches(cdb, 7, 1) == UInt32[1 0;]
    end

    close(cdb)

    @test_throws ArgumentError ColmapDatabase(joinpath(mktempdir(), "no.db"))
end
