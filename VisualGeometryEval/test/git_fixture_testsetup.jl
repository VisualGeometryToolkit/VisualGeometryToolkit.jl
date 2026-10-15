# The git-repository fixture of the provenance tests (experiments_data_tests.jl).

@testsetup module GitFixture
    export git!, gitrepo
    # A throwaway git repository with one commit (CLI git: the fixture must not
    # share code with what it tests).
    git!(dir, args...) = run(pipeline(`git -C $dir -c user.name=t -c user.email=t@t -c commit.gpgsign=false -c tag.gpgsign=false $args`;
                                      stdout = devnull, stderr = devnull))
    function gitrepo()
        dir = mktempdir()
        git!(dir, "init", "-q")
        write(joinpath(dir, "a.txt"), "a\n")
        git!(dir, "add", "a.txt"); git!(dir, "commit", "-q", "-m", "one")
        return dir
    end
end
