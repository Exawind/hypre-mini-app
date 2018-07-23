
#include "HypreSystem.h"
#include "utils.h"
#include "mpi.h"

#include "yaml-cpp/yaml.h"

#include <iostream>
#include <chrono>
#include <iomanip>

void solve_system(nalu::HypreSystem& linsys)
{
    linsys.load();
    linsys.solve();

    linsys.check_solution();
    linsys.output_linear_system();
    linsys.summarize_timers();
}

void memcheck_hypre(nalu::HypreSystem& linsys, int nsteps)
{
    int iproc, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    for (int i=0; i < nsteps; i++) {
        linsys.load();
        linsys.solve();

        const size_t bytes = nalu::current_memory_usage();
        size_t bytesSum = 0.0;
        MPI_Reduce(&bytes, &bytesSum, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        bytesSum /= nproc;
        if (iproc == 0) {
            std::cout << "Step: " << std::setw(5) << (i + 1)
                      << "; Memory = " << nalu::human_bytes(bytesSum)
                      << std::endl;
        }

        linsys.teardown();
    }
}

int main(int argc, char* argv[])
{
    int iproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);

    auto start = std::chrono::system_clock::now();


    if (argc != 2) {
        std::cout << "ERROR!! Incorrect arguments passed to program." << std::endl
                  << "Usage: hypre_app INPUT_FILE" << std::endl << std::endl;
        return 1;
    }

    std::string yaml_filename(argv[1]);
    YAML::Node inpfile = YAML::LoadFile(yaml_filename);

    bool do_memcheck = false;
    int memcheck_nsteps = 0;
    if (inpfile["memcheck"]) {
        YAML::Node mcheck = inpfile["memcheck"];
        do_memcheck = mcheck["perform_memcheck"].as<bool>();
        memcheck_nsteps = mcheck["num_steps"].as<int>();
    }

    {
        nalu::HypreSystem linsys(MPI_COMM_WORLD, inpfile);

        if (do_memcheck)
            memcheck_hypre(linsys, memcheck_nsteps);
        else
            solve_system(linsys);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    if (iproc == 0)
        std::cout << "Total time: " << elapsed.count() << " seconds" << std::endl;

    MPI_Finalize();
    return 0;
}
