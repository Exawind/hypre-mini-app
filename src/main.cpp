
#include "HypreSystem.h"
#include "mpi.h"

#include <iostream>
#include <chrono>

int main(int argc, char* argv[])
{
    int iproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);

    auto start = std::chrono::system_clock::now();

    nalu::HypreSystem linsys(MPI_COMM_WORLD);

    if (argc != 4) {
        std::cout << "ERROR!! Incorrect arguments passed to program." << std::endl
                  << "Usage: hypre_app MATRIX RHS SLN" << std::endl << std::endl;
        return 1;
    }

    std::string matfile(argv[1]);
    std::string rhsfile(argv[2]);
    std::string slnfile(argv[3]);

    linsys.load_matrix(matfile);
    linsys.load_rhs_vector(rhsfile);
    linsys.load_sln_vector(slnfile);

    linsys.setup_preconditioner();
    linsys.setup_solver();
    linsys.solve();

    linsys.check_solution();
    // linsys.output_linear_system();
    linsys.summarize_timers();

    MPI_Barrier(MPI_COMM_WORLD);
    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    if (iproc == 0)
        std::cout << "Total time: " << elapsed.count() << " seconds" << std::endl;

    MPI_Finalize();
    return 0;
}
