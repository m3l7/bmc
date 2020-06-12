#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <random>
#include <omp.h>

#define Nm 128 // number of points in each dimension
#define Nm2 (Nm*Nm)
#define mesh_x0 -2.
#define mesh_x1 2.
#define mesh_v0 -2.
#define mesh_v1 2.
#define Nt 100
#define n_mc_steps 100
#define h 1.
#define sigma 0.1
#define subcycles 12
#define hsubsycle (h/subcycles)

std::default_random_engine generator;
std::normal_distribution<double> distribution(0., 1.);

/** Init phase space (x-v) linspace for the rectangular mesh
 *  The mesh is defined as [mesh_x0, mesh_x1] X [mesh_v0, mesh_v1]
 * 
 * @param  {double*} Xs : X output linspace
 * @param  {double*} Vs : V output linspace
 */
void initMeshLinSpace(double *Xs, double *Vs) {
    double x, y;
    int ix;
    for (int i = 0; i < Nm2; ++i) {
        Xs[i] = (i % Nm) / (Nm - 1.) * (mesh_x1 - mesh_x0) + mesh_x0;
        Vs[i] = trunc(i / Nm) / (Nm - 1.) * (mesh_v1 - mesh_v0) + mesh_v0;
    }
}

/**
 *  Check if point i is inside the target domain Omega
 *  Omega is set to be at x = mesh_x0 and v \in [-0.5, 0]
 * @param  {int} i      : 
 * @param  {double*} Xs : 
 * @param  {double*} Vs : 
 * @return {bool}       : 
 */
bool indexInOmega(int i, double *Xs, double *Vs) {
    if ((i % Nm == 0) && (Vs[i] >= -0.5) && (Vs[i] <= 0)) {
        return true;
    } else {
        return false;
    }
}

/**
 * initialize the probabilities with the indicator function of omega
 *
 * @param  {double*} meshP : output probabilities for the mesh
 * @param  {double*} Xs   : 
 * @param  {double*} Vs   : 
 */
void initMeshProbabilities(double *meshP, double *Xs, double *Vs) {
    for (int i = 0; i < Nm2; ++i) {
        if (indexInOmega(i, Xs, Vs)) {
            meshP[i] = 1.;
        } else {
            meshP[i] = 0.;
        }
    }
}


/**
 * Advance the particle in time using symplectic Euler with subcycles
 * @param  {double} x0 : 
 * @param  {double} v0 : 
 * @param  {double*} x : 
 * @param  {double*} v : 
 */
void particleStepForward(double x0, double v0, double *x, double *v) {
    // FIXME: Fix the random number generator
    // std::random_device rd{};
    // std::mt19937 gen{rd()};
    for (int i = 0; i < subcycles; ++i) {
        // hamiltonian push
        *x = x0 * (1 - hsubsycle*hsubsycle) + hsubsycle * v0;
        *v = v0 - hsubsycle * x0;
    }

    // update v with maryama euler
    *v += sigma * sqrt(h) * distribution(generator);
}

/**
 * Deposit a particle, with continuous coordinates, into the mesh
 * and compute the Phi function contribution of the particle (eq. 21 - Hirvijoki 2019)
 * @param  {double*} meshP0 : Mesh probabilities of the previous time step
 * @param  {double} x      : Particle x position
 * @param  {double} v      : Particle v position
 * @param  {double*} Xs    : X linspace
 * @param  {double*} Vs    : V linspace
 * @return {double}        : Phi contribution of the particle (eq. 21)
 */
double particlePhiContribution(double *meshP0, double x, double v, double *Xs, double *Vs) {

    double ret = 0;

    // find lower left vertex of the simplex including the point
    int ix, iv;
    if (x < mesh_x0) {
        ix = 0;
    } else if (x > mesh_x1) {
        ix = Nm - 2;
    } else {
        ix = (x - mesh_x0) / (mesh_x1 - mesh_x0) * double(Nm - 1);
    }

    if (v < mesh_v0) {
        iv = 0;
    } else if (v > mesh_v1) {
        iv = Nm - 2;
    } else {
        iv = (v - mesh_v0) / (mesh_v1 - mesh_v0) * double(Nm - 1);
    }
    int mesh_i_bl = ix + Nm * iv;

    double lengths[2][2];
    double tot_length = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int mesh_i = mesh_i_bl + i + Nm*j;
            double dx = Xs[mesh_i] - x;
            double dv = Vs[mesh_i] - v;
            lengths[i][j] = sqrt(dx*dx + dv*dv);
            tot_length += lengths[i][j];
        }
    }

    // distribute the particle to the vertices of the simplex based on the distance from them
    ret += meshP0[mesh_i_bl] * lengths[0][0];
    ret += meshP0[mesh_i_bl + 1] * lengths[1][0];
    ret += meshP0[mesh_i_bl + Nm] * lengths[0][1];
    ret += meshP0[mesh_i_bl + 1 + Nm] * lengths[1][1];

    return ret / tot_length;
}
/**
 * Compute mesh probabilities, starting from the mesh probabilities of the previous time step
 * @param  {double*} meshP0 : Mesh probabilities of the previous timestep
 * @param  {double*} meshP1 : Output mesh probabilities of the current timestep
 * @param  {double*} Xs    : X linspace
 * @param  {double*} Vs    : V linspace
 * @param  {int} mpiID     :
 * @param  {int} mpiNP     : 
 */
void computeMeshProbabilities(double *meshP0, double *meshP1, double *Xs, double *Vs, int mpiID, int mpiNP) {
    #pragma omp parallel for schedule(static, 1)
    for(int i=mpiID*Nm2/mpiNP; i<(mpiID+1)*Nm2/mpiNP; ++i) {

        if (indexInOmega(i, Xs, Vs)) {
            // particle already in target space Omega, set phi to 1
            meshP1[i] = 1;
            continue;
        }

        double x, v;
        meshP1[i] = 0;
        for (int j = 0; j < n_mc_steps; ++j) {
            particleStepForward(Xs[i], Vs[i], &x, &v);
            meshP1[i] += particlePhiContribution(meshP0, x, v, Xs, Vs);
        }
        meshP1[i] /= n_mc_steps;
    }
}

int main(int argc, char **argv) {

    FILE *out = fopen("out.txt", "w");
    fprintf(out, "\n");

    int mpierr = MPI_Init(&argc, &argv);
    int mpiID = 0, mpiNP = 1;

    mpierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiID);
    mpierr = MPI_Comm_size(MPI_COMM_WORLD, &mpiNP);

	srand(time(NULL));

    // init linspaces and mesh probabilities
    // we only need to store the probabilities for 2 consecutive time steps
    double *Xs = (double*)malloc(sizeof(double) * Nm2);
    double *Vs = (double*)malloc(sizeof(double) * Nm2);
    double *mesh0 = (double*)malloc(sizeof(double) * Nm2);
    double *mesh1 = (double*)malloc(sizeof(double) * Nm2);

    initMeshLinSpace(Xs, Vs);
    initMeshProbabilities(mesh0, Xs, Vs);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int t = 0; t < Nt; ++t) {
        if (mpiID == 0) {
            printf("timestep %i\n", t);
        }
        computeMeshProbabilities(mesh0, mesh1, Xs, Vs, mpiID, mpiNP);

        // sync & shift meshes
        MPI_Allgather(&mesh1[mpiID*Nm2/mpiNP], Nm2/mpiNP, MPI_DOUBLE, mesh0, Nm2/mpiNP, MPI_DOUBLE, MPI_COMM_WORLD);

        // no MPI version
        // memcpy(mesh0, mesh1, sizeof(double)*Nm2);
    } 

    // write to file
    if (mpiID == 0) {
        printf("Writing to file\n");
        int ix, iv;
        for (int i=0; i < Nm2; ++i) {
            ix = i % Nm;
            iv = i / Nm;

            fprintf(out, "%i\t%i\t%i\t%f\n", ix, iv, i, mesh0[i]);
        }
    }

    	
    free(mesh1);
	free(mesh0);
	free(Xs);
	free(Vs);

    mpierr = MPI_Finalize();

    return 0;
}