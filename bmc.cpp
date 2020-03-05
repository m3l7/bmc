#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <mpi.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <random>
// #include <omp.h>

#define Nm 100 // number of points in each dimension
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

void initMeshIndexes(double *Xs, double *Vs) {
    double x, y;
    int ix;
    for (int i = 0; i < Nm2; ++i) {
        Xs[i] = (i % Nm) / (Nm - 1.) * (mesh_x1 - mesh_x0) + mesh_x0;
        Vs[i] = trunc(i / Nm) / (Nm - 1.) * (mesh_v1 - mesh_v0) + mesh_v0;
    }
}

bool indexInOmega(int i, double *Xs, double *Vs) {
    if ((i % Nm == 0) && (Vs[i] >= -0.5) && (Vs[i] <= 0)) {
        return true;
    } else {
        return false;
    }
}

// initialize the probabilities with the indicator function of omega
void initMeshProbabilities(double *mesh, double *Xs, double *Vs) {
    for (int i = 0; i < Nm2; ++i) {
        if (indexInOmega(i, Xs, Vs)) {
            mesh[i] = 1.;
        } else {
            mesh[i] = 0.;
        }
    }
}



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

void particleMeshDeposit(double *meshTmp, double x, double v, double *Xs, double *Vs) {

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

    double lengths[2][2];
    double tot_length = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int mesh_i = ix + i + Nm * (j + iv);
            double dx = Xs[mesh_i] - x;
            double dv = Vs[mesh_i] - v;
            lengths[i][j] = sqrt(dx*dx + dv*dv);
            tot_length += lengths[i][j];
        }
    }

    // distribute the particle to the vertices of the simplex based on the distance from them
    int mesh_i_bl = ix + Nm * iv;
    meshTmp[mesh_i_bl] += lengths[0][0] / tot_length;
    meshTmp[mesh_i_bl + 1] += lengths[0][1] / tot_length;
    meshTmp[mesh_i_bl + Nm] += lengths[1][0] / tot_length;
    meshTmp[mesh_i_bl + 1 + Nm] += lengths[1][1] / tot_length;
}

void computeMeshProbabilities(double *mesh0, double *mesh1, double *meshTmp, double *Xs, double *Vs) {
    for (int i = 0; i < Nm2; ++i) {

        if (indexInOmega(i, Xs, Vs)) {
            // particle already on omega, set phi to 1
            mesh1[i] = 1;
            continue;
        }

        // erase the mesh helper tmp
        for (int j = 0; j < Nm2; ++j)
            meshTmp[j] = 0.;

        double x, v;
        mesh1[i] = 0;
        for (int j = 0; j < n_mc_steps; ++j) {
            particleStepForward(Xs[i], Vs[i], &x, &v);
            // if (i == 1399) {
            //     printf("%f %f %f %f\n", Xs[i], Vs[i], x, v);
            // }
            particleMeshDeposit(meshTmp, x, v, Xs, Vs);
            // if (i == 1399) {
            //     printf("mc %i %f\n", j, meshTmp[1150]);
            // }
        }

        // normalize mesh and multiply by previous timestep mesh
        // then sum all elements and save to new mesh1
        for (int j = 0; j < Nm2; ++j) {
            mesh1[i] += meshTmp[j] / n_mc_steps * mesh0[j];
            // if (i == 1399) {
            //     printf("%i %f\n", j, mesh1[i]);
            // }
        }
    }

}

int main() {

    FILE *out = fopen("out.txt", "w");
    fprintf(out, "\n");

    srand(time(NULL));

    // init mesh
    double *Xs = (double*)malloc(sizeof(double) * Nm2);
    double *Vs = (double*)malloc(sizeof(double) * Nm2);
    double *mesh0 = (double*)malloc(sizeof(double) * Nm2);
    double *mesh1 = (double*)malloc(sizeof(double) * Nm2);
    double *mesh_tmp = (double*)malloc(sizeof(double) * Nm2);
    double *mesh_shift;

    initMeshIndexes(Xs, Vs);
    initMeshProbabilities(mesh0, Xs, Vs);

    for (int t = 0; t < Nt; ++t) {
        printf("timestep %i\n", t);
        computeMeshProbabilities(mesh0, mesh1, mesh_tmp, Xs, Vs);

        // shift meshes
        memcpy(mesh0, mesh1, sizeof(double)*Nm2);
    } 

    // TODO write to file
    printf("Writing to file\n");
    int ix, iv;
    for (int i=0; i < Nm2; ++i) {
        ix = i % Nm;
        iv = i / Nm;

        fprintf(out, "%i\t%i\t%i\t%f\n", ix, iv, i, mesh0[i]);
    }

    return 0;
}