#include "bmc5d.hpp"

#define Nm 128 // number of points in each dimension
#define Nm2 (Nm*Nm)
#define mesh_x0 -2.
#define mesh_x1 2.
#define mesh_v0 -2.
#define mesh_v1 2.
#define Nt 100
#define h 1.
#define sigma 0.1
#define subcycles 12
#define hsubsycle (h/subcycles)
#define NHermite 20
#define ForwardMCsteps 1000
#define PI2E3_2 15.7496099457
#define PI2E0_5 2.50662827463
#define USE_FORWARD_MONTECARLO false

std::default_random_engine generator;
std::normal_distribution<double> distribution(0., 1.);

int main(int argc, char **argv) {

    srand(time(NULL));

    FILE *out = fopen("out.txt", "w");
    fprintf(out, "\n");

    #ifdef ENABLEMPI
        int mpierr = MPI_Init(&argc, &argv);
    #endif
    int mpiID = 0, mpiNP = 1;

    #ifdef ENABLEMPI
        mpierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiID);
        mpierr = MPI_Comm_size(MPI_COMM_WORLD, &mpiNP);
    #endif

    // init probability mesh
    dist_5d *meshP = createProbabilityMesh();

    // Compute segments of the target domain Omega
    // Segment *targetSegments;
    // int nTargetSegments = computeTargetDomainSegments(&targetSegments, Xs, Vs);

    // compute Legendre-Hermite polynomials
    double hermiteK[NHermite], hermiteW[NHermite];
    cgqf(NHermite, 6, 0, 0, 0, 0.5, hermiteK, hermiteW);
    struct HermiteParams hermiteParams = {.knots=hermiteK, .weights=hermiteW, .order=NHermite};

    #ifdef ENABLEMPI
        MPI_Barrier(MPI_COMM_WORLD);
    #endif

    // decide whether to use backward or forward
    if (USE_FORWARD_MONTECARLO) {
        forwardMonteCarlo(meshP, mpiID, mpiNP);
    } else {
        backwardMonteCarlo(meshP, hermiteParams, mpiID, mpiNP);
    }

    // write to file
    if (mpiID == 0) {
        printf("Writing to file\n");

        for (int i_r = 0; i_r < meshP->n_r; ++i_r)
        for (int i_theta = 0; i_theta < meshP->n_theta; ++i_theta)
        for (int i_phi = 0; i_phi < meshP->n_phi; ++i_phi)
        for (int i_vpara = 0; i_vpara < meshP->n_vpara; ++i_vpara)
        for (int i_vperp = 0; i_vperp < meshP->n_vperp; ++i_vperp) {
            unsigned long dist5d_index = dist_5D_index(i_r, i_theta, i_phi, i_vpara, i_vperp,
                meshP->n_theta, meshP->n_phi, meshP->n_vpara, meshP->n_vperp);

            fprintf(out, "%i\t%i\t%i\t%i\t%i\t%f\n", i_r, i_theta, i_phi, i_vpara, i_vperp, meshP->histogram[dist5d_index]);
        }

    }

    free(meshP);

    #ifdef ENABLEMPI
        mpierr = MPI_Finalize();
    #endif

    return 0;
}
/** Perform a backward monte carlo computation
 *  and compute the probability mesh after the time Nt
 * 
 * @param  {double*} meshP0               : initial probability mesh
 * @param  {double*} Xs                  : X linspace
 * @param  {double*} Vs                  : V linspace
 * @param  {int} nTargetSegments         : number of target domain segments: len(targetSegments)
 * @param  {Segment*} targetSegments     : target segments array
 * @param  {HermiteParams} hermiteParams : Hermite integration parameters
 * @param  {int} mpiID                   : 
 * @param  {int} mpiNP                   : 
 */
void backwardMonteCarlo(dist_5d *meshP0, HermiteParams hermiteParams, int mpiID, int mpiNP) {

    // we need a support mesh for storing 2 consecutive time steps
    dist_5d *meshP1 = createProbabilityMesh();

    // init probability mesh
    initMeshProbabilities(meshP0);

    for (int t = 0; t < Nt; ++t) {
        if (mpiID == 0) {
            printf("timestep %i\n", t);
        }
        computeMeshProbabilities(meshP0, meshP1, hermiteParams, mpiID, mpiNP);

        // sync & shift meshes
        #ifdef ENABLEMPI
            MPI_Allgather(&meshP1[mpiID*Nm2/mpiNP], Nm2/mpiNP, MPI_DOUBLE, meshP0, Nm2/mpiNP, MPI_DOUBLE, MPI_COMM_WORLD);
        #endif

        // no MPI version
        #ifndef ENABLEMPI
            memcpy(meshP0, meshP1, sizeof(double)*Nm2);
        #endif
    } 

    free(meshP1);
}

/** Perform a forward monte carlo computation
 *  and compute the probability mesh after the time Nt
 * 
 * @param  {double*} meshP               : output probability mesh
 * @param  {double*} Xs                  : X linspace
 * @param  {double*} Vs                  : V linspace
 * @param  {int} nTargetSegments         : number of target domain segments: len(targetSegments)
 * @param  {Segment*} targetSegments     : target segments array
 * @param  {int} mpiID                   : 
 * @param  {int} mpiNP                   : 
 */
void forwardMonteCarlo(dist_5d *meshP, int mpiID, int mpiNP) {
    for(int i_r=mpiID*meshP->n_r/mpiNP; i_r<(mpiID+1)*meshP->n_r/mpiNP; ++i_r)
    for(int i_theta=0; i_theta<meshP->n_theta; ++i_theta)
    for(int i_phi=0; i_phi<meshP->n_phi; ++i_phi)
    for(int i_vpara=0; i_vpara<meshP->n_vpara; ++i_vpara)
    for(int i_vperp=0; i_vperp<meshP->n_vperp; ++i_vperp) {

        unsigned long i = dist_5D_index(i_r, i_theta, i_phi, i_vpara, i_vperp,
            meshP->n_theta, meshP->n_phi, meshP->n_vpara, meshP->n_vperp);
        meshP->histogram[i] = 0;

        if (vertexInTargetDomain(i_r, i_theta, i_phi, i_vpara, i_vperp)) {
            // particle already in target space Omega, set phi to 1
            meshP->histogram[i] = 1;
            continue;
        } else if (vertexOutsideBoundaries(i_r, i_theta, i_phi, i_vpara, i_vperp)) {
            // particle outside of the boundaries or lies on the boundary (but not in the target domain)
            meshP->histogram[i] = 0;
            continue;
        }

        double x0, v0, xi, x1, v1;
        for (int j = 0; j < ForwardMCsteps; ++j) {
            x0 = Xs[i%Nm];
            v0 = Vs[i/Nm];
            for (int t=0; t<Nt; ++t) {
                xi = distribution(generator);

                particleStepForward(x0, v0, xi, &x1, &v1);

                if (coordinatesOutsideBoundaries(x1, v1)) {
                        // printf("lool\n");
                    // the particle is outside the boundary
                    // check if it hit the target domain or not
                    if (trajectoryHitTarget(x0, v0, x1, v1, Xs, Vs, targetSegments, nTargetSegments)) {
                        // printf("hit!\n");
                        meshP[i] += 1.;
                    }
                    break;
                }

                x0 = x1;
                v0 = v1;
            }
        }
        meshP[i] /= ForwardMCsteps;
    }

    #ifdef ENABLEMPI
        MPI_Allgather(&meshP[mpiID*Nm2/mpiNP], Nm2/mpiNP, MPI_DOUBLE, meshP, Nm2/mpiNP, MPI_DOUBLE, MPI_COMM_WORLD);
    #endif
}

dist_5d* createProbabilityMesh() {
    dist_5d ret;
    ret.max_phi = MAX_PHI;
    ret.min_phi = MIN_PHI;
    ret.n_phi = N_PHI;
    ret.max_theta = MAX_THETA;
    ret.min_theta = MIN_THETA;
    ret.n_theta = N_THETA;
    ret.max_r = MAX_R;
    ret.min_r = MIN_R;
    ret.n_r = N_R;

    ret.max_vpara = MAX_VPARA;
    ret.min_vpara = MIN_VPARA;
    ret.n_vpara = N_VPARA;
    ret.max_vperp = MAX_VPERP;
    ret.min_vperp = MIN_VPERP;
    ret.n_vperp = N_VPERP;

    ret.histogram = (real*)malloc(sizeof(real) * (ret.n_r * ret.n_theta * ret.n_phi * ret.n_vpara * ret.n_vperp));

    return &ret;
}

/**
 * initialize the probabilities with the indicator function of omega
 *
 * @param  {double*} meshP : output probabilities for the mesh
 * @param  {double*} Xs   : 
 * @param  {double*} Vs   : 
 */
void initMeshProbabilities(dist_5d *meshP) {
    for (int i = 0; i < Nm2; ++i) {
        if (vertexInTargetDomain(i, Xs, Vs)) {
            meshP[i] = 1.;
        } else {
            meshP[i] = 0.;
        }
    }
}

unsigned long dist_5D_index(int i_r, int i_phi, int i_z, int i_vpara,
                            int i_vperp, int n_phi,
                            int n_z, int n_vpara, int n_vperp) {
    return i_r    * (n_phi * n_z * n_vpara * n_vperp)
        + i_phi   * (n_z * n_vpara * n_vperp)
        + i_z     * (n_vpara * n_vperp)
        + i_vpara * (n_vperp)
        + i_vperp;
}

// Given three colinear points p, q, r, the function checks if 
// point q lies on line segment 'pr' 
bool onSegment(Point p, Point q, Point r) 
{ 
    if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) && 
        q.v <= std::max(p.v, r.v) && q.v >= std::min(p.v, r.v)) 
       return true; 
  
    return false; 
}

// To find orientation of ordered triplet (p, q, r). 
// The function returns following values 
// 0 --> p, q and r are colinear 
// 1 --> Clockwise 
// 2 --> Counterclockwise 
int orientation(Point p, Point q, Point r) 
{ 
    // See https://www.geeksforgeeks.org/orientation-3-ordered-points/ 
    // for details of below formula. 
    int val = (q.v - p.v) * (r.x - q.x) - 
              (q.x - p.x) * (r.v - q.v); 
  
    if (val == 0) return 0;  // colinear 
  
    return (val > 0)? 1: 2; // clock or counterclock wise 
}

/**
 * Check if 2 segments p1q1 and p2q2 intersect
 * @param  {Point} p1 : 
 * @param  {Point} q1 : 
 * @param  {Point} p2 : 
 * @param  {Point} q2 : 
 * @return {bool}     : 
 */
bool doIntersect(Point p1, Point q1, Point p2, Point q2) 
{ 
    // Find the four orientations needed for general and 
    // special cases 
    int o1 = orientation(p1, q1, p2); 
    int o2 = orientation(p1, q1, q2); 
    int o3 = orientation(p2, q2, p1); 
    int o4 = orientation(p2, q2, q1); 
  
    // General case 
    if (o1 != o2 && o3 != o4) 
        return true; 
  
    // Special Cases 
    // p1, q1 and p2 are colinear and p2 lies on segment p1q1 
    if (o1 == 0 && onSegment(p1, p2, q1)) return true; 
  
    // p1, q1 and q2 are colinear and q2 lies on segment p1q1 
    if (o2 == 0 && onSegment(p1, q2, q1)) return true; 
  
    // p2, q2 and p1 are colinear and p1 lies on segment p2q2 
    if (o3 == 0 && onSegment(p2, p1, q2)) return true; 
  
     // p2, q2 and q1 are colinear and q1 lies on segment p2q2 
    if (o4 == 0 && onSegment(p2, q1, q2)) return true; 
  
    return false; // Doesn't fall in any of the above cases 
}

/** Init phase space (x-v) linspace for the rectangular mesh
 *  The mesh is defined as [mesh_x0, mesh_x1] X [mesh_v0, mesh_v1]
 * 
 * @param  {double*} Xs : X output linspace
 * @param  {double*} Vs : V output linspace
 */
void initMeshLinSpace(double *Xs, double *Vs) {
    double x, y;
    int ix;
    for (int i = 0; i < Nm; ++i) {
        Xs[i] = i / (Nm - 1.) * (mesh_x1 - mesh_x0) + mesh_x0;
        Vs[i] = i / (Nm - 1.) * (mesh_v1 - mesh_v0) + mesh_v0;
    }
}

/**
 *  Check if point i is inside the target domain Omega
 *  Omega is set to be at x = mesh_x0 and v \in [-0.5, 0]
 * @param  {int} i      : index of the vertex
 * @param  {double*} Xs : 
 * @param  {double*} Vs : 
 * @return {bool}       : 
 */
bool vertexInTargetDomain(int i_r, int i_theta, int i_phi, int i_vpara, int i_vperp) {
    double v = Vs[i/Nm];
    double x = Xs[i&Nm];
    
    if ((i % Nm == 0) && (v >= -0.5) && (v <= 0)) {
        return true;
    }
    // if ((i/Nm == 22) && (x < -1.7)) {
    //     return true;
    // }
    return false;
}

/**
 * Compute the segments of the target boundary
 * Try to compute the target segments starting from the target points.
 * This function is quite awkward and it will be replaced with a proper function from ASCOT
 * A better approach would be to compute directly the segments, either by hand..
 * ...or with some coordinate parametrization and discretization
 * @param  {Segment**} targetSegments : output array of segments
 * @param  {double*} Xs : 
 * @param  {double*} Vs : 
 * @return {int}        : number of target segments
 */
int computeTargetDomainSegments(Segment **targetSegments, double *Xs, double *Vs) {

    *targetSegments = (Segment *)malloc(0);
    int ret = -1;
    int p0;
    struct Segment s;
    for (int i=0; i<Nm2; ++i) {
        if (vertexInTargetDomain(i, Xs, Vs)) {
            ret++;
            if (ret > 0) {
                s = {p0, i};
                *targetSegments = (Segment *)realloc(*targetSegments, ret * sizeof(Segment));
                (*targetSegments)[ret-1] = s;
            }
            p0 = i;
        }
    }
    return ret;
}

/**
 * Check if the input x-v cooedinates are outside the boundary
 * @param  {double} x   : 
 * @param  {double} v   : 
 * @param  {double*} Xs : X linspace
 * @param  {double*} Vs : V linspace
 * @return {bool}       : 
 */
bool coordinatesOutsideBoundaries(double x, double v) {
    if ((x <= mesh_x0) || (x >= mesh_x1)) {
        return true;
    }
    if ((v <= mesh_v0) || (v >= mesh_v1)) {
        return true;
    }

    // TEST NEW DOMAIN
    if ((x>1.5) && (v>-0.5) && (v<0.5)) {
        return true;
    }
    if ((v<-1.7) && (x>-0.5) && (x<0.5)) {
        return true;
    }
    // if ((x<-1.7) && (v>-0.5) && (v<=0)) {
    //     return true;
    // }

    return false;
}

/**
 *  Check if point i is outside or on the boundary
 * @param  {int} i      : index of the vertex
 * @param  {double*} Xs : 
 * @param  {double*} Vs : 
 * @return {bool}       : 
 */
bool vertexOutsideBoundaries(int i_r, int i_theta, int i_phi, int i_vpara, int i_vperp) {
    return coordinatesOutsideBoundaries(Xs[i%Nm], Vs[i/Nm]);
}

/**
 * Check if a trajectory between 2 phase-space points hit the target domain
 * @param  {double} x0  : 
 * @param  {double} v0  : 
 * @param  {double} x1  : 
 * @param  {double} v1  : 
 * @param  {double*} Xs : X linspace
 * @param  {double*} Vs : V linspace
 * @return {bool}       : 
 */
bool trajectoryHitTarget(double x0, double v0, double x1, double v1, double *Xs, double *Vs, Segment *targetSegments, int nTargetSegments) {
    struct Point p0, p1, t0, t1;
    int t0_idx, t1_idx;
    for (int i=0; i < nTargetSegments; ++i) {
        int u = i;
        // loop in all target domain segments and check if the trajectory hit the segment
        p0 = {x0, v0};
        p1 = {x1, v1};
        t0_idx = targetSegments[i].p0;
        t1_idx = targetSegments[i].p1;
        t0 = {Xs[t0_idx % Nm], Vs[t0_idx / Nm]};
        t1 = {Xs[t1_idx % Nm], Vs[t1_idx / Nm]};

            // printf("%f\n", ((p1.x - p0.x) * (t1.v - t0.v) - (p1.v - p0.v) * (t1.x - t0.x)));
        if (doIntersect(p0, p1, t0, t1)) {
            // check if the trajectory is entering or exiting the target domain
            if (((p1.x - p0.x) * (t1.v - t0.v) - (p1.v - p0.v) * (t1.x - t0.x)) <= 0) {
                // exiting
                return true;
            }
        }
    }
    return false;
}


/**
 * Advance the particle in time using symplectic Euler with subcycles
 * @param  {double} x0 : 
 * @param  {double} v0 : 
 * @param  {double} xi : 
 * @param  {double*} x : 
 * @param  {double*} v : 
 */
void particleStepForward(double x0, double v0, double xi, double *x, double *v) {
    // FIXME: Fix the random number generator
    // std::random_device rd{};
    // std::mt19937 gen{rd()};
    for (int i = 0; i < subcycles; ++i) {
        // hamiltonian push
        *x = x0 * (1 - hsubsycle*hsubsycle) + hsubsycle * v0;
        *v = v0 - hsubsycle * x0;

        if (coordinatesOutsideBoundaries(*x, *v)) {
            // the particle went outside the boundary in the subcycle. Exit
            return;
        }
    }

    // update v with maryama euler
    *v += sigma * sqrt(h) * xi;
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
    int ix = (x - mesh_x0) / (mesh_x1 - mesh_x0) * double(Nm - 1);
    int iv = (v - mesh_v0) / (mesh_v1 - mesh_v0) * double(Nm - 1);
    int mesh_i_bl = ix + Nm * iv;

    // find distance of the point to each of the 4 vertices
    double lengths[2][2], tot_length = 0, dx, dv;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            dx = Xs[ix + i] - x;
            dv = Vs[iv + j] - v;
            lengths[i][j] = sqrt(dx*dx + dv*dv);
            tot_length += lengths[i][j];
        }
    }

    // distribute the particle to the vertices of the simplex based on the distance from them
    // and weight with the probability of the previous time step
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
void computeMeshProbabilities(dist_5d *meshP0, dist_5d *meshP1, HermiteParams hermiteParams, int mpiID, int mpiNP) {
    #pragma omp parallel for schedule(static, 1)
    for(int i=mpiID*Nm2/mpiNP; i<(mpiID+1)*Nm2/mpiNP; ++i) {

        meshP1[i] = 0;
        double x1, v1;
        int ix = i%Nm, iv = i/Nm;

        if (vertexInTargetDomain(i, Xs, Vs)) {
            // particle already in target space Omega, set phi to 1
            meshP1[i] = 1;
            continue;
        } else if (vertexOutsideBoundaries(i, Xs, Vs)) {
            // particle outside of the boundaries or lies on the boundary (but not in the target domain)
            continue;
        }

        for (int j = 0; j < NHermite; ++j) {
            particleStepForward(Xs[ix], Vs[iv], hermiteParams.knots[j], &x1, &v1);

            if (coordinatesOutsideBoundaries(x1, v1)) {
                // the particle is outside the boundary
                // check if it hit the target domain or not
                if (trajectoryHitTarget(Xs[ix], Vs[iv], x1, v1, Xs, Vs, targetSegments, nTargetSegments)) {
                    meshP1[i] += hermiteParams.weights[j];
                }
            } else {
                // particle still inside the boundary.
                // weight with the probability of the previous timestep
                meshP1[i] += particlePhiContribution(meshP0, x1, v1, Xs, Vs) * hermiteParams.weights[j];
            }

        }
        meshP1[i] /= PI2E0_5;
    }
}
