#ifndef BMC_HPP

#ifdef ENABLEMPI
    #include <mpi.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>
#include <random>
#include "hermite_rule.hpp"

typedef double real;  /**< Double precision float   */
typedef long integer; /**< Double precision integer */
typedef unsigned long int a5err;

typedef struct {
    int n_r;        /**< number of r bins       */
    real min_r;     /**< value of lowest r bin  */
    real max_r;     /**< value of highest r bin */

    int n_theta;      /**< number of poloidal angle bins */
    real min_theta;   /**< value of lowest pol bin       */
    real max_theta;   /**< value of highest pol bin      */

    int n_phi;        /**< number of phi bins       */
    real min_phi;     /**< value of lowest phi bin  */
    real max_phi;     /**< value of highest phi bin */

    int n_vpara;      /**< number of v_parallel bins       */
    real min_vpara;   /**< value of lowest v_parallel bin  */
    real max_vpara;   /**< value of highest v_parallel bin */

    int n_vperp;      /**< number of v_perpendicular bins       */
    real min_vperp;   /**< value of lowest v_perpendicular bin  */
    real max_vperp;   /**< value of highest v_perpendicular bin */

    // int n_time;       /**< number of time bins       */
    // real min_time;    /**< value of lowest time bin  */
    // real max_time;    /**< value of highest time bin */

    // int n_q;          /**< number of charge bins       */
    // real min_q;       /**< value of lowest charge bin  */
    // real max_q;       /**< value of highest charge bin */

    real* histogram;  /**< pointer to start of histogram array */
} dist_5d;

typedef struct {
    /* Physical coordinates and parameters */
    real r;       /**< Particle R coordinate [m]          */
    real phi;     /**< Particle phi coordinate [phi]      */
    real z;       /**< Particle z coordinate [m]          */
    real rdot;    /**< dr/dt [m/s]                        */
    real phidot;  /**< dphi/dt [rad/s]                    */
    real zdot;    /**< dz/dt [m/s]                        */
    real mass;    /**< Mass [kg]                          */
    real charge;  /**< Charge [C]                         */
    real time;    /**< Marker simulation time [s]         */

    /* Magnetic field data */
    real B_r;        /**< Magnetic field R component at
                                              marker position [T]             */
    real B_phi;      /**< Magnetic field phi component at
                                              marker position [T]             */
    real B_z;        /**< Magnetic field z component at
                                              marker position [T]             */

    real B_r_dr;     /**< dB_R/dR at marker position [T/m]     */
    real B_phi_dr;   /**< dB_phi/dR at marker position [T/m]   */
    real B_z_dr;     /**< dB_z/dR at marker position [T/m]     */
    real B_r_dphi;   /**< dB_R/dphi at marker position [T/m]   */
    real B_phi_dphi; /**< dB_phi/dphi at marker position [T/m] */
    real B_z_dphi;   /**< dB_z/dphi at marker position [T/m]   */
    real B_r_dz;     /**< dB_R/dz at marker position [T/m]     */
    real B_phi_dz;   /**< dB_phi/dz at marker position [T/m]   */
    real B_z_dz;     /**< dB_z/dz at marker position [T/m]     */

    /* Quantities used in diagnostics */
    real weight;  /**< Marker weight                      */
    real cputime; /**< Marker wall-clock time [s]         */
    real rho;     /**< Marker rho coordinate              */
    real theta;   /**< Marker poloidal coordinate [rad]   */

    integer id;       /**< Unique ID for the marker       */
    integer endcond;  /**< Marker end condition           */
    integer walltile; /**< ID of walltile if marker has
                                               hit the wall                   */

    /* Meta data */
    integer running; /**< Indicates whether this marker is
                                              currently simulated (1) or not  */
    a5err err;       /**< Error flag, zero if no error    */
    integer index;   /**< Marker index at marker queue    */
} particle_fo;

struct HermiteParams {
    double* knots;
    double* weights;
    int order;
};
struct Segment {
    int p0;
    int p1;
};

struct Point 
{ 
    double x; 
    double v; 
};

int main(int argc,char **argv);
void backwardMonteCarlo(double *meshP0, double *Xs, double *Vs, int nTargetSegments, Segment *targetSegments, HermiteParams hermiteParams, int mpiID, int mpiNP);
void forwardMonteCarlo(double *meshP, double *Xs, double *Vs, int nTargetSegments, Segment *targetSegments, int mpiID, int mpiNP);
void computeMeshProbabilities(double *meshP0, double *meshP1, double *Xs, double *Vs, Segment *targetSegments, int nTargetSegments, HermiteParams hermiteParams, int mpiID, int mpiNP);
double particlePhiContribution(double *meshP0,double x,double v,double *Xs,double *Vs);
void particleStepForward(double x0,double v0,double xi, double *x,double *v);
void initMeshProbabilities(double *meshP,double *Xs,double *Vs);
bool trajectoryHitTarget(double x0,double v0,double x1,double v1,double *Xs,double *Vs,Segment *targetSegments,int nTargetSegments);
bool vertexOutsideBoundaries(int i,double *Xs,double *Vs);
bool coordinatesOutsideBoundaries(double x,double v);
int computeTargetDomainSegments(Segment **targetSegments,double *Xs,double *Vs);
bool vertexInTargetDomain(int i,double *Xs,double *Vs);
void initMeshLinSpace(double *Xs,double *Vs);
bool doIntersect(Point p1,Point q1,Point p2,Point q2);
int orientation(Point p,Point q,Point r);
bool onSegment(Point p,Point q,Point r);

#endif