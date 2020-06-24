#ifndef BMC_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <random>
#include <omp.h>

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
void computeMeshProbabilities(double *meshP0,double *meshP1,double *Xs,double *Vs,Segment *targetSegments,int nTargetSegments,int mpiID,int mpiNP);
double particlePhiContribution(double *meshP0,double x,double v,double *Xs,double *Vs);
void particleStepForward(double x0,double v0,double *x,double *v);
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