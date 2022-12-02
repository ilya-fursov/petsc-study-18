/*
 * ex_DM.cpp
 *
 *  Created on: 28 Mar 2017
 *      Author: ilya
 */


static char help[] = "Testing how 2D grid is partitioned, and how presence of a well affects it\n"
		"Options:\n"
		"-M <10> : number of grid blocks in X\n"
		"-N <10> : number of grid blocks in Y\n"
		"-W <0>  : number of cells in 1/2 well along X\n"
		"-w_angle <0> : well azimuth\n";



//#include <petscdm.h>
#include <cmath>
#include <vector>
#include <iostream>

#include <petscdmda.h>
#include "petscdraw.h"
#include "petscviewer.h"
#include "petscksp.h"
//#include <petscsnes.h>


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
	PetscInitialize(&argc,&argv,(char*)0,help);
	PetscInt M = 10, N = 10, xs, ys, xm, ym;
	PetscInt Sz, Sz0, W = 0, dnz, onz;
	PetscScalar **array, w_angle = 0;
	int rank, size;

	PetscErrorCode ierr;
	DM da;
	Vec x, y, z;
	Mat A;


	ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);
	ierr  = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
	ierr  = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL); CHKERRQ(ierr);
																						// well goes through the grid center
	ierr  = PetscOptionsGetInt(NULL,NULL,"-W",&W,NULL); CHKERRQ(ierr);					// length of 1/2 well in cells (along X)
																						// W==0 : no well
	ierr  = PetscOptionsGetReal(NULL,NULL,"-w_angle",&w_angle,NULL); CHKERRQ(ierr);		// well azimuth, degrees

	// partitioning from DMDA
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, N, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &da); CHKERRQ(ierr);
	PetscCall(DMSetUp(da));
	ierr = DMCreateGlobalVector(da, &x); CHKERRQ(ierr);
	ierr = VecDuplicate(x, &z); CHKERRQ(ierr);
	ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, x, &array); CHKERRQ(ierr);

	for (int i = xs; i < xs+xm; i++)
		for (int j = ys; j < ys+ym; j++)
		{
			array[j][i] = rank;
		}

	ierr = DMDAVecRestoreArray(da, x, &array); CHKERRQ(ierr);
	ierr = VecView(x, PETSC_VIEWER_DRAW_WORLD); CHKERRQ(ierr);
	PetscCall(PetscSleep(3));

	// partitioning from Mat, METIS
	// a) create system's matrix
	dnz = PetscMax(5, 2*W);
	onz = PetscMax(2, 2*W);
	Sz = M*N + (W > 0 ? 1:0);		// + 1 component for well
	Sz0 = M*N;						// only grid part
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Matrix size is %d x %d\n", Sz, Sz); CHKERRQ(ierr);
	ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, Sz, Sz, dnz, 0, onz, 0, &A); CHKERRQ(ierr);
	std::vector<int> well;			// well coords
	if (rank == 0)
	{
		for (int Ii = 0; Ii < Sz0; Ii++)
		{
			int i = Ii%M;
			int j = Ii/M;
			PetscScalar v0 = 4, v1 = -1;
			if (i-1 >= 0)
			{
				int J = Ii-1;
				ierr = MatSetValues(A, 1, &Ii, 1, &J, &v1, INSERT_VALUES); CHKERRQ(ierr);
			}
			else
				v0 -= 1;
			if (i+1 < M)
			{
				int J = Ii+1;
				ierr = MatSetValues(A, 1, &Ii, 1, &J, &v1, INSERT_VALUES); CHKERRQ(ierr);
			}
			else
				v0 -= 1;
			if (j-1 >= 0)
			{
				int J = Ii-M;
				ierr = MatSetValues(A, 1, &Ii, 1, &J, &v1, INSERT_VALUES); CHKERRQ(ierr);
			}
			else
				v0 -= 1;
			if (j+1 < N)
			{
				int J = Ii+M;
				ierr = MatSetValues(A, 1, &Ii, 1, &J, &v1, INSERT_VALUES); CHKERRQ(ierr);
			}
			else
				v0 -= 1;
			ierr = MatSetValues(A, 1, &Ii, 1, &Ii, &v0, INSERT_VALUES); CHKERRQ(ierr);
		}
		for (int Ii = Sz0; Ii < Sz; Ii++)
		{
			PetscScalar v0 = 2*W-1;					// well component
			ierr = MatSetValues(A, 1, &Ii, 1, &Ii, &v0, INSERT_VALUES); CHKERRQ(ierr);
		}
		// insert more well terms
		for (int k = -W+1; k < W; k++)
		{
			int i = M/2 + k;
			int j = N/2 + int(tan(w_angle/180*3.1415)*k);
			int Ii = i + j*M;
			well.push_back(Ii);
			PetscScalar v1 = -1;
			ierr = MatSetValues(A, 1, &Sz0, 1, &Ii, &v1, INSERT_VALUES); CHKERRQ(ierr);	// Sz0 - last row
			ierr = MatSetValues(A, 1, &Ii, 1, &Sz0, &v1, INSERT_VALUES); CHKERRQ(ierr);	// Sz0 - last col
		}
	}
	ierr = MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);
	if (rank == 0)
	{
		for (int k = -W+1; k < W; k++)
		{
			int i = M/2 + k;
			int j = N/2 + int(tan(w_angle/180*3.1415)*k);
			int Ii = i + j*M;
			PetscScalar v0 = 1;
			ierr = MatSetValues(A, 1, &Ii, 1, &Ii, &v0, ADD_VALUES); CHKERRQ(ierr);
		}
	}
	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatSetFromOptions(A); CHKERRQ(ierr);

	// PARTITIONING
	MatPartitioning part;
	IS is, isnum, is2sided;
	const PetscInt *ind;
	PetscScalar *arr;
	PetscInt issize, vecsize;
	AO ao;
	PetscViewer ascii;
	ierr = DMDAGetAO(da, &ao); CHKERRQ(ierr);
	ierr = MatPartitioningCreate(PETSC_COMM_WORLD, &part); CHKERRQ(ierr);
	ierr = MatPartitioningSetAdjacency(part, A); CHKERRQ(ierr);
	ierr = MatPartitioningSetFromOptions(part); CHKERRQ(ierr);
	ierr = MatPartitioningApply(part, &is); CHKERRQ(ierr);

	//------------------------------------------------------------------------------------------
	ierr = ISPartitioningToNumbering(is,&isnum);CHKERRQ(ierr);		// output index sets to file
	ierr = ISBuildTwoSided(is,NULL,&is2sided);CHKERRQ(ierr);
	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "output_ex_DM_is.txt", &ascii); CHKERRQ(ierr);
	ierr = PetscViewerPushFormat(ascii, PETSC_VIEWER_DEFAULT); CHKERRQ(ierr);
	ierr = ISView(is, ascii); CHKERRQ(ierr);
	ierr = ISView(isnum, ascii); CHKERRQ(ierr);
	ierr = ISView(is2sided, ascii); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&ascii); CHKERRQ(ierr);

	ierr = MatCreateVecs(A, &y, NULL); CHKERRQ(ierr);
	ierr = VecGetLocalSize(y, &vecsize); CHKERRQ(ierr);
	ierr = ISGetLocalSize(is, &issize); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Vec size = %d, IS size = %d\n", rank, vecsize, issize); CHKERRQ(ierr);
	// looks like Vec and IS are distributed in the same manner

	// FILL y with processor number
	ierr = ISGetIndices(is, &ind); CHKERRQ(ierr);
	ierr = VecGetArray(y, &arr); CHKERRQ(ierr);
	if (vecsize != issize)
		SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Vec and IS local sizes mismatch");
	for (int i = 0; i < issize; i++)
		arr[i] = ind[i];

	ierr = ISRestoreIndices(is, &ind); CHKERRQ(ierr);
	ierr = VecRestoreArray(y, &arr); CHKERRQ(ierr);
	//ierr = VecView(y, PETSC_VIEWER_DRAW_WORLD); CHKERRQ(ierr);	TODO can be uncommented

	// assign y to z
	int i0, i1;
	bool has_well = false;
	ierr = VecGetOwnershipRange(y, &i0, &i1); CHKERRQ(ierr);
	if (i1 > Sz0)
	{
		i1 = Sz0;		// remove the well component
		has_well = true;
	}
	int *idx = new int[i1-i0];
	for (int i = i0; i < i1; i++)
		idx[i-i0] = i;
	ierr = AOApplicationToPetsc(ao, i1-i0, idx); CHKERRQ(ierr);
	ierr = AOApplicationToPetsc(ao, well.size(), well.data()); CHKERRQ(ierr);	// transform well coords

	PetscScalar well_proc;		// y value for the well component
	ierr = VecGetArray(y, &arr); CHKERRQ(ierr);
	ierr = VecSetValues(z, i1-i0, idx, arr, INSERT_VALUES); CHKERRQ(ierr);
	ierr = VecAssemblyBegin(z); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(z); CHKERRQ(ierr);
	if (has_well)
	{
		well_proc = arr[i1-i0];
		ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] well component processor number: %g\n", rank, well_proc); CHKERRQ(ierr);
	}
	if (rank == 0 && well.size() > 0)
	{
		// set the grid values along the well (for viewing)
		ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] adding %zu values as a well trajectory\n", rank, well.size()); CHKERRQ(ierr);
		std::vector<double> vals(well.size(), 5);
		ierr = VecSetValues(z, well.size(), well.data(), vals.data(), ADD_VALUES); CHKERRQ(ierr);
	}
	ierr = VecAssemblyBegin(z); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(z); CHKERRQ(ierr);
	ierr = VecRestoreArray(y, &arr); CHKERRQ(ierr);
	ierr = VecView(z, PETSC_VIEWER_DRAW_WORLD); CHKERRQ(ierr);
	PetscCall(PetscSleep(3));

	ierr = MatPartitioningDestroy(&part); CHKERRQ(ierr);
	ierr = MatDestroy(&A); CHKERRQ(ierr);
	ierr = VecDestroy(&x); CHKERRQ(ierr);
	ierr = VecDestroy(&y); CHKERRQ(ierr);
	ierr = VecDestroy(&z); CHKERRQ(ierr);
	delete [] idx;

	ierr = PetscFinalize();

	return 0;
}

