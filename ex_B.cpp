
static char help[] = "Stationary diffusivity + advection, using DMDA\n"
		"Options:\n"
		"-N <10> : number of grid blocks in each dimension\n"
		"-T <1> : number of time steps in advection\n"
		"-save <1> : save every *th step\n"
		"-dt <0.01> : time step length in advection\n"
		"-p_ : options for diffusivity ksp\n"
		"-sw_ : options for advection ksp\n";


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

	PetscErrorCode ierr;
	int rank, size, N = 10, T = 1, save = 1, xs, ys, xm, ym, itnum, N2;
	MatStencil row, col[5];
	PetscScalar val[5], dt = 0.01;
	PetscReal norm, norminf, norm1, swsum;

	DM da;
	PetscViewer vtk;
	KSP ksp;
	Mat A, B;	// for diffusivity and advection respectively
	Vec x, b;	// diffusivity
	Vec y, c;	// advection
	Vec resid, xloc, act, act0;
	double **array, **arract;

	PetscPreLoadBegin(PETSC_FALSE, "main section");

	// Get rank, size, options
	ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);
	ierr  = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
	ierr  = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetInt(NULL,NULL,"-T",&T,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetInt(NULL,NULL,"-save",&save,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetScalar(NULL,NULL,"-dt",&dt,NULL); CHKERRQ(ierr);
	N2 = N*N;

	ierr = PetscPrintf(PETSC_COMM_WORLD, "Grid size: %d x %d, advection steps: %d, advection step size: %g\n", N, N, T, (double)dt); CHKERRQ(ierr);
	ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, "output_B_diff.vts", FILE_MODE_WRITE, &vtk); CHKERRQ(ierr);

	PetscReal h = 1.0/double(N);
	PetscReal h2 = h*h;
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, N, N, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &da); CHKERRQ(ierr);
	PetscCall(DMSetUp(da));
	ierr = DMDASetUniformCoordinates(da, 0.0, 1, 0.0, 1, 0.0, 0.0); CHKERRQ(ierr);

	// ------------------------ DIFFUSIVITY -----------------------
	PetscPreLoadStage("Diffusivity");

	ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);
	//ierr = MatCreateVecs(A, &x, &b); CHKERRQ(ierr); 		// this didn't work for 2D array viewing
	ierr = DMCreateGlobalVector(da, &x); CHKERRQ(ierr);
	ierr = VecDuplicate(x, &b); CHKERRQ(ierr);
	ierr = VecDuplicate(b, &resid); CHKERRQ(ierr);
	ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0); CHKERRQ(ierr);
	val[0] = val[1] = val[2] = val[3] = -1;

	// act0, act indicate which cells are active
	ierr = VecDuplicate(b, &act0); CHKERRQ(ierr);			// act0 - global vec
	ierr = DMCreateLocalVector(da, &act); CHKERRQ(ierr);	// act - local vec
	ierr = VecSet(act0, 0); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, act0, &arract); CHKERRQ(ierr);
	for (int i = xs; i < xs+xm; i++)
		for (int j = ys; j < ys+ym; j++)
		{
			if (((i-N/6)*(i-N/6) + (j-N/5)*(j-N/5))*80 <= N2)
				arract[j][i] = 1;

			if (((i-3*N/8)*(i-3*N/8) + (j-3*N/10)*(j-3*N/10))*90 <= N2)
				arract[j][i] = 1;
		}
	ierr = DMDAVecRestoreArray(da, act0, &arract); CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da, act0, INSERT_VALUES, act); CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da, act0, INSERT_VALUES, act); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, act, &arract); CHKERRQ(ierr);

	ierr = VecSet(x, 0); CHKERRQ(ierr);
	ierr = VecSet(b, 0); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, x, &array); CHKERRQ(ierr);
	for (int i = xs; i < xs+xm; i++)
		for (int j = ys; j < ys+ym; j++)
		{
			if (arract[j][i] == 1)	// inactive cell
			{
				row.i = i;
				row.j = j;
				val[4] = 1;
				array[j][i] = 0;
				ierr  = MatSetValuesStencil(A, 1, &row, 1, &row, val+4, INSERT_VALUES); CHKERRQ(ierr);
			}
			else
			{
				row.i = i;
				row.j = j;
				col[0].i = i-1;
				col[0].j = j;
				col[1].i = i;
				col[1].j = j-1;

				col[2].i = i+1;
				col[2].j = j;
				col[3].i = i;
				col[3].j = j+1;

				col[4].i = i;
				col[4].j = j;
				val[4] = 4;

				if (i-1 < 0)		// handle boundaries
					val[4] -= 1;
				if (j-1 < 0)
					val[4] -= 1;
				if (i+1 >= N)
				{
					col[2].i = -1;	// above the same feature is activated automatically
					val[4] -= 1;
				}
				if (j+1 >= N)
				{
					col[3].j = -1;
					val[4] -= 1;
				}

				for (int k = 0; k < 4; k++)
					if (col[k].j != -1 && col[k].i != -1 && arract[col[k].j][col[k].i] == 1)
					{
						col[k].i = -1;
						col[k].j = -1;
						val[4] -= 1;
					}

				ierr  = MatSetValuesStencil(A, 1, &row, 5, col, val, INSERT_VALUES); CHKERRQ(ierr);
			}
		}

	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	
	// set solution at 5 points
	if (xs <= 0 && 0 < xs+xm && ys <= 0 && 0 < ys+ym)			// check if index belongs to the proc
		array[0][0] = -1;
	if (xs <= N-1 && N-1 < xs+xm && ys <= N-1 && N-1 < ys+ym)
		array[N-1][N-1] = -1;
	if (xs <= 0 && 0 < xs+xm && ys <= N-1 && N-1 < ys+ym)
		array[N-1][0] = -1;
	if (xs <= N-1 && N-1 < xs+xm && ys <= 0 && 0 < ys+ym)
		array[0][N-1] = -1;
	if (xs <= N/2 && N/2 < xs+xm && ys <= N/2 && N/2 < ys+ym)
		array[N/2][N/2] = 1;
	ierr = DMDAVecRestoreArray(da, x, &array); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da, act, &arract); CHKERRQ(ierr);
	
	// set boundary conditions by zeroing rows; NOTE collective call of MatZeroRowsStencil()!
	row.i = row.j = -1;
	if (xs <= 0 && 0 <= xs+xm && ys <= 0 && 0 <= ys+ym)
		row.i = row.j = 0;
	ierr = MatZeroRowsStencil(A, 1, &row, 2.0, x, b); CHKERRQ(ierr);

	row.i = row.j = -1;
	if (xs <= N-1 && N-1 <= xs+xm && ys <= N-1 && N-1 <= ys+ym)
		row.i = row.j = N-1;
	ierr = MatZeroRowsStencil(A, 1, &row, 2.0, x, b); CHKERRQ(ierr);

	row.i = row.j = -1;
	if (xs <= 0 && 0 <= xs+xm && ys <= N-1 && N-1 <= ys+ym)
		{row.i = 0; row.j = N-1;}
	ierr = MatZeroRowsStencil(A, 1, &row, 2.0, x, b); CHKERRQ(ierr);

	row.i = row.j = -1;
	if (xs <= N-1 && N-1 <= xs+xm && ys <= 0 && 0 <= ys+ym)
		{row.i = N-1; row.j = 0;}
	ierr = MatZeroRowsStencil(A, 1, &row, 2.0, x, b); CHKERRQ(ierr);

	row.i = row.j = -1;
	if (xs <= N/2 && N/2 <= xs+xm && ys <= N/2 && N/2 <= ys+ym)
		row.i = row.j = N/2;
	ierr = MatZeroRowsStencil(A, 1, &row, 2.0, x, b); CHKERRQ(ierr);

//	ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
//	ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
//	ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	// Solve the linear system
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting solver...\n"); CHKERRQ(ierr);
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
	ierr = KSPAppendOptionsPrefix(ksp, "p_"); CHKERRQ(ierr);
	ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
	ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

	ierr = KSPGetIterationNumber(ksp, &itnum); CHKERRQ(ierr);
	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

	ierr = DMView(da, vtk); CHKERRQ(ierr);
	ierr = VecView(x, vtk); CHKERRQ(ierr);

	ierr = MatMult(A, x, resid); CHKERRQ(ierr);
	ierr = VecAXPY(resid, -1, b); CHKERRQ(ierr);
	ierr = VecNorm(resid, NORM_2, &norm); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Iterations: %d, residual 2-norm: %g\n", itnum, (double)norm); CHKERRQ(ierr);

	ierr = VecDestroy(&b); CHKERRQ(ierr);
	ierr = VecDestroy(&resid); CHKERRQ(ierr);
	ierr = VecDestroy(&act0); CHKERRQ(ierr);
	ierr = MatDestroy(&A); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&vtk); CHKERRQ(ierr);
	//ierr = PetscLogStagePop(); CHKERRQ(ierr);


	// ------------------------ ADVECTION -----------------------
	PetscPreLoadStage("Advection");

	ierr = DMCreateMatrix(da, &B); CHKERRQ(ierr);
	ierr = MatSetOption(B, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE); CHKERRQ(ierr);
	ierr = DMCreateLocalVector(da, &xloc); CHKERRQ(ierr);	// xloc - "x with ghost points"
	ierr = DMCreateGlobalVector(da, &y); CHKERRQ(ierr);		// y - advection solution
	ierr = VecDuplicate(y, &c); CHKERRQ(ierr);				// c - advection rhs
	ierr = VecDuplicate(c, &resid); CHKERRQ(ierr);

	ierr = DMGlobalToLocalBegin(da, x, INSERT_VALUES, xloc); CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da, x, INSERT_VALUES, xloc); CHKERRQ(ierr);
	ierr = VecDestroy(&x); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, xloc, &array); CHKERRQ(ierr);	// array - stores pressure
	double **yarr;												// yarr - initial y
	ierr = DMDAVecGetArray(da, y, &yarr); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, act, &arract); CHKERRQ(ierr);

	//ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] DEBUG-1\n", rank); CHKERRQ(ierr);	// DEBUG
	for (int i = xs; i < xs+xm; i++)			// fill the matrix and initial y
		for (int j = ys; j < ys+ym; j++)
		{
			if (((i-N/2)*(i-N/2) + (j-N/2)*(j-N/2))*100 <= N2)				// initial y
				yarr[j][i] = 1;
			else
				yarr[j][i] = 0;

			row.i = i;
			row.j = j; 					//      4
			col[0].i = i-1;				//	0 | 2 | 3
			col[0].j = j;			    //      1
			col[1].i = i;
			col[1].j = j-1;
			col[2].i = i;
			col[2].j = j;
			col[3].i = i+1;
			col[3].j = j;
			col[4].i = i;
			col[4].j = j+1;

			PetscScalar p0, p1, p2, p3;
			p0 = p1 = p2 = p3 = 0;
			val[0] = val[1] = val[3] = val[4] = 0;
			val[2] = h2/dt;

			if (i+1 < N && arract[j][i] == 0 && arract[j][i+1] == 0)
				p0 = array[j][i] - array[j][i+1];
			else
				col[3].i = -1;

			if (i-1 >= 0 && arract[j][i] == 0 && arract[j][i-1] == 0)
				p1 = array[j][i] - array[j][i-1];
			else
				col[0].i = -1;

			if (j+1 < N && arract[j][i] == 0 && arract[j+1][i] == 0)
				p2 = array[j][i] - array[j+1][i];
			else
				col[4].j = -1;

			if (j-1 >= 0 && arract[j][i] == 0 && arract[j-1][i] == 0)
				p3 = array[j][i] - array[j-1][i];
			else
				col[1].j = -1;

			if (p0 > 0)
				val[2] += p0;
			else
				val[3] = p0;

			if (p1 > 0)
				val[2] += p1;
			else
				val[0] = p1;

			if (p2 > 0)
				val[2] += p2;
			else
				val[4] = p2;

			if (p3 > 0)
				val[2] += p3;
			else
				val[1] = p3;

			ierr  = MatSetValuesStencil(B, 1, &row, 5, col, val, INSERT_VALUES); CHKERRQ(ierr);
		}

	ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da, xloc, &array); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da, y, &yarr); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da, act, &arract); CHKERRQ(ierr);

	// output initial advection vector
	char fname[100];
	sprintf(fname, "output_B_adv_%d.vts", 0);
	ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, fname, FILE_MODE_WRITE, &vtk); CHKERRQ(ierr);
	ierr = VecView(y, vtk); CHKERRQ(ierr);
	ierr = VecSum(y, &swsum); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "integ(initial sw) = %g\n", swsum*h2); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&vtk); CHKERRQ(ierr);

	// set up KSP
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
	ierr = KSPAppendOptionsPrefix(ksp, "sw_"); CHKERRQ(ierr);
	ierr = KSPSetOperators(ksp, B, B); CHKERRQ(ierr);
	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

	//---------------------------------------
	// DEBUG check of matrix B
//	Vec e, Bte;
//	ierr = VecDuplicate(y, &e); CHKERRQ(ierr);
//	ierr = VecDuplicate(y, &Bte); CHKERRQ(ierr);
//	ierr = VecSet(e, 1); CHKERRQ(ierr);
//	ierr = MatMultTranspose(B, e, Bte); CHKERRQ(ierr);
//	ierr = VecScale(Bte, dt/h2); CHKERRQ(ierr);
//	ierr = VecAXPY(e, -1, Bte); CHKERRQ(ierr);
//	ierr = VecNorm(e, NORM_INFINITY, &norm); CHKERRQ(ierr);
//	ierr = PetscPrintf(PETSC_COMM_WORLD, "|Bte - e|_inf = %g\n", norm); CHKERRQ(ierr);
//
//	ierr = VecSet(e, 1); CHKERRQ(ierr);
//	ierr = KSPSolveTranspose(ksp, e, Bte); CHKERRQ(ierr);
//	ierr = VecScale(Bte, h2/dt); CHKERRQ(ierr);
//	ierr = VecAXPY(e, -1, Bte); CHKERRQ(ierr);
//	ierr = VecNorm(e, NORM_INFINITY, &norm); CHKERRQ(ierr);
//	ierr = PetscPrintf(PETSC_COMM_WORLD, "|B^(-t)e - e|_inf = %g\n", norm); CHKERRQ(ierr);
//
//	ierr = VecDestroy(&e); CHKERRQ(ierr);
//	ierr = VecDestroy(&Bte); CHKERRQ(ierr);
	// DEBUG
	//---------------------------------------

	for (int t = 0; t < T; t++)
	{
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Advection solver (%d)...\n", t); CHKERRQ(ierr);
		ierr = VecCopy(y, c); CHKERRQ(ierr);
		ierr = VecScale(c, h2/dt); CHKERRQ(ierr);
		ierr = KSPSolve(ksp, c, y); CHKERRQ(ierr);

		ierr = KSPGetIterationNumber(ksp, &itnum); CHKERRQ(ierr);
		ierr = VecSum(y, &swsum); CHKERRQ(ierr);
		ierr = MatMult(B, y, resid); CHKERRQ(ierr);
		ierr = VecAXPY(resid, -1, c); CHKERRQ(ierr);
		ierr = VecNorm(resid, NORM_2, &norm); CHKERRQ(ierr);
		ierr = VecNorm(resid, NORM_INFINITY, &norminf); CHKERRQ(ierr);
		ierr = VecNorm(resid, NORM_1, &norm1); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Iterations: %d, residual 2-norm: %g, inf-norm: %g, 1-norm: %g, integ(sw): %0.20g\n",
				itnum, (double)norm, (double)norminf, (double)norm1, (double)swsum*h2); CHKERRQ(ierr);

		if (t % save == 0)
		{
			sprintf(fname, "output_B_adv_%d.vts", t+1);
			ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, fname, FILE_MODE_WRITE, &vtk); CHKERRQ(ierr);
			ierr = VecView(y, vtk); CHKERRQ(ierr);
			ierr = PetscViewerDestroy(&vtk); CHKERRQ(ierr);
		}
	}

	// ------------------------ FINALIZING ----------------------
	// Destroy objects
	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
	ierr = VecDestroy(&xloc); CHKERRQ(ierr);
	ierr = VecDestroy(&y); CHKERRQ(ierr);
	ierr = VecDestroy(&c); CHKERRQ(ierr);
	ierr = VecDestroy(&act); CHKERRQ(ierr);
	ierr = DMDestroy(&da); CHKERRQ(ierr);

	PetscPreLoadEnd();

	ierr = PetscFinalize();

	return 0;
	//PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */









