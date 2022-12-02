// Manually define Laplace eq-n
// Boundary cond-s: left & right: Dirichlet, top & bottom: Neumann

// Prints some stuff on PC (somewhat broken now)

// Converts solution for vtk output, and outputs.

static char help[] = "Mixed stuff...\n";


//#include <petscdm.h>
#include <cmath>
#include <vector>
#include <iostream>

#include <petscdmda.h>
#include "petscdraw.h"
#include "petscviewer.h"
#include "petscksp.h"
//#include <petscsnes.h>


//#undef __FUNCT__
//#define __FUNCT__ "main"
void cout_vec(const double *arr, int sz);
int main(int argc,char **argv)
{
	PetscInitialize(&argc,&argv,(char*)0,help);

	PetscErrorCode ierr;
	int rank, size, N = 5, S = 1;
	double p1, p2, p3, p4;

	PetscPreLoadBegin(PETSC_FALSE, "main section");

	// Set up the viewers
	PetscViewer view, vascii, vtk;
	PetscDraw draw;
	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &view); CHKERRQ(ierr);
	ierr = PetscViewerSetType(view,PETSCVIEWERDRAW); CHKERRQ(ierr);

	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &vascii); CHKERRQ(ierr);
	ierr = PetscViewerSetType(vascii,PETSCVIEWERASCII); CHKERRQ(ierr);

	ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, "output.vts", FILE_MODE_WRITE, &vtk); CHKERRQ(ierr);
	//ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "testing.vtk", &vtk); CHKERRQ(ierr);
	//ierr = PetscViewerSetFormat(vtk, PETSC_VIEWER_ASCII_VTK); CHKERRQ(ierr);

	//ierr = PetscViewerDrawSetPause(view, p1); CHKERRQ(ierr);
	//ierr = PetscViewerDrawSetHold(view, PETSC_TRUE); CHKERRQ(ierr);
	//ierr = PetscViewerDrawGetDraw(view, 0, &draw); CHKERRQ(ierr);
	//ierr = PetscDrawResizeWindow(draw, 800, 800);

	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "acsii_viewer.txt", &vascii);

	// Get rank, size, options
	ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);
	ierr  = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
	ierr  = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetInt(NULL,NULL,"-s",&S,NULL); CHKERRQ(ierr);			// # subdomains, when applicable
	ierr  = PetscOptionsGetReal(NULL,NULL,"-p1",&p1,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetReal(NULL,NULL,"-p2",&p2,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetReal(NULL,NULL,"-p3",&p3,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetReal(NULL,NULL,"-p4",&p4,NULL); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD,"N = %d\n", N); CHKERRQ(ierr);

	Mat A;
	Vec rhs, vis, x;
	
	int sz = (N+1)*(N+1);		// A is sz*sz; sz - size of inner grid area
	double N2 = (N+2)*(N+2);
	double h = 1.0/double(N+2);
	//																		d_nz			o_nz
	//														glob. rows, cols
	ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, sz, sz, 5, PETSC_NULL, 2, PETSC_NULL, &A); CHKERRQ(ierr);
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, sz, &rhs); CHKERRQ(ierr);
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, sz, &x); CHKERRQ(ierr);
	ierr = VecSet(rhs, 0); CHKERRQ(ierr);
	
	int r0, r1;
	ierr = MatGetOwnershipRange(A, &r0, &r1); CHKERRQ(ierr);
	double vals[5] = {N2, -4*N2, N2, N2, N2};
	for (int i = r0; i < r1; i++)
	{
		int i0 = i/(N+1);	// (i0, j0) - index in grid
		int j0 = i%(N+1);
		if (i0 > 0 && i0 < N && j0 > 0 && j0 < N)	// nodes not adjacent to boundary
		{
			int idxn[5] = {(i0+1)*(N+1)+j0, i0*(N+1)+j0, (i0-1)*(N+1)+j0, i0*(N+1)+j0+1, i0*(N+1)+j0-1};
			ierr = MatSetValues(A, 1, &i, 5, idxn, vals, INSERT_VALUES); CHKERRQ(ierr);
		}
		else	// node adjacent to boundary
		{
			std::vector<int> idxn = {i0*(N+1)+j0};
			std::vector<double> vals = {-4*N2};

			if (i0 != 0)
			{
				idxn.push_back((i0-1)*(N+1)+j0);
				vals.push_back(N2);
			}
			else
			{
				double rhsval = -N2 * pow(h*(j0+1), 2);
				ierr = VecSetValues(rhs, 1, &i, &rhsval, INSERT_VALUES); CHKERRQ(ierr);		// Dirichlet
			}

			if (i0 != N)
			{
				idxn.push_back((i0+1)*(N+1)+j0);
				vals.push_back(N2);
			}
			else
			{
				double rhsval = -N2 * (1 - pow(h*(j0+1), 2));
				ierr = VecSetValues(rhs, 1, &i, &rhsval, INSERT_VALUES); CHKERRQ(ierr);		// Dirichlet
			}

			if (j0 != 0)
			{
				idxn.push_back(i0*(N+1)+j0-1);
				vals.push_back(N2);
			}
			else
				vals[0] += N2;							// Neumann
				
			if (j0 != N)
			{
				idxn.push_back(i0*(N+1)+j0+1);
				vals.push_back(N2);
			}
			else
				vals[0] += N2;							// Neumann

			ierr = MatSetValues(A, 1, &i, idxn.size(), idxn.data(), vals.data(), INSERT_VALUES); CHKERRQ(ierr);
		}
	}

	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	
	ierr = VecAssemblyBegin(rhs); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(rhs); CHKERRQ(ierr);	
	
//	ierr = MatView(A, view); CHKERRQ(ierr);
//	ierr = MatView(A, vascii); CHKERRQ(ierr);
//	ierr = PetscSleep(p1); CHKERRQ(ierr);
	
//	ierr = VecView(rhs, view); CHKERRQ(ierr);
//	ierr = VecView(rhs, vascii); CHKERRQ(ierr);
//	ierr = PetscSleep(p2); CHKERRQ(ierr);


	// Solve the linear system
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting solver...\n"); CHKERRQ(ierr);
	KSP ksp;
	PC pc;
	PetscLogEvent EV;
	int itnum;
	//ierr = PCCreate(PETSC_COMM_WORLD, &pc); CHKERRQ(ierr);

	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
	ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

	ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
//	ierr = PCSetFromOptions(pc); CHKERRQ(ierr);
//	ierr = KSPSetPC(ksp, pc); CHKERRQ(ierr);
	//ierr = PCGASMSetTotalSubdomains(pc, S); CHKERRQ(ierr);			// *** GASM ***
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Set %d subdomains\n", S); CHKERRQ(ierr);
	ierr = KSPSetPC(ksp, pc); CHKERRQ(ierr);

	ierr = PetscLogEventRegister("Solver work", 0, &EV); CHKERRQ(ierr);
	ierr = PetscLogEventActivate(EV); CHKERRQ(ierr);
	ierr = PetscLogEventBegin(EV,0,0,0,0); CHKERRQ(ierr);

	ierr = KSPSolve(ksp, rhs, x); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Solver finished\n"); CHKERRQ(ierr);
	ierr = PetscLogEventEnd(EV,0,0,0,0); CHKERRQ(ierr);

	// get info on pc=ASM subdomains
	IS *isarr, *isarrloc;
	int ndom;
	ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
	//ierr = PCASMGetLocalSubdomains(pc, &ndom, &isarr, &isarrloc); CHKERRQ(ierr);
	//ierr = ISView(isarr[0], vascii); CHKERRQ(ierr);
	std::cout << "rank " << rank << ", number of domains " << ndom << ", isarr " << isarr << ", isarrloc " << isarrloc << "\n";
	int levels;
	ierr = PCMGGetLevels(pc, &levels); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Levels: %d\n", levels); CHKERRQ(ierr);

	//-------------------------------------

	ierr = KSPGetIterationNumber(ksp, &itnum); CHKERRQ(ierr);
	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Iterations: %d\n", itnum); CHKERRQ(ierr);
	//ierr = PCDestroy(&pc); CHKERRQ(ierr);

	// DMDA stuff, to handle vectors
	DM da;
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, N+1, N+1, 
				 		PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
	PetscCall(DMSetUp(da));
	ierr = DMCreateGlobalVector(da, &vis); CHKERRQ(ierr);
	
	// Get mappings from DMDA
//	ISLocalToGlobalMapping map;
//	ierr = DMGetLocalToGlobalMapping(da, &map); CHKERRQ(ierr);
//	ierr = ISLocalToGlobalMappingView(map, vascii); CHKERRQ(ierr);
	AO ao;
	ierr = DMDAGetAO(da, &ao); CHKERRQ(ierr);
	//ierr = AOView(ao, vascii); CHKERRQ(ierr);
	
	// transfer from "rhs" to "vis"
	int i0, i1;
	ierr = VecGetOwnershipRange(rhs, &i0, &i1); CHKERRQ(ierr);
	std::cout << "rank " << rank << "\ti0 " << i0 << "\ti1 " << i1 << "\n";	// DEBUG
	int *idx = new int[i1-i0];
	for (int i = i0; i < i1; i++)
		idx[i-i0] = i;
	ierr = AOApplicationToPetsc(ao, i1-i0, idx); CHKERRQ(ierr);	  // transform indices for proper work of VecSetValues
	
	// Copy to new vector ***
	double *arr;
	ierr = VecGetArray(x, &arr); CHKERRQ(ierr);
	ierr = VecSetValues(vis, i1-i0, idx, arr, INSERT_VALUES); CHKERRQ(ierr);
	ierr = VecAssemblyBegin(vis); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(vis); CHKERRQ(ierr);
	delete [] idx;
	ierr = VecRestoreArray(x, &arr); CHKERRQ(ierr);
	
//	ierr = VecGetOwnershipRange(vis, &i0, &i1); CHKERRQ(ierr);	// cout local elements of vis!
//	ierr = VecGetArray(vis, &arr); CHKERRQ(ierr);
//	std::cout << "rank " << rank << "\t";
//	cout_vec(arr, i1-i0);
//	ierr = VecRestoreArray(vis, &arr); CHKERRQ(ierr);
	
	
//	ierr = DMView(da, view); CHKERRQ(ierr);
//	ierr = DMView(da, vascii); CHKERRQ(ierr);
	ierr = DMDASetUniformCoordinates(da, 0.0, 1, 0.0, 1, 0.0, 0.0); CHKERRQ(ierr);
	ierr = DMView(da, vtk); CHKERRQ(ierr);
//	ierr = PetscSleep(p3); CHKERRQ(ierr);
	
	//ierr = VecView(vis, view); CHKERRQ(ierr);
	//ierr = VecView(vis, vascii); CHKERRQ(ierr);
	ierr = VecView(vis, vtk); CHKERRQ(ierr);
	//ierr = PetscSleep(p3); CHKERRQ(ierr);
	

		

	// Destroy objects
	ierr = PetscViewerDestroy(&vtk); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&view); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&vascii); CHKERRQ(ierr);

	ierr = DMDestroy(&da); CHKERRQ(ierr);
	ierr = VecDestroy(&vis); CHKERRQ(ierr);
	ierr = VecDestroy(&rhs); CHKERRQ(ierr);
	ierr = VecDestroy(&x); CHKERRQ(ierr);
	ierr = MatDestroy(&A); CHKERRQ(ierr);

	PetscPreLoadEnd();

	ierr = PetscFinalize();

	return 0;
	//PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
void cout_vec(const double *arr, int sz)
{
	for (int i = 0; i < sz; i++)
		std::cout << arr[i] << "\t";
	std::cout << "\n";
}








