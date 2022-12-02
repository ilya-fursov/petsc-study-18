// examine DMDA, local and global vecs, ISLocalToGlobal, AO
// using ViewerDraw, ViewerASCII


static char help[] = "Examine DMDA and numberings\n";


//#include <petscdm.h>
#include <petscdmda.h>
#include "petscdraw.h"
#include "petscviewer.h"
#include <iostream>
//#include <petscsnes.h>

typedef struct
{
	double a;
	double b;
	double c;
} Field;

//#undef __FUNCT__
//#define __FUNCT__ "main"
int main(int argc,char **argv)
{
	PetscInitialize(&argc,&argv,(char*)0,help);

	PetscErrorCode ierr;
	int rank, size, N = 5;
	double p1, p2, p3, p4;
	DM da;

	// Set up the viewers
	PetscViewer view, vascii;
	PetscDraw draw;
	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &view); CHKERRQ(ierr);
	ierr = PetscViewerSetType(view,PETSCVIEWERDRAW); CHKERRQ(ierr);

	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &vascii); CHKERRQ(ierr);
	ierr = PetscViewerSetType(vascii,PETSCVIEWERASCII); CHKERRQ(ierr);

	//ierr = PetscViewerDrawSetPause(view, p1); CHKERRQ(ierr);
	ierr = PetscViewerDrawSetHold(view, PETSC_TRUE); CHKERRQ(ierr);
	ierr = PetscViewerDrawGetDraw(view, 0, &draw); CHKERRQ(ierr);
	ierr = PetscDrawResizeWindow(draw, 800, 800);

	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "acsii_viewer.txt", &vascii);

	// Get rank, size, options
	ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);
	ierr  = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
	ierr  = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetReal(NULL,NULL,"-p1",&p1,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetReal(NULL,NULL,"-p2",&p2,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetReal(NULL,NULL,"-p3",&p3,NULL); CHKERRQ(ierr);
	ierr  = PetscOptionsGetReal(NULL,NULL,"-p4",&p4,NULL); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD,"N = %d\n", N); CHKERRQ(ierr);

	// create DMDA
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, N, N, PETSC_DECIDE, PETSC_DECIDE,
		  3, 1, NULL, NULL, &da); CHKERRQ(ierr);
	PetscCall(DMSetUp(da));

	ierr = DMView(da, view); CHKERRQ(ierr);		// view DMDA
	ierr = PetscViewerASCIIPrintf(vascii, "DMDA object:\n"); CHKERRQ(ierr);
	ierr = DMView(da, vascii); CHKERRQ(ierr);
	ierr = PetscSleep(p1); CHKERRQ(ierr);

	// create vectors
	Vec ga, la;
	ierr = DMCreateGlobalVector(da, &ga); CHKERRQ(ierr);
	ierr = DMCreateLocalVector(da, &la); CHKERRQ(ierr);
	ierr = VecSet(ga, 3); CHKERRQ(ierr);
	ierr = VecSet(la, rank+1); CHKERRQ(ierr);

	Field **x;
	ierr = DMDAVecGetArray(da, ga, &x); CHKERRQ(ierr);
	
	int i0, j0, m0, n0;
	int i1, j1, m1, n1;
	ierr = DMDAGetCorners(da, &i0, &j0, NULL, &m0, &n0, NULL); CHKERRQ(ierr);
	ierr = DMDAGetGhostCorners(da, &i1, &j1, NULL, &m1, &n1, NULL); CHKERRQ(ierr);
	//ierr = PetscPrintf(PETSC_COMM_WORLD, "i0 = %d, j0 = %d, m0 = %d, n0 = %d\n", i0, j0, m0, n0); CHKERRQ(ierr);
	printf("RANK = %d\tcorners i0 = %d, j0 = %d, m0 = %d, n0 = %d\n", rank, i0, j0, m0, n0);
	printf("RANK = %d\tgh. cor i1 = %d, j1 = %d, m1 = %d, n1 = %d\n", rank, i1, j1, m1, n1);
	
	for (int i = i0; i < i0+m0; i++)
		for (int j = j0; j < j0+n0; j++)
		{
			x[j][i].a = i*N + j;
			x[j][i].b = i*N + j + 1000;
			x[j][i].c = i*N + j + 2000;
		}

	//DMGlobalToLocalBegin(da, ga, INSERT_VALUES, la);
	//DMGlobalToLocalEnd(da, ga, INSERT_VALUES, la);	

	//DMLocalToGlobalBegin(da, la, INSERT_VALUES, ga);  // !
	//DMLocalToGlobalEnd(da, la, INSERT_VALUES, ga);	// !

	ierr = DMDAVecRestoreArray(da, ga, &x); CHKERRQ(ierr);

	ierr = PetscViewerASCIIPrintf(vascii, "\nGlobal vector:\n"); CHKERRQ(ierr);
	ierr = VecView(ga, view); CHKERRQ(ierr);
	ierr = VecView(ga, vascii); CHKERRQ(ierr);
	ierr = PetscSleep(p2); CHKERRQ(ierr);

	ierr = PetscViewerASCIIPrintf(vascii, "\nLocal vector:\n"); CHKERRQ(ierr);
	ierr = VecView(la, view); CHKERRQ(ierr);		// only rank-0 seems to view
	ierr = VecView(la, vascii); CHKERRQ(ierr);		// only rank-0 seems to view
	ierr = PetscSleep(p3); CHKERRQ(ierr);

	// Get mappings from DMDA
	ISLocalToGlobalMapping map;
	ierr = DMGetLocalToGlobalMapping(da, &map); CHKERRQ(ierr);
	ierr = ISLocalToGlobalMappingView(map, view); CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(vascii, "\nIS local to global:\n"); CHKERRQ(ierr);
	ierr = ISLocalToGlobalMappingView(map, vascii); CHKERRQ(ierr);
	ierr = PetscSleep(p4); CHKERRQ(ierr);
	
	AO ao;
	ierr = DMDAGetAO(da, &ao); CHKERRQ(ierr);
	ierr = AOView(ao, view); CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(vascii, "\nAO:\n"); CHKERRQ(ierr);
	ierr = AOView(ao, vascii); CHKERRQ(ierr);
	ierr = PetscSleep(p4); CHKERRQ(ierr);


	// Destroy objects
	ierr = PetscViewerDestroy(&view); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&vascii); CHKERRQ(ierr);
	ierr = DMDestroy(&da); CHKERRQ(ierr);
	ierr = VecDestroy(&ga); CHKERRQ(ierr);
	ierr = VecDestroy(&la); CHKERRQ(ierr); 
	//ierr = ISLocalToGlobalMappingDestroy(map); CHKERRQ(ierr);	
	ierr = PetscFinalize();

	return 0;
	//PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */







