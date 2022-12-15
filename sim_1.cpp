#include <vector>
#include <iostream>
#include <petscts.h>
#include <petscdmda.h>

static char help[] = "Sim version 1. Simple, structured, linear. Options:\n" \
					"-Ni\n" \
					"-Nj\n" \
					"-Lx\n" \
					"-Ly\n" \
					"-Lz\n" \
					"-Tmax\n" \
					"-dt0\n" \
					"-k1\n" \
					"-k2\n" \
					"-k3\n" \
					"-k4\n" \
					"-cf\n" \
					"-cw\n" \
					"-poro\n" \
					"-uw\n" \
					"-rw\n" \
					"-Q0\n" \
					"-Pinit" \
					"-savearr\n";

const int BUFFSIZE = 500;

struct AppCtx
{
  MPI_Comm  comm;
  DM        da;

  PetscInt  Ni, Nj;    		// mesh size
  PetscReal Lx, Ly, Lz;    	// area size
  PetscReal Tmax, dt0;		// max time, initial time step
  PetscReal dx, dy, dz, PV0;		// geometry, PVOL

  PetscScalar k1, k2, k3, k4;		// permeability in boxes
  PetscScalar cf, cw;		// compressibilities
  PetscScalar poro, uw, rw;	// porosity, water visc., well radius
  PetscScalar Q0, Pinit;	// well rate, initial pressure
  PetscBool savearr;		// save pressure grids?

  Vec uloc1;				// work vector (local)
  Vec perm;					// global vec for permeability
  Vec Tx, Ty;				// local vecs for transmissibilities (i -> i+1), (j -> j+1)
  const PetscScalar C = 0.008527;	// Darcy const

  std::vector<double> wbhp;	// 3*T row-major array storing: (time, prod-bhp, inj-bhp)

  AppCtx() : comm(MPI_COMM_NULL), da(NULL), Ni(100), Nj(100), Lx(1000), Ly(1000), Lz(10), Tmax(60), dt0(1),
		  	 dx(0), dy(0), dz(0), PV0(0),
		     k1(15), k2(50), k3(150), k4(45), cf(6e-5), cw(5e-5),
		     poro(0.2), uw(0.5), rw(0.1), Q0(100), Pinit(200), savearr(PETSC_TRUE), uloc1(NULL),
		     perm(NULL), Tx(NULL), Ty(NULL) {};

  PetscErrorCode SetOptions();
  PetscErrorCode CreateObjects();
  PetscErrorCode DestroyObjects();

  PetscScalar getk(PetscInt i, PetscInt j) const;		// perm for block (i,j)
};

//PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec global_in, Vec global_out, void *ctx);
//PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec global_in, Mat AA, Mat BB, void *ctx);

PetscErrorCode IFunction(TS ts, PetscReal t, Vec u, Vec u_t, Vec F, void *ctx);
PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec U_t, PetscReal a, Mat Amat, Mat Pmat, void *ctx);

inline PetscReal Max2(PetscReal a, PetscReal b, PetscInt &ind);		// max(a, b), sets ind=0|1 depending on who is max
PetscErrorCode CalcBHP(Vec u, std::vector<double> &res, void *ctx);	// fills 'res' with data (BHP) derived from 'u'
PetscErrorCode monitor(TS ts, PetscInt step, PetscReal time, Vec u, void *mctx);
PetscErrorCode SaveVecVtk(Vec x, const char fname[], AppCtx *ctx);		// write grid to vtk file
PetscErrorCode SaveVecAscii(Vec x, const char fname[], AppCtx *ctx);	// write grid to ascii file


//---------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
	PetscFunctionBeginUser;
	PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

	AppCtx    ctx;               	// user-defined application context
	TS        ts;
	Mat       A;              		// I-Jacobian
	Vec       u;                    // approximate solution vector

	ctx.comm = PETSC_COMM_WORLD;
	PetscCallBack("SetOptions", ctx.SetOptions());
	PetscCallBack("CreateObjects", ctx.CreateObjects());
	PetscCallBack("SaveVecVtk", SaveVecVtk(ctx.perm, "sim1_outk.vts", &ctx));
	PetscCall(DMCreateGlobalVector(ctx.da, &u));			// current solution

	PetscCall(TSCreate(ctx.comm, &ts));
	PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
	PetscCall(TSSetIFunction(ts, NULL, IFunction, &ctx));

	PetscCall(TSMonitorSet(ts, monitor, &ctx, NULL));

	PetscCall(TSSetDM(ts, ctx.da));
	PetscCall(DMCreateMatrix(ctx.da, &A));
	PetscCall(TSSetIJacobian(ts, A, A, IJacobian, &ctx));
	PetscCall(TSSetTimeStep(ts, ctx.dt0));

	PetscCall(TSSetType(ts, TSBEULER));
	PetscCall(TSSetMaxSteps(ts, 10000));
	PetscCall(TSSetMaxTime(ts, ctx.Tmax));
	PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
	PetscCall(TSSetFromOptions(ts));

	// Solution
	PetscCall(VecSet(u, ctx.Pinit));			// initial conditions
	PetscCall(TSSolve(ts, u));

	// dtors
	PetscCall(TSDestroy(&ts));
	PetscCall(VecDestroy(&u));
	PetscCall(MatDestroy(&A));
	PetscCallBack("DestroyObjects", ctx.DestroyObjects());

	PetscCall(PetscFinalize());
	return 0;
}
//---------------------------------------------------------------------------------------------
PetscErrorCode AppCtx::SetOptions()
{
	PetscFunctionBeginUser;

	PetscCall(PetscOptionsGetInt(NULL, NULL, "-Ni", &Ni, NULL));
	PetscCall(PetscOptionsGetInt(NULL, NULL, "-Nj", &Nj, NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-Lx",&Lx,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-Ly",&Ly,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-Lz",&Lz,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-Tmax",&Tmax,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-dt0",&dt0,NULL));

	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-k1",&k1,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-k2",&k2,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-k3",&k3,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-k4",&k4,NULL));

	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-cf",&cf,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-cw",&cw,NULL));

	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-poro",&poro,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-uw",&uw,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-rw",&rw,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-Q0",&Q0,NULL));
	PetscCall(PetscOptionsGetScalar(NULL,NULL,"-Pinit",&Pinit,NULL));

	PetscCall(PetscOptionsHasName(NULL,NULL, "-savearr", &savearr));

	dx = Lx/Ni;
	dy = Ly/Nj;
	dz = Lz;
	PV0 = poro*dx*dy*dz;

	PetscFunctionReturn(0);
}
//---------------------------------------------------------------------------------------------
PetscErrorCode AppCtx::CreateObjects()
{
	PetscFunctionBeginUser;

	PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, Ni, Nj, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &da));
	PetscCall(DMSetUp(da));
	PetscCall(DMDASetUniformCoordinates(da, dx/2, Lx-dx/2, dy/2, Ly-dy/2, 0.0, 0.0));
	PetscCall(DMCreateLocalVector(da, &uloc1));		// assembly in, local

	PetscCall(DMCreateGlobalVector(da, &perm));		// global vec for permeability
	PetscCall(VecDuplicate(uloc1, &Tx));			// local vecs for transmissibilities (i -> i+1), (j -> j+1)
	PetscCall(VecDuplicate(uloc1, &Ty));

	Vec TI, TJ;										// global work vecs for transmissibilities
	PetscCall(VecDuplicate(perm, &TI));
	PetscCall(VecDuplicate(perm, &TJ));
	PetscCall(VecSet(TI, 0));
	PetscCall(VecSet(TJ, 0));

	// fill permeability and transmissibilities
	PetscScalar **k, **ti, **tj;					// global arrays
	PetscInt xs, ys, xm, ym;						// corners

	PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));
	PetscCall(DMDAVecGetArray(da, perm, &k));
	for (PetscInt j = ys; j < ys+ym; j++)
		for (PetscInt i = xs; i < xs+xm; i++)
			k[j][i] = getk(i, j);
	PetscCall(DMDAVecRestoreArray(da, perm, &k));

	PetscCall(DMGlobalToLocalBegin(da, perm, INSERT_VALUES, uloc1));	// scatter perm to local uloc1 - for transm calculations
	PetscCall(DMGlobalToLocalEnd(da, perm, INSERT_VALUES, uloc1));

	PetscCall(DMDAVecGetArray(da, uloc1, &k));		// loc
	PetscCall(DMDAVecGetArray(da, TI, &ti));		// glob
	PetscCall(DMDAVecGetArray(da, TJ, &tj));		// glob

	const PetscScalar geomX = 2*C*dy*dz/dx;
	const PetscScalar geomY = 2*C*dx*dz/dy;
	for (PetscInt j = ys; j < ys+ym; j++)
		for (PetscInt i = xs; i < xs+xm; i++)
		{
			if (i+1 < Ni)
				ti[j][i] = geomX/(1/k[j][i] + 1/k[j][i+1]);				// k has ghost points!
			if (j+1 < Nj)
				tj[j][i] = geomY/(1/k[j][i] + 1/k[j+1][i]);
		}

	PetscCall(DMDAVecRestoreArray(da, uloc1, &k));
	PetscCall(DMDAVecRestoreArray(da, TI, &ti));
	PetscCall(DMDAVecRestoreArray(da, TJ, &tj));

	PetscCall(DMGlobalToLocalBegin(da, TI, INSERT_VALUES, Tx));
	PetscCall(DMGlobalToLocalEnd(da, TI, INSERT_VALUES, Tx));

	PetscCall(DMGlobalToLocalBegin(da, TJ, INSERT_VALUES, Ty));
	PetscCall(DMGlobalToLocalEnd(da, TJ, INSERT_VALUES, Ty));

	PetscCall(VecDestroy(&TI));
	PetscCall(VecDestroy(&TJ));

	wbhp = std::vector<double>();		// reset the BHP data

	PetscFunctionReturn(0);
}
//---------------------------------------------------------------------------------------------
PetscErrorCode AppCtx::DestroyObjects()
{
	PetscFunctionBeginUser;

	PetscCall(DMDestroy(&da));
	PetscCall(VecDestroy(&uloc1));

	PetscCall(VecDestroy(&perm));
	PetscCall(VecDestroy(&Tx));
	PetscCall(VecDestroy(&Ty));

	// write the saved 'wbhp'
	PetscCheck(wbhp.size() % 3 == 0, comm, PETSC_ERR_ARG_SIZ, "wbhp.size() = %zu", wbhp.size());
	FILE *F;
	PetscCall(PetscFOpen(comm, "Sim1_BHPout.txt", "w", &F));
	for (size_t i = 0; i < wbhp.size(); i += 3)
		PetscCall(PetscFPrintf(comm, F, "%g\t%-20.16g\t%-20.16g\n", wbhp[i], wbhp[i+1], wbhp[i+2]));
	PetscCall(PetscFClose(comm, F));

	PetscFunctionReturn(0);
}
//---------------------------------------------------------------------------------------------
PetscScalar AppCtx::getk(PetscInt i, PetscInt j) const		// perm for block (i,j)
{
	if (i < Ni/2)	// k1, k3
	{
		if (j < Nj/2)
			return k1;
		else
			return k3;
	}
	else			// k2, k4
	{
		if (j < Nj/2)
			return k2;
		else
			return k4;
	}
}
//---------------------------------------------------------------------------------------------
// Callbacks
//---------------------------------------------------------------------------------------------
PetscErrorCode IFunction(TS ts, PetscReal t, Vec u, Vec u_t, Vec F, void *ctx)
{
	PetscFunctionBeginUser;
	AppCtx *Ctx = (AppCtx*)ctx;
	const PetscScalar **p, **pt, **ti, **tj;
	PetscScalar **f;
	PetscInt xs, ys, xm, ym;		// corners

	const PetscScalar p0 = Ctx->Pinit;
	const PetscScalar ct = Ctx->cf + Ctx->cw;

	PetscCall(DMDAGetCorners(Ctx->da, &xs, &ys, NULL, &xm, &ym, NULL));

	PetscCall(DMGlobalToLocalBegin(Ctx->da, u, INSERT_VALUES, Ctx->uloc1));
	PetscCall(DMGlobalToLocalEnd(Ctx->da, u, INSERT_VALUES, Ctx->uloc1));

	PetscCall(DMDAVecGetArrayRead(Ctx->da, Ctx->Tx, &ti));			// local
	PetscCall(DMDAVecGetArrayRead(Ctx->da, Ctx->Ty, &tj));			// local
	PetscCall(DMDAVecGetArrayRead(Ctx->da, Ctx->uloc1, &p));		// local
	PetscCall(DMDAVecGetArrayRead(Ctx->da, u_t, &pt));				// global
	PetscCall(DMDAVecGetArray(Ctx->da, F, &f));						// global, out

	// calculate the residual
	for (PetscInt j = ys; j < ys+ym; j++)
		for (PetscInt i = xs; i < xs+xm; i++)
		{
			const PetscScalar pi = p[j][i];
			PetscScalar res = Ctx->PV0 * PetscExpScalar(ct*(pi - p0)) * ct * pt[j][i];		// accumulation

			if (i==0 && j==0)
				res += Ctx->Q0;							// flow to producer
			if (i==Ctx->Ni-1 && j==Ctx->Nj-1)
				res -= Ctx->Q0;							// flow from injector

			PetscScalar pu;				// upstream pressure
			if (i-1 >= 0)				// flow to neighbours
			{
				pu = PetscMax(p[j][i-1], pi);
				res += ti[j][i-1]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0)) * (pi - p[j][i-1]);
			}
			if (i+1 < Ctx->Ni)
			{
				pu = PetscMax(p[j][i+1], pi);
				res += ti[j][i]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0)) * (pi - p[j][i+1]);
			}
			if (j-1 >= 0)
			{
				pu = PetscMax(p[j-1][i], pi);
				res += tj[j-1][i]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0)) * (pi - p[j-1][i]);
			}
			if (j+1 < Ctx->Nj)
			{
				pu = PetscMax(p[j+1][i], pi);
				res += tj[j][i]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0)) * (pi - p[j+1][i]);
			}

			f[j][i] = res;
		}

	PetscCall(DMDAVecRestoreArrayRead(Ctx->da, Ctx->Tx, &ti));
	PetscCall(DMDAVecRestoreArrayRead(Ctx->da, Ctx->Ty, &tj));
	PetscCall(DMDAVecRestoreArrayRead(Ctx->da, Ctx->uloc1, &p));
	PetscCall(DMDAVecRestoreArrayRead(Ctx->da, u_t, &pt));
	PetscCall(DMDAVecRestoreArray(Ctx->da, F, &f));

	PetscFunctionReturn(0);
}
//---------------------------------------------------------------------------------------------
// F_u + a*F_ut
PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec U_t, PetscReal a, Mat Amat, Mat Pmat, void *ctx)
{
	PetscFunctionBeginUser;
	AppCtx *Ctx = (AppCtx*)ctx;
	const PetscScalar **p, **pt, **ti, **tj;
	PetscInt xs, ys, xm, ym;		// corners
	MatStencil row, col[5];
	PetscScalar vals[5];

	const PetscScalar p0 = Ctx->Pinit;
	const PetscScalar ct = Ctx->cf + Ctx->cw;

	PetscCall(DMDAGetCorners(Ctx->da, &xs, &ys, NULL, &xm, &ym, NULL));

	PetscCall(DMGlobalToLocalBegin(Ctx->da, U, INSERT_VALUES, Ctx->uloc1));
	PetscCall(DMGlobalToLocalEnd(Ctx->da, U, INSERT_VALUES, Ctx->uloc1));

	PetscCall(DMDAVecGetArrayRead(Ctx->da, Ctx->Tx, &ti));			// local
	PetscCall(DMDAVecGetArrayRead(Ctx->da, Ctx->Ty, &tj));			// local
	PetscCall(DMDAVecGetArrayRead(Ctx->da, Ctx->uloc1, &p));		// local
	PetscCall(DMDAVecGetArrayRead(Ctx->da, U_t, &pt));				// global

	// insert the values
	for (PetscInt j = ys; j < ys+ym; j++)
		for (PetscInt i = xs; i < xs+xm; i++)
		{
			PetscInt ind;			// for Max2()

			row.i = i;
			row.j = j;

			col[0].i = i;
			col[0].j = j;			//
			col[1].i = i-1;			//
			col[1].j = j;			//		   3
			col[2].i = i+1;			//		1  0  2
			col[2].j = j;			//		   4
			col[3].i = i;			//
			col[3].j = j-1;			//
			col[4].i = i;
			col[4].j = j+1;

			vals[0] = vals[1] = vals[2] = vals[3] = vals[4] = 0;

			const PetscScalar pi = p[j][i];

			//           res = Ctx->PV0 * PetscExpScalar(ct*(pi - p0)) * ct *      pt[j][i];
			PetscScalar diag = Ctx->PV0 * PetscExpScalar(ct*(pi - p0)) * ct * ct * pt[j][i];		// accumulation
			diag            += Ctx->PV0 * PetscExpScalar(ct*(pi - p0)) * ct *      a;

			PetscScalar pu;				// upstream pressure
			if (i-1 >= 0)				// flow to neighbours
			{
				pu = Max2(p[j][i-1], pi, ind);
				//res        += ti[j][i-1]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0)) * (pi - p[j][i-1]);

				PetscScalar x = ti[j][i-1]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0));
				diag += x;
				vals[1] -= x;

				PetscScalar aux = x * Ctx->cw * (pi - p[j][i-1]);
				if (ind == 1)
					diag += aux;
				else
					vals[1] += aux;
			}
			else
				col[1].i = -1;

			if (i+1 < Ctx->Ni)
			{
				pu = Max2(p[j][i+1], pi, ind);
				//res        += ti[j][i]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0)) * (pi - p[j][i+1]);

				PetscScalar x = ti[j][i]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0));
				diag += x;
				vals[2] -= x;

				PetscScalar aux = x * Ctx->cw * (pi - p[j][i+1]);
				if (ind == 1)
					diag += aux;
				else
					vals[2] += aux;
			}
			else
				col[2].i = -1;

			if (j-1 >= 0)
			{
				pu = Max2(p[j-1][i], pi, ind);
				//res        += tj[j-1][i]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0)) * (pi - p[j-1][i]);

				PetscScalar x = tj[j-1][i]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0));
				diag += x;
				vals[3] -= x;

				PetscScalar aux = x * Ctx->cw * (pi - p[j-1][i]);
				if (ind == 1)
					diag += aux;
				else
					vals[3] += aux;
			}
			else
				col[3].j = -1;

			if (j+1 < Ctx->Nj)
			{
				pu = Max2(p[j+1][i], pi, ind);
				//res        += tj[j][i]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0)) * (pi - p[j+1][i]);

				PetscScalar x = tj[j][i]/Ctx->uw * PetscExpScalar(Ctx->cw*(pu - p0));
				diag += x;
				vals[4] -= x;

				PetscScalar aux = x * Ctx->cw * (pi - p[j+1][i]);
				if (ind == 1)
					diag += aux;
				else
					vals[4] += aux;
			}
			else
				col[4].j = -1;

			vals[0] = diag;

			PetscCall(MatSetValuesStencil(Pmat, 1, &row, 5, col, vals, INSERT_VALUES));
		}

	PetscCall(MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY));
	PetscCall(MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY));
	if (Pmat != Amat)
	{
	    PetscCall(MatAssemblyBegin(Amat, MAT_FINAL_ASSEMBLY));
	    PetscCall(MatAssemblyEnd(Amat, MAT_FINAL_ASSEMBLY));
	}

	PetscCall(MatSetOption(Pmat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));	// TODO well, this option seems to be ok; no change noticed

	PetscCall(DMDAVecRestoreArrayRead(Ctx->da, Ctx->Tx, &ti));
	PetscCall(DMDAVecRestoreArrayRead(Ctx->da, Ctx->Ty, &tj));
	PetscCall(DMDAVecRestoreArrayRead(Ctx->da, Ctx->uloc1, &p));
	PetscCall(DMDAVecRestoreArrayRead(Ctx->da, U_t, &pt));

	PetscFunctionReturn(0);
}
//---------------------------------------------------------------------------------------------
PetscReal Max2(PetscReal a, PetscReal b, PetscInt &ind)		// max(a, b), sets ind=0|1 depending on who is max
{
	if (a < b)
	{
		ind = 1;
		return b;
	}
	else
	{
		ind = 0;
		return a;
	}
}
//---------------------------------------------------------------------------------------------
PetscErrorCode CalcBHP(Vec u, std::vector<double> &res, void *ctx)	// fills 'res' with data (BHP) derived from 'u'
{																	// 'res' will be accessible on rank-0
	PetscFunctionBeginUser;
	AppCtx *Ctx = (AppCtx*)ctx;
	const PetscScalar **p, **k;		// global
	PetscScalar P, K, mob;			// pressure, permeability, mobility at wells
	PetscInt xs, ys, xm, ym;		// corners

	res = std::vector<double>(2, 0.0);		// the result is reduced to 'res'
	std::vector<double> work = res;			// the distributed predecessor of 'res'

	const PetscScalar pi = PetscAcosScalar(-1.0);
	const PetscScalar r0 = PetscSqrtScalar(Ctx->dx*Ctx->dx + Ctx->dy*Ctx->dy) * 0.14;	// Peaceman
	const PetscScalar T = 2*pi*Ctx->C * Ctx->dz / PetscLogScalar(r0/Ctx->rw);

	PetscCall(DMDAGetCorners(Ctx->da, &xs, &ys, NULL, &xm, &ym, NULL));
	PetscCall(DMDAVecGetArrayRead(Ctx->da, u, &p));				// global
	PetscCall(DMDAVecGetArrayRead(Ctx->da, Ctx->perm, &k));		// global

	if (xs <= 0 && ys <= 0)							// prod
	{
		P = p[0][0];
		K = k[0][0];
		mob = PetscExpScalar(Ctx->cw*(P - Ctx->Pinit)) / Ctx->uw;
		work[0] = P - Ctx->Q0/(T*K*mob);
	}
	if (Ctx->Ni-1 < xs+xm && Ctx->Nj-1 < ys+ym)		// inj
	{
		P = p[Ctx->Nj-1][Ctx->Ni-1];
		K = k[Ctx->Nj-1][Ctx->Ni-1];
		mob = PetscExpScalar(Ctx->cw*(P - Ctx->Pinit)) / Ctx->uw;
		work[1] = P + Ctx->Q0/(T*K*mob);
	}

	PetscCall(DMDAVecRestoreArrayRead(Ctx->da, u, &p));
	PetscCall(DMDAVecRestoreArrayRead(Ctx->da, Ctx->perm, &k));

	PetscCallMPI(MPI_Reduce(work.data(), res.data(), res.size(), MPI_DOUBLE, MPI_SUM, 0, Ctx->comm));
	PetscFunctionReturn(0);
}
//---------------------------------------------------------------------------------------------
PetscErrorCode monitor(TS ts, PetscInt step, PetscReal time, Vec u, void *mctx)
{
	PetscFunctionBeginUser;

	std::vector<double> bhp;
	AppCtx *Ctx = (AppCtx*)mctx;

	PetscCallBack("CalcBHP", CalcBHP(u, bhp, mctx));
	PetscCheck(bhp.size() == 2, Ctx->comm, PETSC_ERR_ARG_SIZ, "bhp.size() == %zu", bhp.size());
	Ctx->wbhp.push_back(time);
	Ctx->wbhp.push_back(bhp[0]);
	Ctx->wbhp.push_back(bhp[1]);

	if (Ctx->savearr)
	{
		// save the pressure grid to vtk
		char buff[BUFFSIZE];
		PetscCall(PetscSNPrintf(buff, BUFFSIZE, "sim1_outP_%d.vts", step));
		PetscCallBack("SaveVecVtk", SaveVecVtk(u, buff, Ctx));

		// save the pressure grid to ascii
		PetscCall(PetscSNPrintf(buff, BUFFSIZE, "sim1_asciiP_%d.txt", step));
		PetscCallBack("SaveVecAscii", SaveVecAscii(u, buff, Ctx));
	}

	PetscFunctionReturn(0);
}
//---------------------------------------------------------------------------------------------
PetscErrorCode SaveVecVtk(Vec x, const char fname[], AppCtx *ctx)		// write grid to vtk file
{
	PetscFunctionBeginUser;

	PetscViewer vtk;

	PetscCall(PetscViewerVTKOpen(ctx->comm, fname, FILE_MODE_WRITE, &vtk));
	PetscCall(VecView(x, vtk));
	PetscCall(PetscViewerDestroy(&vtk));

	PetscFunctionReturn(0);
}
//---------------------------------------------------------------------------------------------
PetscErrorCode SaveVecAscii(Vec x, const char fname[], AppCtx *ctx)		// write grid to ascii file
{
	PetscFunctionBeginUser;

	PetscViewer va;
	PetscInt xs, ys, xm, ym;		// corners
	const PetscScalar **p;			// global

	PetscCall(PetscViewerASCIIOpen(ctx->comm, fname, &va));
	PetscCall(DMDAGetCorners(ctx->da, &xs, &ys, NULL, &xm, &ym, NULL));
	PetscCall(DMDAVecGetArrayRead(ctx->da, x, &p));						// global

	for (PetscInt j = ys; j < ys+ym; j++)
		for (PetscInt i = xs; i < xs+xm; i++)
			PetscCall(PetscViewerASCIIPrintf(va, "%.16g\n", p[j][i]));	// high precision output, only proc-0

	PetscCall(DMDAVecRestoreArrayRead(ctx->da, x, &p));

	// now, plain VecView
	PetscCall(PetscViewerASCIIPrintf(va, "------------------\n"));
	PetscCall(VecView(x, va));											// all procs

	PetscCall(PetscViewerDestroy(&va));

	PetscFunctionReturn(0);
}
//---------------------------------------------------------------------------------------------

