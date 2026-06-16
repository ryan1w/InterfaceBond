// clang-format off

#include "fix_move_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "domain_kokkos.h"
#include "error.h"
#include "force.h"
#include "lattice.h"
#include "memory_kokkos.h"
#include "respa.h"
#include "update.h"
#include "utils.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

template <class DeviceType>
FixMoveKokkos<DeviceType>::FixMoveKokkos(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), vxflag(0), vyflag(0), vzflag(0), nlevels_respa(0), nrestart(4),
    time_origin(update->ntimestep), ntimestep(update->ntimestep), vx(0.0), vy(0.0), vz(0.0),
    dt(0.0), dtv(0.0), dtf(0.0), xperiodic(0), yperiodic(0), zperiodic(0), triclinic(0),
    nsend(0), nrecv1(0), nextrarecv1(0),
    xoriginal(nullptr)
{
  if (narg < 4) utils::missing_cmd_args(FLERR, "fix move/kk", error);
  if (strcmp(arg[3], "linear") != 0)
    error->all(FLERR, 3, "Fix move/kk currently supports only linear style");
  if (narg < 7) utils::missing_cmd_args(FLERR, "fix move/kk linear", error);

  if (strcmp(arg[4], "NULL") == 0)
    vxflag = 0;
  else {
    vxflag = 1;
    vx = utils::numeric(FLERR, arg[4], false, lmp);
  }

  if (strcmp(arg[5], "NULL") == 0)
    vyflag = 0;
  else {
    vyflag = 1;
    vy = utils::numeric(FLERR, arg[5], false, lmp);
  }

  if (strcmp(arg[6], "NULL") == 0)
    vzflag = 0;
  else {
    vzflag = 1;
    vz = utils::numeric(FLERR, arg[6], false, lmp);
  }

  int scaleflag = 1;
  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "units") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix move/kk units", error);
      if (strcmp(arg[iarg + 1], "box") == 0)
        scaleflag = 0;
      else if (strcmp(arg[iarg + 1], "lattice") == 0)
        scaleflag = 1;
      else
        error->all(FLERR, iarg + 1, "Unknown fix move/kk units setting {}", arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "update") == 0) {
      error->all(FLERR, iarg, "Fix move/kk does not support the update keyword");
    } else {
      error->all(FLERR, iarg, "Unknown fix move/kk keyword {}", arg[iarg]);
    }
  }

  if (domain->dimension == 2 && vzflag && vz != 0.0)
    error->all(FLERR, 3, "Fix move/kk cannot set linear z motion for 2d problem");

  if (scaleflag) {
    if (vxflag) vx *= domain->lattice->xlattice;
    if (vyflag) vy *= domain->lattice->ylattice;
    if (vzflag) vz *= domain->lattice->zlattice;
  }

  restart_global = 1;
  restart_peratom = 1;
  peratom_flag = 1;
  size_peratom_cols = 3;
  peratom_freq = 1;
  time_integrate = 1;
  create_attribute = 1;
  maxexchange = 4;

  kokkosable = 1;
  exchange_comm_device = 1;
  sort_device = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = X_MASK | V_MASK | MASK_MASK;
  if (!vxflag || !vyflag || !vzflag) datamask_read |= F_MASK | RMASS_MASK | TYPE_MASK;
  datamask_modify = X_MASK | V_MASK;

  grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);

  d_count = typename AT::t_int_scalar("move/kk:count");
  h_count = Kokkos::create_mirror_view(d_count);

  prd[0] = domain->prd[0];
  prd[1] = domain->prd[1];
  prd[2] = domain->prd[2];
  for (int i = 0; i < 6; i++) h[i] = domain->h[i];
  triclinic = domain->triclinic;

  atomKK->sync(execution_space, X_MASK | IMAGE_MASK | MASK_MASK);
  x = atomKK->k_x.view<DeviceType>();
  image = atomKK->k_image.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  xorig = k_xoriginal.view<DeviceType>();

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixMoveKokkosStoreOriginal>(
                           0, atomKK->nlocal),
                       *this);
  copymode = 0;

  k_xoriginal.modify<DeviceType>();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> FixMoveKokkos<DeviceType>::~FixMoveKokkos()
{
  if (copymode) return;

  atom->delete_callback(id, Atom::GROW);
  atom->delete_callback(id, Atom::RESTART);

  memoryKK->destroy_kokkos(k_xoriginal, xoriginal);
  xoriginal = nullptr;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> int FixMoveKokkos<DeviceType>::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void FixMoveKokkos<DeviceType>::init()
{
  if (domain->triclinic)
    error->all(FLERR, "Fix move/kk linear currently supports only orthogonal simulation boxes");

  dt = update->dt;
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  if (!vxflag || !vyflag || !vzflag) {
    atomKK->k_mass.modify<LMPHostType>();
    atomKK->k_mass.sync<DeviceType>();
  }

  if (utils::strmatch(update->integrate_style, "^respa"))
    nlevels_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void FixMoveKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  atomKK->sync(execution_space, datamask_read);
  atomKK->modified(execution_space, datamask_modify);

  k_xoriginal.sync<DeviceType>();

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  xorig = k_xoriginal.view<DeviceType>();

  xperiodic = domain->xperiodic;
  yperiodic = domain->yperiodic;
  zperiodic = domain->zperiodic;
  prd[0] = domain->prd[0];
  prd[1] = domain->prd[1];
  prd[2] = domain->prd[2];
  prd_half[0] = 0.5 * domain->prd[0];
  prd_half[1] = 0.5 * domain->prd[1];
  prd_half[2] = 0.5 * domain->prd[2];
  ntimestep = update->ntimestep;

  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagFixMoveKokkosInitialIntegrate>(0, nlocal), *this);
  copymode = 0;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void FixMoveKokkos<DeviceType>::final_integrate()
{
  if (vxflag && vyflag && vzflag) return;

  atomKK->sync(execution_space, V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);
  atomKK->modified(execution_space, V_MASK);

  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagFixMoveKokkosFinalIntegrate>(0, nlocal), *this);
  copymode = 0;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixMoveKokkos<DeviceType>::initial_integrate_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa - 1) initial_integrate(vflag);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixMoveKokkos<DeviceType>::final_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa - 1) final_integrate();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void FixMoveKokkos<DeviceType>::operator()(
    TagFixMoveKokkosStoreOriginal, const int &i) const
{
  if (mask[i] & groupbit) {
    Few<double, 3> x_i;
    x_i[0] = x(i, 0);
    x_i[1] = x(i, 1);
    x_i[2] = x(i, 2);

    const auto unwrapped =
        DomainKokkos::unmap(Few<double, 3>(prd), Few<double, 6>(h), triclinic, x_i, image(i));

    xorig(i, 0) = unwrapped[0];
    xorig(i, 1) = unwrapped[1];
    xorig(i, 2) = unwrapped[2];
  } else {
    xorig(i, 0) = 0.0;
    xorig(i, 1) = 0.0;
    xorig(i, 2) = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void FixMoveKokkos<DeviceType>::operator()(
    TagFixMoveKokkosInitialIntegrate, const int &i) const
{
  if (!(mask[i] & groupbit)) return;

  const double delta = (ntimestep - time_origin) * dt;
  const double xold0 = x(i, 0);
  const double xold1 = x(i, 1);
  const double xold2 = x(i, 2);

  double dtfm = 0.0;
  const int nveflag = !vxflag || !vyflag || !vzflag;

  if (rmass.data()) {
    if (nveflag) dtfm = dtf / rmass(i);

    if (vxflag) {
      v(i, 0) = vx;
      x(i, 0) = xorig(i, 0) + vx * delta;
    } else {
      v(i, 0) += dtfm * f(i, 0);
      x(i, 0) += dtv * v(i, 0);
    }

    if (vyflag) {
      v(i, 1) = vy;
      x(i, 1) = xorig(i, 1) + vy * delta;
    } else {
      v(i, 1) += dtfm * f(i, 1);
      x(i, 1) += dtv * v(i, 1);
    }

    if (vzflag) {
      v(i, 2) = vz;
      x(i, 2) = xorig(i, 2) + vz * delta;
    } else {
      v(i, 2) += dtfm * f(i, 2);
      x(i, 2) += dtv * v(i, 2);
    }
  } else {
    if (nveflag) dtfm = dtf / mass(type(i));

    if (vxflag) {
      v(i, 0) = vx;
      x(i, 0) = xorig(i, 0) + vx * delta;
    } else {
      v(i, 0) += dtfm * f(i, 0);
      x(i, 0) += dtv * v(i, 0);
    }

    if (vyflag) {
      v(i, 1) = vy;
      x(i, 1) = xorig(i, 1) + vy * delta;
    } else {
      v(i, 1) += dtfm * f(i, 1);
      x(i, 1) += dtv * v(i, 1);
    }

    if (vzflag) {
      v(i, 2) = vz;
      x(i, 2) = xorig(i, 2) + vz * delta;
    } else {
      v(i, 2) += dtfm * f(i, 2);
      x(i, 2) += dtv * v(i, 2);
    }
  }

  if (xperiodic) {
    if (x(i, 0) - xold0 > prd[0]) {
      const int n = static_cast<int>((x(i, 0) - xold0) / prd[0]);
      x(i, 0) -= n * prd[0];
    }
    while (x(i, 0) - xold0 > prd_half[0]) x(i, 0) -= prd[0];
    if (xold0 - x(i, 0) > prd[0]) {
      const int n = static_cast<int>((xold0 - x(i, 0)) / prd[0]);
      x(i, 0) += n * prd[0];
    }
    while (xold0 - x(i, 0) > prd_half[0]) x(i, 0) += prd[0];
  }

  if (yperiodic) {
    if (x(i, 1) - xold1 > prd[1]) {
      const int n = static_cast<int>((x(i, 1) - xold1) / prd[1]);
      x(i, 1) -= n * prd[1];
    }
    while (x(i, 1) - xold1 > prd_half[1]) x(i, 1) -= prd[1];
    if (xold1 - x(i, 1) > prd[1]) {
      const int n = static_cast<int>((xold1 - x(i, 1)) / prd[1]);
      x(i, 1) += n * prd[1];
    }
    while (xold1 - x(i, 1) > prd_half[1]) x(i, 1) += prd[1];
  }

  if (zperiodic) {
    if (x(i, 2) - xold2 > prd[2]) {
      const int n = static_cast<int>((x(i, 2) - xold2) / prd[2]);
      x(i, 2) -= n * prd[2];
    }
    while (x(i, 2) - xold2 > prd_half[2]) x(i, 2) -= prd[2];
    if (xold2 - x(i, 2) > prd[2]) {
      const int n = static_cast<int>((xold2 - x(i, 2)) / prd[2]);
      x(i, 2) += n * prd[2];
    }
    while (xold2 - x(i, 2) > prd_half[2]) x(i, 2) += prd[2];
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void FixMoveKokkos<DeviceType>::operator()(
    TagFixMoveKokkosFinalIntegrate, const int &i) const
{
  if (!(mask[i] & groupbit)) return;

  double dtfm;
  if (rmass.data())
    dtfm = dtf / rmass(i);
  else
    dtfm = dtf / mass(type(i));

  if (!vxflag) v(i, 0) += dtfm * f(i, 0);
  if (!vyflag) v(i, 1) += dtfm * f(i, 1);
  if (!vzflag) v(i, 2) += dtfm * f(i, 2);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> double FixMoveKokkos<DeviceType>::memory_usage()
{
  return (double) atom->nmax * 3 * sizeof(double);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void FixMoveKokkos<DeviceType>::write_restart(FILE *fp)
{
  int n = 0;
  double list[1];
  list[n++] = time_origin;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(list, sizeof(double), n, fp);
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void FixMoveKokkos<DeviceType>::restart(char *buf)
{
  int n = 0;
  auto *list = (double *) buf;

  time_origin = static_cast<int>(list[n++]);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void FixMoveKokkos<DeviceType>::grow_arrays(int nmax)
{
  memoryKK->grow_kokkos(k_xoriginal, xoriginal, nmax, "move/kk:xoriginal");
  xorig = k_xoriginal.view<DeviceType>();
  array_atom = xoriginal;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixMoveKokkos<DeviceType>::copy_arrays(int i, int j, int /*delflag*/)
{
  k_xoriginal.sync_host();

  xoriginal[j][0] = xoriginal[i][0];
  xoriginal[j][1] = xoriginal[i][1];
  xoriginal[j][2] = xoriginal[i][2];

  k_xoriginal.modify_host();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void FixMoveKokkos<DeviceType>::set_arrays(int i)
{
  k_xoriginal.sync_host();

  if (!(atom->mask[i] & groupbit)) {
    xoriginal[i][0] = 0.0;
    xoriginal[i][1] = 0.0;
    xoriginal[i][2] = 0.0;
  } else {
    domain->unmap(atom->x[i], atom->image[i], xoriginal[i]);
    if (update->ntimestep != time_origin) {
      const double delta = (update->ntimestep - time_origin) * update->dt;
      if (vxflag) xoriginal[i][0] -= vx * delta;
      if (vyflag) xoriginal[i][1] -= vy * delta;
      if (vzflag) xoriginal[i][2] -= vz * delta;
    }
  }

  k_xoriginal.modify_host();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> int FixMoveKokkos<DeviceType>::pack_exchange(int i, double *buf)
{
  k_xoriginal.sync_host();

  int n = 0;
  buf[n++] = xoriginal[i][0];
  buf[n++] = xoriginal[i][1];
  buf[n++] = xoriginal[i][2];

  return n;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> int FixMoveKokkos<DeviceType>::unpack_exchange(int nlocal, double *buf)
{
  k_xoriginal.sync_host();

  int n = 0;
  xoriginal[nlocal][0] = buf[n++];
  xoriginal[nlocal][1] = buf[n++];
  xoriginal[nlocal][2] = buf[n++];

  k_xoriginal.modify_host();

  return n;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> int FixMoveKokkos<DeviceType>::pack_restart(int i, double *buf)
{
  k_xoriginal.sync_host();

  int n = 1;
  buf[n++] = xoriginal[i][0];
  buf[n++] = xoriginal[i][1];
  buf[n++] = xoriginal[i][2];
  buf[0] = n;

  return n;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void FixMoveKokkos<DeviceType>::unpack_restart(int nlocal, int nth)
{
  k_xoriginal.sync_host();

  double **extra = atom->extra;

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int>(extra[nlocal][m]);
  m++;

  xoriginal[nlocal][0] = extra[nlocal][m++];
  xoriginal[nlocal][1] = extra[nlocal][m++];
  xoriginal[nlocal][2] = extra[nlocal][m++];

  k_xoriginal.modify_host();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> int FixMoveKokkos<DeviceType>::maxsize_restart()
{
  return nrestart;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> int FixMoveKokkos<DeviceType>::size_restart(int /*nlocal*/)
{
  return nrestart;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void FixMoveKokkos<DeviceType>::reset_dt()
{
  error->all(FLERR, Error::NOLASTLINE, "Resetting timestep size is not allowed with fix move/kk");
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixMoveKokkos<DeviceType>::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  k_xoriginal.sync_device();

  Sorter.sort(LMPDeviceType(), k_xoriginal.d_view);

  k_xoriginal.modify_device();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void FixMoveKokkos<DeviceType>::pack_exchange_item(
    const int &mysend, int &offset, const bool & /*final*/) const
{
  const int i = d_exchange_sendlist(mysend);

  int m = nsend + offset;
  d_buf(mysend) = m;
  d_buf(m++) = xorig(i, 0);
  d_buf(m++) = xorig(i, 1);
  d_buf(m++) = xorig(i, 2);
  if (mysend == nsend - 1) d_count() = m;
  offset = m - nsend;

  const int j = d_copylist(mysend);
  if (j > -1) {
    xorig(i, 0) = xorig(j, 0);
    xorig(i, 1) = xorig(j, 1);
    xorig(i, 2) = xorig(j, 2);
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
int FixMoveKokkos<DeviceType>::pack_exchange_kokkos(const int &nsend,
                                                    DAT::tdual_xfloat_2d &k_buf,
                                                    DAT::tdual_int_1d k_exchange_sendlist,
                                                    DAT::tdual_int_1d k_copylist,
                                                    ExecutionSpace space)
{
  k_buf.sync<DeviceType>();
  k_exchange_sendlist.sync<DeviceType>();
  k_copylist.sync<DeviceType>();

  d_buf = typename AT::t_xfloat_1d_um(k_buf.template view<DeviceType>().data(),
                                      k_buf.extent(0) * k_buf.extent(1));
  d_exchange_sendlist = k_exchange_sendlist.view<DeviceType>();
  d_copylist = k_copylist.view<DeviceType>();
  this->nsend = nsend;

  k_xoriginal.sync<DeviceType>();
  xorig = k_xoriginal.view<DeviceType>();

  Kokkos::deep_copy(d_count, 0);

  copymode = 1;
  FixMoveKokkosPackExchangeFunctor<DeviceType> pack_exchange_functor(this);
  Kokkos::parallel_scan(nsend, pack_exchange_functor);
  copymode = 0;

  k_buf.modify<DeviceType>();
  if (space == Host)
    k_buf.sync<LMPHostType>();
  else
    k_buf.sync<LMPDeviceType>();

  k_xoriginal.modify<DeviceType>();

  Kokkos::deep_copy(h_count, d_count);
  return h_count();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void FixMoveKokkos<DeviceType>::operator()(
    TagFixMoveKokkosUnpackExchange, const int &i) const
{
  const int index = d_indices(i);

  if (index > -1) {
    int m = d_buf(i);
    if (i >= nrecv1) m = nextrarecv1 + d_buf(nextrarecv1 + i - nrecv1);

    xorig(index, 0) = d_buf(m++);
    xorig(index, 1) = d_buf(m++);
    xorig(index, 2) = d_buf(m++);
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixMoveKokkos<DeviceType>::unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf,
                                                       DAT::tdual_int_1d &k_indices, int nrecv,
                                                       int nrecv1, int nextrarecv1,
                                                       ExecutionSpace /*space*/)
{
  k_buf.sync<DeviceType>();
  k_indices.sync<DeviceType>();

  d_buf = typename AT::t_xfloat_1d_um(k_buf.template view<DeviceType>().data(),
                                      k_buf.extent(0) * k_buf.extent(1));
  d_indices = k_indices.view<DeviceType>();

  this->nrecv1 = nrecv1;
  this->nextrarecv1 = nextrarecv1;

  k_xoriginal.sync<DeviceType>();
  xorig = k_xoriginal.view<DeviceType>();

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagFixMoveKokkosUnpackExchange>(0, nrecv), *this);
  copymode = 0;

  k_xoriginal.modify<DeviceType>();
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixMoveKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixMoveKokkos<LMPHostType>;
#endif
}
