// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_bond_break_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "comm_kokkos.h"
#include "error.h"
#include "fix_bond_history.h"
#include "force.h"
#include "modify.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"
#include "respa.h"
#include "update.h"
#include "utils.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixBondBreakKokkos<DeviceType>::FixBondBreakKokkos(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), nmax(0), breakcount(0), breakcounttotal(0)
{
  if (narg < 6) error->all(FLERR,"Illegal fix bond/break/kk command");

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0) error->all(FLERR, "Illegal fix bond/break/kk command");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  btype = utils::expand_type_int(FLERR, arg[4], Atom::BOND, lmp);
  double cutoff = utils::numeric(FLERR, arg[5], false, lmp);

  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/break/kk command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/break/kk command");

  cutsq = cutoff * cutoff;

  fraction = 1.0;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"prob") == 0) {
      if (iarg + 3 > narg) error->all(FLERR,"Illegal fix bond/break/kk command");
      fraction = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      int seed = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if (fraction < 0.0 || fraction > 1.0)
        error->all(FLERR,"Illegal fix bond/break/kk command");
      if (seed <= 0) error->all(FLERR,"Illegal fix bond/break/kk command");
      iarg += 3;
    } else error->all(FLERR,"Illegal fix bond/break/kk command");
  }

  if (fraction != 1.0)
    error->all(FLERR,"Fix bond/break/kk currently supports only prob 1.0");

  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix bond/break/kk with non-molecular systems");

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  neighborKK = (NeighborKokkos *) neighbor;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | TAG_MASK | MASK_MASK | BOND_MASK | SPECIAL_MASK | ANGLE_MASK;
  datamask_modify = BOND_MASK | SPECIAL_MASK | ANGLE_MASK;
  forward_comm_device = 1;
  reverse_comm_device = 1;

  comm_forward = MAX(2, 2+atom->maxspecial);
  comm_reverse = 2;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixBondBreakKokkos<DeviceType>::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixBondBreakKokkos<DeviceType>::init()
{
  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels;
  else nlevels_respa = 0;

  check_supported();

  angleflag = atom->nangles ? 1 : 0;
  lastcheck = -1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixBondBreakKokkos<DeviceType>::check_supported()
{
  if (fraction != 1.0)
    error->all(FLERR,"Fix bond/break/kk currently supports only prob 1.0");

  if (atom->map_style == 0)
    error->all(FLERR,"Fix bond/break/kk requires an atom map");

  if (atom->maxspecial > MAXSPECIAL_STACK)
    error->all(FLERR,"Fix bond/break/kk needs a larger device special-list buffer");

  if (atom->ndihedrals || atom->nimpropers)
    error->all(FLERR,"Fix bond/break/kk currently supports bonds and angles only");

  if (!modify->get_fix_by_style("BOND_HISTORY").empty())
    error->all(FLERR,"Fix bond/break/kk does not yet support bond history fixes");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixBondBreakKokkos<DeviceType>::grow_views()
{
  if (atom->nmax <= nmax) return;

  nmax = atom->nmax;
  k_partner = DAT::tdual_tagint_1d("bond/break/kk:partner", nmax);
  k_finalpartner = DAT::tdual_tagint_1d("bond/break/kk:finalpartner", nmax);
  k_error = DAT::tdual_int_1d("bond/break/kk:error", 1);
  k_distsq = DAT::tdual_float_1d("bond/break/kk:distsq", nmax);
  k_combined = Kokkos::DualView<unsigned long long*, LMPDeviceType>("bond/break/kk:combined", nmax);
}

template<class DeviceType>
void FixBondBreakKokkos<DeviceType>::post_integrate()
{
  if (update->ntimestep % nevery) return;

  check_supported();

  if (lastcheck < neighbor->lastcall) {
    atomKK->sync(Host, SPECIAL_MASK);
    check_ghosts();
  }

  // Match the base fix timing: updated ghost coordinates are needed here
  // because post_integrate runs before the normal Verlet communication.
  comm->forward_comm();

  atomKK->sync(execution_space, datamask_read);
  if (atom->map_style == 1) atomKK->k_map_array.sync<DeviceType>();
  else if (atom->map_style == 2) atomKK->k_map_hash.sync<DeviceType>();

  grow_views();

  nlocal = atom->nlocal;
  map_style = atom->map_style;
  maxspecial = atom->maxspecial;
  k_map_array = atomKK->k_map_array;
  k_map_hash = atomKK->k_map_hash;
  const auto commKK = dynamic_cast<CommKokkos *>(comm);
  const bool forward_comm_on_device =
    (execution_space != Host) && commKK && !commKK->forward_fix_comm_classic;
  const bool reverse_comm_on_device =
    (execution_space != Host) && commKK && reverse_comm_device &&
    !commKK->reverse_comm_classic && !commKK->reverse_comm_on_host;
  neighborKK->k_bondlist.sync<DeviceType>();
  bondlist = neighborKK->k_bondlist.view<DeviceType>();
  nbondlist = neighborKK->nbondlist;

  x = atomKK->k_x.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  nspecial = atomKK->k_nspecial.view<DeviceType>();
  special = atomKK->k_special.view<DeviceType>();
  num_bond = atomKK->k_num_bond.view<DeviceType>();
  bond_type = atomKK->k_bond_type.view<DeviceType>();
  bond_atom = atomKK->k_bond_atom.view<DeviceType>();
  num_angle = atomKK->k_num_angle.view<DeviceType>();
  angle_type = atomKK->k_angle_type.view<DeviceType>();
  angle_atom1 = atomKK->k_angle_atom1.view<DeviceType>();
  angle_atom2 = atomKK->k_angle_atom2.view<DeviceType>();
  angle_atom3 = atomKK->k_angle_atom3.view<DeviceType>();

  partner = k_partner.view<DeviceType>();
  finalpartner = k_finalpartner.view<DeviceType>();
  error_flag = k_error.view<DeviceType>();
  distsq = k_distsq.view<DeviceType>();
  combined = k_combined.view<DeviceType>();

  Kokkos::deep_copy(combined, (unsigned long long)0);

  int npossible = 0;
  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagFixBondBreakFind>(0,nbondlist),
                          *this, npossible);
  copymode = 0;

  int npossible_all;
  MPI_Allreduce(&npossible, &npossible_all, 1, MPI_INT, MPI_SUM, world);
  if (!npossible_all) {
    breakcount = 0;
    return;
  }

  Kokkos::deep_copy(finalpartner, (tagint) 0);
  Kokkos::deep_copy(error_flag, 0);

  // Extract partner tags and distsq from combined.
  // Packing rsq_f32_bits in the high 32 bits means atomic_max on the packed
  // value selects the bond with the largest distance: one atomic per bond,
  // no gap between distance and partner updates.
  {
    const int nall = nlocal + atom->nghost;
    auto lpartner  = partner;
    auto ldistsq   = distsq;
    auto ltag      = tag;
    auto lcombined = combined;
    Kokkos::parallel_for("bond/break/kk:extract",
        Kokkos::RangePolicy<DeviceType>(0, nall),
        KOKKOS_LAMBDA(int i) {
          const unsigned long long c = lcombined(i);
          if (c == 0) {
            lpartner(i) = 0;
            ldistsq(i)  = (LMP_FLOAT)0.0;
          } else {
            const int j = (int)(c & 0xFFFFFFFFULL);
            lpartner(i) = ltag(j);
            union { unsigned int u; float f; } b;
            b.u = (unsigned int)(c >> 32);
            ldistsq(i) = (LMP_FLOAT)b.f;
          }
        });
  }

  k_partner.modify<DeviceType>();
  if (!force->newton_bond && !forward_comm_on_device) k_partner.sync_host();
  if (force->newton_bond) {
    k_distsq.modify<DeviceType>();
    if (!reverse_comm_on_device) {
      k_partner.sync_host();
      k_distsq.sync_host();
    }
    comm->reverse_comm(this);
    if (reverse_comm_on_device) {
      k_partner.modify<DeviceType>();
      k_distsq.modify<DeviceType>();
      if (!forward_comm_on_device) k_partner.sync_host();
    } else {
      k_partner.modify<LMPHostType>();
      k_distsq.modify<LMPHostType>();
      if (forward_comm_on_device) k_partner.sync<DeviceType>();
    }
  }

  // Forward comm (commflag=1): propagate owned-atom partner choices to ghost
  // copies so the mutual-agreement check in Break works for remote partners.
  commflag = 1;
  comm->forward_comm(this, 1);
  if (forward_comm_on_device) {
    k_partner.modify<DeviceType>();
  } else {
    k_partner.modify<LMPHostType>();
    k_partner.sync<DeviceType>();
  }

  int nbreak = 0;
  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagFixBondBreakBreak>(0,nlocal),
                          *this, nbreak);
  copymode = 0;

  k_error.modify<DeviceType>();
  k_error.sync_host();
  if (k_error.h_view[0])
    error->all(FLERR,"Fix bond/break/kk found inconsistent bond topology");

  MPI_Allreduce(&nbreak, &breakcount, 1, MPI_INT, MPI_SUM, world);
  breakcounttotal += breakcount;
  atom->nbonds -= breakcount;

  if (breakcount) next_reneighbor = update->ntimestep;
  if (!breakcount) return;

  // Forward comm (commflag=2): send finalpartner and updated 1-2 specials
  // to ghost copies on other ranks so Angles and Special kernels are correct.
  k_finalpartner.modify<DeviceType>();
  atomKK->k_special.modify<DeviceType>();
  atomKK->k_nspecial.modify<DeviceType>();
  if (!forward_comm_on_device) {
    k_finalpartner.sync_host();
    atomKK->k_special.sync_host();
    atomKK->k_nspecial.sync_host();
  }
  commflag = 2;
  comm->forward_comm(this);
  if (forward_comm_on_device) {
    k_finalpartner.modify<DeviceType>();
    atomKK->k_special.modify<DeviceType>();
    atomKK->k_nspecial.modify<DeviceType>();
  } else {
    k_finalpartner.modify<LMPHostType>();
    k_finalpartner.sync<DeviceType>();
    atomKK->k_special.modify<LMPHostType>();
    atomKK->k_nspecial.modify<LMPHostType>();
    atomKK->k_special.sync<DeviceType>();
    atomKK->k_nspecial.sync<DeviceType>();
  }

  int nangles = 0;
  if (angleflag) {
    copymode = 1;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagFixBondBreakAngles>(0,nlocal),
                            *this, nangles);
    copymode = 0;

    int all;
    MPI_Allreduce(&nangles, &all, 1, MPI_INT, MPI_SUM, world);
    if (!force->newton_bond) all /= 3;
    atom->nangles -= all;
  }

  Kokkos::deep_copy(error_flag, 0);

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixBondBreakSpecial>(0,nlocal),
                       *this);
  copymode = 0;

  k_error.modify<DeviceType>();
  k_error.sync_host();
  if (k_error.h_view[0])
    error->all(FLERR,"Fix bond/break/kk special list exceeded maxspecial");

  atomKK->modified(execution_space, datamask_modify);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixBondBreakKokkos<DeviceType>::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa - 1) post_integrate();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double FixBondBreakKokkos<DeviceType>::compute_vector(int n)
{
  if (n == 0) return (double) breakcount;
  return (double) breakcounttotal;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double FixBondBreakKokkos<DeviceType>::memory_usage()
{
  double bytes = 2.0 * nmax * sizeof(tagint);
  bytes += (double) nmax * sizeof(LMP_FLOAT);
  bytes += (double) nmax * sizeof(unsigned long long);
  bytes += sizeof(int);
  return bytes;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int FixBondBreakKokkos<DeviceType>::map_atom(tagint global) const
{
  return AtomKokkos::map_kokkos<DeviceType>(global, map_style, k_map_array, k_map_hash);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixBondBreakKokkos<DeviceType>::operator()(TagFixBondBreakFind, const int &n,
                                                int &npossible) const
{
  const int i1 = bondlist(n,0);
  const int i2 = bondlist(n,1);
  const int type = bondlist(n,2);

  if (!(mask(i1) & groupbit)) return;
  if (!(mask(i2) & groupbit)) return;
  if (type != btype) return;

  const X_FLOAT delx = x(i1,0) - x(i2,0);
  const X_FLOAT dely = x(i1,1) - x(i2,1);
  const X_FLOAT delz = x(i1,2) - x(i2,2);
  const LMP_FLOAT rsq = delx*delx + dely*dely + delz*delz;
  if (rsq <= cutsq) return;

  npossible++;

  // Pack: high 32 bits = IEEE float bits of rsq (positive floats compare as uint),
  //       low  32 bits = local index of the partner atom.
  // A single atomic_max on the packed uint64 atomically selects the farthest bond.
  union { float f; unsigned int u; } bits;
  bits.f = (float)rsq;
  const unsigned long long val1 = ((unsigned long long)bits.u << 32) | (unsigned int)i2;
  const unsigned long long val2 = ((unsigned long long)bits.u << 32) | (unsigned int)i1;
  Kokkos::atomic_max(&combined(i1), val1);
  Kokkos::atomic_max(&combined(i2), val2);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixBondBreakKokkos<DeviceType>::operator()(TagFixBondBreakBreak, const int &i,
                                                int &nbreak) const
{
  const tagint ptag = partner(i);
  if (ptag == 0) return;

  const int j = map_atom(ptag);
  if (j < 0) return;
  if (partner(j) != tag(i)) return;

  const int nbond = num_bond(i);
  for (int m = 0; m < nbond; m++) {
    if (bond_atom(i,m) == ptag) {
      for (int k = m; k < nbond - 1; k++) {
        bond_atom(i,k) = bond_atom(i,k+1);
        bond_type(i,k) = bond_type(i,k+1);
      }
      num_bond(i) = nbond - 1;
      break;
    }
  }

  const int n1 = nspecial(i,0);
  const int n3 = nspecial(i,2);
  int m = 0;
  while (m < n1 && special(i,m) != ptag) m++;
  if (m == n1) {
    Kokkos::atomic_fetch_or(&error_flag(0), 1);
    return;
  }

  for (; m < n3 - 1; m++) special(i,m) = special(i,m+1);
  nspecial(i,0)--;
  nspecial(i,1)--;
  nspecial(i,2)--;

  finalpartner(i) = ptag;

  if (tag(i) < ptag) nbreak++;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
bool FixBondBreakKokkos<DeviceType>::broken_edge(tagint id1, tagint id2) const
{
  const int i1 = map_atom(id1);
  if (i1 >= 0 && finalpartner(i1) == id2) return true;

  const int i2 = map_atom(id2);
  if (i2 >= 0 && finalpartner(i2) == id1) return true;

  return false;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
bool FixBondBreakKokkos<DeviceType>::affected_by_break(int i) const
{
  if (finalpartner(i) != 0) return true;

  const int n3 = nspecial(i,2);
  for (int m = 0; m < n3; m++) {
    const int j = map_atom(special(i,m));
    if (j < 0) continue;

    const tagint ptag = finalpartner(j);
    if (ptag == 0) continue;

    for (int k = 0; k < n3; k++)
      if (special(i,k) == ptag) return true;
  }

  return false;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixBondBreakKokkos<DeviceType>::operator()(TagFixBondBreakAngles, const int &i,
                                                int &nangles) const
{
  if (!affected_by_break(i)) return;

  int nangle = num_angle(i);
  int m = 0;
  while (m < nangle) {
    const tagint id1 = angle_atom1(i,m);
    const tagint id2 = angle_atom2(i,m);
    const tagint id3 = angle_atom3(i,m);

    const bool found = broken_edge(id1,id2) || broken_edge(id2,id3);
    if (!found) {
      m++;
      continue;
    }

    for (int k = m; k < nangle - 1; k++) {
      angle_type(i,k) = angle_type(i,k+1);
      angle_atom1(i,k) = angle_atom1(i,k+1);
      angle_atom2(i,k) = angle_atom2(i,k+1);
      angle_atom3(i,k) = angle_atom3(i,k+1);
    }
    nangle--;
    nangles++;
  }

  num_angle(i) = nangle;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
bool FixBondBreakKokkos<DeviceType>::contains_tag(const tagint *list, int n, tagint value) const
{
  for (int i = 0; i < n; i++)
    if (list[i] == value) return true;
  return false;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixBondBreakKokkos<DeviceType>::append_unique(tagint *list, int &n, tagint value) const
{
  if (contains_tag(list,n,value)) return;
  if (n >= maxspecial || n >= MAXSPECIAL_STACK) {
    Kokkos::atomic_fetch_or(&error_flag(0), 1);
    return;
  }
  list[n++] = value;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixBondBreakKokkos<DeviceType>::operator()(TagFixBondBreakSpecial, const int &i) const
{
  if (!affected_by_break(i)) return;

  tagint copy[MAXSPECIAL_STACK];

  const tagint itag = tag(i);
  const int n1 = nspecial(i,0);

  int cn1 = 0;
  for (int m = 0; m < n1; m++) copy[cn1++] = special(i,m);

  int cn2 = cn1;
  for (int m = 0; m < cn1; m++) {
    const int j = map_atom(copy[m]);
    if (j < 0) continue;
    const int jn1 = nspecial(j,0);
    for (int k = 0; k < jn1; k++) {
      const tagint candidate = special(j,k);
      if (candidate != itag) append_unique(copy,cn2,candidate);
    }
  }

  int cn3 = cn2;
  for (int m = cn1; m < cn2; m++) {
    const int j = map_atom(copy[m]);
    if (j < 0) continue;
    const int jn1 = nspecial(j,0);
    for (int k = 0; k < jn1; k++) {
      const tagint candidate = special(j,k);
      if (candidate != itag) append_unique(copy,cn3,candidate);
    }
  }

  for (int m = cn1; m < cn3; m++) special(i,m) = copy[m];

  nspecial(i,0) = cn1;
  nspecial(i,1) = cn2;
  nspecial(i,2) = cn3;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixBondBreakKokkos<DeviceType>::check_ghosts()
{
  int **nspecial_h = atom->nspecial;
  tagint **special_h = atom->special;
  int nlocal_h = atom->nlocal;

  int flag = 0;
  for (int i = 0; i < nlocal_h; i++) {
    int n = nspecial_h[i][1];  // 1-3 neighbors: 2-hop coverage for special rebuild
    for (int j = 0; j < n; j++)
      if (atom->map(special_h[i][j]) < 0) flag = 1;
  }

  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);
  if (flagall)
    error->all(FLERR,"Fix bond/break/kk needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixBondBreakKokkos<DeviceType>::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0, last = first + n;
  for (int i = first; i < last; i++) {
    buf[m++] = ubuf(k_partner.h_view[i]).d;
    buf[m++] = k_distsq.h_view[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixBondBreakKokkos<DeviceType>::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    int j = list[i];
    if (buf[m+1] > k_distsq.h_view[j]) {
      k_partner.h_view[j] = (tagint) ubuf(buf[m]).i;
      k_distsq.h_view[j] = buf[m+1];
    }
    m += 2;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixBondBreakKokkos<DeviceType>::pack_reverse_comm_kokkos(
    int n, int first, DAT::tdual_xfloat_1d &buf)
{
  auto d_buf = buf.template view<DeviceType>();
  auto d_partner = k_partner.template view<DeviceType>();
  auto d_distsq = k_distsq.template view<DeviceType>();

  Kokkos::parallel_for("bond/break/kk:pack_reverse",
      Kokkos::RangePolicy<DeviceType>(0,n),
      KOKKOS_LAMBDA(const int ii) {
        const int i = first + ii;
        const int base = 2 * ii;
        d_buf(base) = d_ubuf(d_partner(i)).d;
        d_buf(base+1) = d_distsq(i);
      });
  buf.template modify<DeviceType>();
  return 2 * n;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixBondBreakKokkos<DeviceType>::unpack_reverse_comm_kokkos(
    int n, DAT::tdual_int_1d k_list, DAT::tdual_xfloat_1d &buf)
{
  auto list = k_list.template view<DeviceType>();
  auto d_buf = buf.template view<DeviceType>();
  auto d_partner = k_partner.template view<DeviceType>();
  auto d_distsq = k_distsq.template view<DeviceType>();

  Kokkos::parallel_for("bond/break/kk:unpack_reverse",
      Kokkos::RangePolicy<DeviceType>(0,n),
      KOKKOS_LAMBDA(const int ii) {
        const int j = list(ii);
        const int base = 2 * ii;
        const X_FLOAT rsq = d_buf(base+1);
        if (rsq > d_distsq(j)) {
          d_partner(j) = (tagint) d_ubuf(d_buf(base)).i;
          d_distsq(j) = (LMP_FLOAT) rsq;
        }
      });
  k_partner.template modify<DeviceType>();
  k_distsq.template modify<DeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixBondBreakKokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf,
                                                      int /*pbc_flag*/, int * /*pbc*/)
{
  int m = 0;
  if (commflag == 1) {
    for (int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = ubuf(k_partner.h_view[j]).d;
    }
    return m;
  }

  // commflag == 2: finalpartner + updated 1-2 special list
  for (int i = 0; i < n; i++) {
    int j = list[i];
    buf[m++] = ubuf(k_finalpartner.h_view[j]).d;
    int ns = atomKK->k_nspecial.h_view(j,0);
    buf[m++] = ubuf(ns).d;
    for (int k = 0; k < ns; k++)
      buf[m++] = ubuf(atomKK->k_special.h_view(j,k)).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixBondBreakKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
{
  int m = 0, last = first + n;
  if (commflag == 1) {
    for (int i = first; i < last; i++)
      k_partner.h_view[i] = (tagint) ubuf(buf[m++]).i;
    return;
  }

  // commflag == 2
  for (int i = first; i < last; i++) {
    k_finalpartner.h_view[i] = (tagint) ubuf(buf[m++]).i;
    int ns = (int) ubuf(buf[m++]).i;
    atomKK->k_nspecial.h_view(i,0) = ns;
    for (int j = 0; j < ns; j++)
      atomKK->k_special.h_view(i,j) = (tagint) ubuf(buf[m++]).i;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixBondBreakKokkos<DeviceType>::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d k_list, DAT::tdual_xfloat_1d &buf,
    int /*pbc_flag*/, int * /*pbc*/)
{
  auto list = k_list.template view<DeviceType>();
  auto d_buf = buf.template view<DeviceType>();

  if (commflag == 1) {
    auto d_partner = k_partner.template view<DeviceType>();
    Kokkos::parallel_for("bond/break/kk:pack_forward_partner",
        Kokkos::RangePolicy<DeviceType>(0,n),
        KOKKOS_LAMBDA(const int i) {
          const int j = list(i);
          d_buf(i) = d_ubuf(d_partner(j)).d;
        });
    buf.template modify<DeviceType>();
    return n;
  }

  // commflag == 2: fixed-stride device packing avoids a host-side
  // variable-length prefix scan.  The receiver reads only the sent nspecial.
  const int stride = comm_forward;
  auto d_finalpartner = k_finalpartner.template view<DeviceType>();
  auto d_nspecial = atomKK->k_nspecial.template view<DeviceType>();
  auto d_special = atomKK->k_special.template view<DeviceType>();
  Kokkos::parallel_for("bond/break/kk:pack_forward_special",
      Kokkos::RangePolicy<DeviceType>(0,n),
      KOKKOS_LAMBDA(const int i) {
        const int j = list(i);
        const int base = i * stride;
        d_buf(base) = d_ubuf(d_finalpartner(j)).d;
        const int ns = d_nspecial(j,0);
        d_buf(base+1) = d_ubuf(ns).d;
        for (int k = 0; k < ns; k++)
          d_buf(base+2+k) = d_ubuf(d_special(j,k)).d;
      });
  buf.template modify<DeviceType>();
  return stride * n;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixBondBreakKokkos<DeviceType>::unpack_forward_comm_kokkos(
    int n, int first, DAT::tdual_xfloat_1d &buf)
{
  auto d_buf = buf.template view<DeviceType>();

  if (commflag == 1) {
    auto d_partner = k_partner.template view<DeviceType>();
    Kokkos::parallel_for("bond/break/kk:unpack_forward_partner",
        Kokkos::RangePolicy<DeviceType>(0,n),
        KOKKOS_LAMBDA(const int ii) {
          const int i = first + ii;
          d_partner(i) = (tagint) d_ubuf(d_buf(ii)).i;
        });
    k_partner.template modify<DeviceType>();
    return;
  }

  const int stride = comm_forward;
  auto d_finalpartner = k_finalpartner.template view<DeviceType>();
  auto d_nspecial = atomKK->k_nspecial.template view<DeviceType>();
  auto d_special = atomKK->k_special.template view<DeviceType>();
  Kokkos::parallel_for("bond/break/kk:unpack_forward_special",
      Kokkos::RangePolicy<DeviceType>(0,n),
      KOKKOS_LAMBDA(const int ii) {
        const int i = first + ii;
        const int base = ii * stride;
        d_finalpartner(i) = (tagint) d_ubuf(d_buf(base)).i;
        const int ns = (int) d_ubuf(d_buf(base+1)).i;
        d_nspecial(i,0) = ns;
        for (int j = 0; j < ns; j++)
          d_special(i,j) = (tagint) d_ubuf(d_buf(base+2+j)).i;
      });
  k_finalpartner.template modify<DeviceType>();
  atomKK->k_nspecial.template modify<DeviceType>();
  atomKK->k_special.template modify<DeviceType>();
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixBondBreakKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixBondBreakKokkos<LMPHostType>;
#endif
}
