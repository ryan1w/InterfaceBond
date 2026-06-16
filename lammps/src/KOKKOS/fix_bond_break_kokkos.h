/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(bond/break/kk,FixBondBreakKokkos<LMPDeviceType>);
FixStyle(bond/break/kk/device,FixBondBreakKokkos<LMPDeviceType>);
FixStyle(bond/break/kk/host,FixBondBreakKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_BOND_BREAK_KOKKOS_H
#define LMP_FIX_BOND_BREAK_KOKKOS_H

#include "fix.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixBondBreakFind{};
struct TagFixBondBreakBreak{};
struct TagFixBondBreakAngles{};
struct TagFixBondBreakSpecial{};

template<class DeviceType>
class FixBondBreakKokkos : public Fix, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixBondBreakKokkos(class LAMMPS *, int, char **);
  int setmask() override;
  void init() override;
  void post_integrate() override;
  void post_integrate_respa(int, int) override;
  double compute_vector(int) override;
  double memory_usage() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_xfloat_1d &,
                                   int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_xfloat_1d &) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_xfloat_1d &) override;
  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_xfloat_1d &) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixBondBreakFind, const int&, int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixBondBreakBreak, const int&, int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixBondBreakAngles, const int&, int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixBondBreakSpecial, const int&) const;

 private:
  static constexpr int MAXSPECIAL_STACK = 64;

  int nevery, btype;
  double cutsq, fraction;
  int angleflag;
  int nlevels_respa;
  bigint lastcheck;
  int nmax;
  int breakcount;
  bigint breakcounttotal;
  int commflag;

  DAT::tdual_tagint_1d k_partner;
  DAT::tdual_tagint_1d k_finalpartner;
  DAT::tdual_int_1d k_error;
  DAT::tdual_float_1d k_distsq;
  Kokkos::DualView<unsigned long long*, LMPDeviceType> k_combined;

  class NeighborKokkos *neighborKK;

  typename AT::t_x_array_randomread x;
  typename AT::t_tagint_1d_randomread tag;
  typename AT::t_int_1d_randomread mask;
  typename AT::t_int_2d bondlist;
  typename AT::t_int_2d nspecial;
  typename AT::t_tagint_2d special;
  typename AT::t_int_1d num_bond;
  typename AT::t_int_2d bond_type;
  typename AT::t_tagint_2d bond_atom;
  typename AT::t_int_1d num_angle;
  typename AT::t_int_2d angle_type;
  typename AT::t_tagint_2d angle_atom1;
  typename AT::t_tagint_2d angle_atom2;
  typename AT::t_tagint_2d angle_atom3;

  typename AT::t_tagint_1d partner;
  typename AT::t_tagint_1d finalpartner;
  typename AT::t_int_1d error_flag;
  typename AT::t_float_1d distsq;
  Kokkos::View<unsigned long long*, typename DeviceType::memory_space> combined;

  int nlocal, nbondlist, map_style, maxspecial;
  DAT::tdual_int_1d k_map_array;
  dual_hash_type k_map_hash;

  void grow_views();
  void check_supported();
  void check_ghosts();

  KOKKOS_INLINE_FUNCTION
  int map_atom(tagint) const;
  KOKKOS_INLINE_FUNCTION
  bool broken_edge(tagint, tagint) const;
  KOKKOS_INLINE_FUNCTION
  bool affected_by_break(int) const;
  KOKKOS_INLINE_FUNCTION
  bool contains_tag(const tagint *, int, tagint) const;
  KOKKOS_INLINE_FUNCTION
  void append_unique(tagint *, int &, tagint) const;
};

}

#endif
#endif
