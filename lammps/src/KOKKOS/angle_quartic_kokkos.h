/* -*- c++ -*- ----------------------------------------------------------
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef ANGLE_CLASS
// clang-format off
AngleStyle(quartic/kk,AngleQuarticKokkos<LMPDeviceType>);
AngleStyle(quartic/kk/device,AngleQuarticKokkos<LMPDeviceType>);
AngleStyle(quartic/kk/host,AngleQuarticKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_ANGLE_QUARTIC_KOKKOS_H
#define LMP_ANGLE_QUARTIC_KOKKOS_H

#include "angle_quartic.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<int NEWTON_BOND, int EVFLAG>
struct TagAngleQuarticCompute {};

template<class DeviceType>
class AngleQuarticKokkos : public AngleQuartic {
 public:
  typedef DeviceType device_type;
  typedef EV_FLOAT value_type;
  typedef ArrayTypes<DeviceType> AT;

  AngleQuarticKokkos(class LAMMPS *);
  ~AngleQuarticKokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void read_restart(FILE *) override;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagAngleQuarticCompute<NEWTON_BOND, EVFLAG>, const int &, EV_FLOAT &) const;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagAngleQuarticCompute<NEWTON_BOND, EVFLAG>, const int &) const;

  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &, const int, const int, const int, F_FLOAT &, F_FLOAT *, F_FLOAT *,
                const F_FLOAT &, const F_FLOAT &, const F_FLOAT &, const F_FLOAT &,
                const F_FLOAT &, const F_FLOAT &) const;

  typename AT::tdual_efloat_1d k_eatom;
  typename AT::tdual_virial_array k_vatom;

 protected:
  class NeighborKokkos *neighborKK;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_2d anglelist;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int nlocal, newton_bond;
  int eflag, vflag;

  typename AT::tdual_ffloat_1d k_k2;
  typename AT::tdual_ffloat_1d k_k3;
  typename AT::tdual_ffloat_1d k_k4;
  typename AT::tdual_ffloat_1d k_theta0;

  typename AT::t_ffloat_1d d_k2;
  typename AT::t_ffloat_1d d_k3;
  typename AT::t_ffloat_1d d_k4;
  typename AT::t_ffloat_1d d_theta0;

  void allocate_kokkos();
  void sync_coeffs_to_kokkos();
};

}    // namespace LAMMPS_NS

#endif
#endif
