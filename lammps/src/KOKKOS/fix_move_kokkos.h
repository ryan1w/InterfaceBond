/* -*- c++ -*- ----------------------------------------------------------

------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(move/kk,FixMoveKokkos<LMPDeviceType>);
FixStyle(move/kk/device,FixMoveKokkos<LMPDeviceType>);
FixStyle(move/kk/host,FixMoveKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_MOVE_KOKKOS_H
#define LMP_FIX_MOVE_KOKKOS_H

#include "fix.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixMoveKokkosStoreOriginal {};
struct TagFixMoveKokkosInitialIntegrate {};
struct TagFixMoveKokkosFinalIntegrate {};
struct TagFixMoveKokkosUnpackExchange {};

template <class DeviceType> class FixMoveKokkos : public Fix, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixMoveKokkos(class LAMMPS *, int, char **);
  ~FixMoveKokkos() override;

  int setmask() override;
  void init() override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void initial_integrate_respa(int, int, int) override;
  void final_integrate_respa(int, int) override;

  double memory_usage() override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_restart(int, double *) override;
  void unpack_restart(int, int) override;
  int maxsize_restart() override;
  int size_restart(int) override;
  void reset_dt() override;

  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &) override;

  int pack_exchange_kokkos(const int &, DAT::tdual_xfloat_2d &, DAT::tdual_int_1d,
                           DAT::tdual_int_1d, ExecutionSpace) override;
  void unpack_exchange_kokkos(DAT::tdual_xfloat_2d &, DAT::tdual_int_1d &, int, int, int,
                              ExecutionSpace) override;

  KOKKOS_INLINE_FUNCTION void operator()(TagFixMoveKokkosStoreOriginal, const int &) const;
  KOKKOS_INLINE_FUNCTION void operator()(TagFixMoveKokkosInitialIntegrate, const int &) const;
  KOKKOS_INLINE_FUNCTION void operator()(TagFixMoveKokkosFinalIntegrate, const int &) const;
  KOKKOS_INLINE_FUNCTION void operator()(TagFixMoveKokkosUnpackExchange, const int &) const;
  KOKKOS_INLINE_FUNCTION void pack_exchange_item(const int &, int &, const bool &) const;

 private:
  int vxflag, vyflag, vzflag;
  int nlevels_respa, nrestart, time_origin;
  bigint ntimestep;
  double vx, vy, vz;
  double dt, dtv, dtf;

  int xperiodic, yperiodic, zperiodic, triclinic;
  double prd[3], prd_half[3], h[6];

  int nsend, nrecv1, nextrarecv1;

  DAT::tdual_x_array k_xoriginal;
  double **xoriginal;

  typename AT::t_x_array x;
  typename AT::t_x_array xorig;
  typename AT::t_v_array v;
  typename AT::t_f_array_const f;
  typename AT::t_float_1d rmass;
  typename AT::t_float_1d mass;
  typename AT::t_int_1d type;
  typename AT::t_int_1d mask;
  typename AT::t_imageint_1d image;

  typename AT::t_int_1d d_exchange_sendlist, d_copylist, d_indices;
  typename AT::t_xfloat_1d_um d_buf;

  typename AT::t_int_scalar d_count;
  HAT::t_int_scalar h_count;
};

template <class DeviceType> struct FixMoveKokkosPackExchangeFunctor {
  typedef DeviceType device_type;
  typedef int value_type;

  FixMoveKokkos<DeviceType> c;

  FixMoveKokkosPackExchangeFunctor(FixMoveKokkos<DeviceType> *c_ptr) : c(*c_ptr) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &i, int &offset, const bool &final) const
  {
    c.pack_exchange_item(i, offset, final);
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
