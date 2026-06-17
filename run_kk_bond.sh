#!/usr/bin/env bash

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

detected_gpus="$(nvidia-smi -L 2>/dev/null | wc -l)"
KOKKOS_GPUS="${KOKKOS_GPUS:-$detected_gpus}"
MPI_RANKS="${MPI_RANKS:-$KOKKOS_GPUS}"

exec mpirun -np "${MPI_RANKS}" ./lammps/build_kk/lmp \
  -k on g "${KOKKOS_GPUS}" t "${OMP_NUM_THREADS}" \
  -pk kokkos gpu/aware on newton on neigh half \
  -in in.make_interface_bonds_kk_bond
