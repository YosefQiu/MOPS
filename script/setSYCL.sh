
module load intel/2024.1.0
module load cray-hdf5/1.12.2.9
module load cray-netcdf/4.9.0.9
module load cuda
module load craype-accel-nvidia80
module load cudatoolkit/11.7
module load libfabric/1.20.1

export MPICH_GPU_SUPPORT_ENABLED=1
export MFAB=/opt/cray/libfabric/1.20.1/lib64
export MMPI=/opt/cray/pe/mpich/8.1.25/ofi/cray/10.0/lib

