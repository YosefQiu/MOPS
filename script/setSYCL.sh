module load intel/2024.1.0
module load cudatoolkit/12.4
module load craype-accel-nvidia80
module load libfabric/1.22.0

export MPICH_GPU_SUPPORT_ENABLED=1
export MFAB=/opt/cray/libfabric/1.22.0/lib64
export MMPI=/opt/cray/pe/mpich/8.1.25/ofi/cray/10.0/lib

