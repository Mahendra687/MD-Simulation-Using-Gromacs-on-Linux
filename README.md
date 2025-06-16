# MD-Simulation-Using-Gromacs-on-Linux
A complete `README.md` formatted workflow for installing **GROMACS 2025.2** with **CUDA 12.4** and **MPI support** (ideal for GPU-based cluster systems like DGX A100). This version assumes you're installing GROMACS under a conda-managed environment (e.g., `gmxgpu`) on a Linux system with access to CUDA and MPI.

---

````markdown
# 🧬 GROMACS 2025.2 Installation (GPU + MPI + CUDA 12.4) on DGX/Linux

This guide walks you through building and installing GROMACS 2025.2 with:
- CUDA 12.4 GPU acceleration
- OpenMPI parallelism (gmx_mpi)
- Conda-based environment management

---

## 📦 Prerequisites

Before installing GROMACS, make sure the following are available:

- ✅ CUDA 12.4 driver installed on your system
- ✅ OpenMPI 4.x or 5.x
- ✅ GCC ≥ 9, CMake ≥ 3.15
- ✅ Conda (Miniconda or Anaconda)

---

## 🧪 1. Create Conda Environment

```bash
conda create -n gmxgpu -y \
    cmake=3.28 \
    gcc_linux-64=11 \
    gxx_linux-64=11 \
    openmpi=5.0.8 \
    fftw=3.3.10 \
    cuda-nvcc=12.4 \
    cuda-cudart-dev=12.4 \
    make \
    git \
    doxygen \
    ninja \
    mamba
conda activate gmxgpu
````

> 💡 Conda provides `mpicc`, `nvcc`, and all compiler dependencies.

---

## 📂 2. Download GROMACS Source

```bash
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-2025.2.tar.gz
tar -xvzf gromacs-2025.2.tar.gz
cd gromacs-2025.2
mkdir build && cd build
```

---

## ⚙️ 3. Configure with CMake

```bash
cmake .. \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DGMX_MPI=ON \
  -DGMX_GPU=CUDA \
  -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
  -DGMX_BUILD_OWN_FFTW=OFF \
  -DGMX_OPENMP=ON \
  -DGMX_USE_RDTSCP=ON \
  -DGMX_PREFER_STATIC_LIBS=OFF \
  -DGMX_SIMD=AVX2_256 \
  -DGMX_DOUBLE=OFF \
  -DGMX_BUILD_UNITTESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=/dgxb_home/USERNAME/apps/gromacs-2025.2-cuda12.4
```

---

## 🛠️ 4. Build and Install

```bash
make -j$(nproc)
make install
```

> Replace `$(nproc)` with `16` or `32` if building on a shared node to avoid overload.

---

## 🔁 5. Source GROMACS Environment

For bash users:

```bash
source /dgxb_home/USERNAME/apps/gromacs-2025.2-cuda12.4/bin/GMXRC.bash
```

You can add this line to your `.bashrc` or a modulefile if you're working on a shared cluster.

---

## ✅ 6. Test Your Installation

```bash
which gmx_mpi
gmx_mpi --version
```

Expected output:

```
GROMACS - gmx_mpi, 2025.2
Precision:           mixed
MPI library:         OpenMPI
GPU support:         CUDA
```

---

## 🧪 Example MPI Run (4 GPUs)

```bash
mpirun -np 4 gmx_mpi mdrun -deffnm md -ntomp 2 -gpu_id 0,1,2,3
```

---

## 💡 Optional: Create Alias

Add this to your `.bashrc` if you want `gmx` to point to `gmx_mpi`:

```bash
alias gmx="gmx_mpi"
```

---

## 🧹 Cleanup

To remove build files:

```bash
cd ~/gromacs-2025.2/
rm -rf build gromacs-2025.2.tar.gz
```

---

## 🧾 Notes

* `gmx_mpi` is used for both preparation and simulation commands.
* Always run `gmx_mpi mdrun` with `mpirun` or `srun` (for SLURM).
* Use `-ntomp` and `-gpu_id` to control threading and GPU assignment.

---

## 📚 Resources

* [GROMACS Manual](https://manual.gromacs.org/documentation/)
* [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
* [OpenMPI Documentation](https://www.open-mpi.org/doc/)

---

## 🙋 Troubleshooting

If `gmx` not found:

* This is expected in an MPI build. Use `gmx_mpi`.

If CUDA version mismatch:

* Verify that `nvcc --version` inside the conda env matches your system driver.

---

© 2025 BioInfo - Omics | Built for DGX-A100 systems | 🧬🖥️🚀

```
