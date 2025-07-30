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
```
#### Part Two: 🚀 GROMACS 2024.5 Installation with CUDA 11.4 Support (User-Level Build) On Cluster: This guide describes step-by-step instructions to build **GROMACS 2024.5** from source with **CUDA 11.4** and **Thread-MPI** support, suitable for systems like DGX or university clusters **without root/sudo access**.


## 📦 Prerequisites

| Dependency | Version / Notes                            |
| ---------- | ------------------------------------------ |
| GCC/G++    | ≥ 7.4 (used: 9.4.0)                        |
| CUDA       | 11.4 (user-installed in `$HOME/cuda-11.4`) |
| CMake      | ≥ 3.16 (used: 3.28.3)                      |
| Python3    | ≥ 3.7 (used: 3.8.10, dev headers optional) |
```
```
### srun session environment:


```bash
srun -N 1 --ntasks-per-node=2 --gres=gpu:1 --mem=16000 --time=24:00:00 --partition=gpu_scholar --pty bash
srun -N 1 --ntasks=2 --cpus-per-task=10 --gres=gpu:4 --mem=64000 --time=48:00:00 --partition=gpu_scholar --pty bash
```
# To monitor GPU usage live:

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

## 📁 Step-by-Step Workflow



### 1. 📥 Download and Extract GROMACS 2024.5

```bash
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-2024.5.tar.gz
tar -xvzf gromacs-2024.5.tar.gz
cd gromacs-2024.5
```

```bash
cd ~/gromacs-2024.5
rm -rf build
mkdir build && cd build
export CUDA_HOME=$HOME/cuda-11.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$HOME/cmake-3.28.3-install/bin:$PATH
```

### 2. 🏗️ Configure CMake for GPU Build
```bash
which nvcc  
nvcc --version
which cmake
cmake --version
```

### 3 Run `cmake`:

```bash
cmake .. \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DGMX_GPU=CUDA \
  -DGMX_THREAD_MPI=ON \
  -DGMX_MPI=OFF \
  -DGMX_SIMD=AVX2_256 \
  -DGMX_BUILD_SHARED_EXE=ON \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_INSTALL_PREFIX=$HOME/gromacs-2024.5-install \
  -DCMAKE_DISABLE_FIND_PACKAGE_MPI=ON
```
### 4. ⚙️ Build, install and Source GROMACS Environment

```bash
make -j$(nproc)
make install
source $HOME/gromacs-2024.5-install/bin/GMXRC
echo 'source $HOME/gromacs-2024.5-install/bin/GMXRC' >> ~/.bashrc
gmx --version # You should see: ✅ Installation Complete!
```

### 5. 🧪 Optional: Run Benchmark/Test NVT, NPT and Final Production Run
```bash
export OMP_NUM_THREADS=10      # optional for 10 OpenMP threads per MPI rank (--cpus-per-task=10)
export CUDA_VISIBLE_DEVICES=0  # optional: force GPU0
gmx mdrun -deffnm NVT -nb gpu -pme gpu -v
gmx mdrun -deffnm NPT -nb gpu -pme gpu -v
```
### 6. Final Run: 

```bash
gmx grompp -f MD.mdp -c NPT.gro -t NPT.cpt -p topol.top -n index.ndx -maxwarn 2 -o MD.tpr
gmx mdrun -deffnm MD -nb gpu -pme gpu -v
gmx mdrun -deffnm MD -cpi MD.cpt -append \
  -ntmpi 2 -npme 1 -ntomp 8 -nb gpu -pme gpu -v
```
### 7. Job Log Check: 

```bash
tail -n 30 MD.log
grep -i "step" MD.log | tail -n 10
```
