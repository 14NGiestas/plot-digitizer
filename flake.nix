{
  description = "Nix flake for plot-digitizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # required for CUDA packages
        };
        python = pkgs.python312;
        commonSystemLibs = with pkgs; [
          pkgs.libxcb
        ];
        packagedCli = python.pkgs.buildPythonApplication {
          pname = "digitizer";
          version = "0.1.0";
          format = "pyproject";
          src = self;

          nativeBuildInputs = with python.pkgs; [
            hatchling
          ];

          buildInputs = commonSystemLibs;

          propagatedBuildInputs = with python.pkgs; [
            matplotlib
            numpy
            opencv4
            pandas
            scikit-image
            scikit-learn
            scipy
          ];

          pythonRemoveDeps = [
            "opencv-python"
          ];

          doCheck = true;
          checkPhase = ''
            ${python.interpreter} -m unittest discover -s tests -p 'test_*.py' -v
          '';
        };
        shellPythonPathHook = ''
          if [ -d "$PWD/src/digitizer" ]; then
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
          else
            export PYTHONPATH="${self}/src''${PYTHONPATH:+:$PYTHONPATH}"
          fi
        '';

        # Core Python packages shared across all shells
        corePythonPkgs = ps: with ps; [
          hatchling
          matplotlib
          numpy
          opencv4
          pandas
          pytest
          scikit-image
          scikit-learn
          scipy
        ];

        # Factory: build a dev shell with optional extra system packages and hook
        mkPyShell = { extraPkgs ? [], shellHook ? "" }: pkgs.mkShell {
          packages = [
            (python.withPackages corePythonPkgs)
            pkgs.uv
          ] ++ commonSystemLibs ++ extraPkgs;
          shellHook = shellPythonPathHook + shellHook;
        };

        # GPU-specific shells are only meaningful on Linux
        gpuShells = pkgs.lib.optionalAttrs pkgs.stdenv.isLinux (
          let
            # --- ROCm / HIP (AMD GPU) ---
            rocmLibs = with pkgs.rocmPackages; [
              rocm-runtime  # HSA runtime  (libhsa-runtime64.so)
              clr           # HIP + OpenCL (libamdhip64.so)
              rocblas       # ROCm BLAS
            ];

            # --- CUDA (NVIDIA GPU) ---
            cudaLibs = with pkgs.cudaPackages; [
              cuda_cudart  # CUDA runtime
              libcublas    # cuBLAS
            ];
          in
          {
            # AMD GPU — ROCm/HIP
            # Tested on: Ryzen 7 8745HS (Hawk Point) with Radeon 780M iGPU (gfx1103 / RDNA3).
            # The iGPU shares system RAM (DDR5) as both CPU and GPU memory.
            #
            # After entering this shell, install PyTorch for ROCm via:
            #   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
            rocm = mkPyShell {
              extraPkgs = rocmLibs;
              shellHook = ''
                export ROCM_PATH="${pkgs.rocmPackages.rocm-runtime}"
                export HIP_PATH="${pkgs.rocmPackages.clr}"
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath rocmLibs}:$LD_LIBRARY_PATH"
                # Hawk Point APU (Ryzen 8000-series) integrates Radeon 780M — gfx1103.
                # HSA_OVERRIDE_GFX_VERSION forces the correct ISA when the ROCm runtime
                # cannot auto-detect the iGPU (common on newer APUs).
                export HSA_OVERRIDE_GFX_VERSION="11.0.3"
                echo "ROCm shell ready (gfx1103 / Radeon 780M)."
                echo "Install PyTorch for ROCm with:"
                echo "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2"
              '';
            };

            # NVIDIA GPU — CUDA
            # After entering this shell, install PyTorch for CUDA via:
            #   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
            cuda = mkPyShell {
              extraPkgs = cudaLibs;
              shellHook = ''
                export CUDA_PATH="${pkgs.cudaPackages.cuda_cudart}"
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath cudaLibs}:$LD_LIBRARY_PATH"
                echo "CUDA shell ready."
                echo "Install PyTorch for CUDA with:"
                echo "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
              '';
            };
          }
        );
      in
      {
        packages.default = packagedCli;

        apps.default = {
          type = "app";
          program = "${pkgs.writeShellScript "digitizer-app" ''
            if [ -n "''${IN_NIX_SHELL:-}" ] && command -v python >/dev/null 2>&1 && python -c "import digitizer" >/dev/null 2>&1; then
              exec python -m digitizer "$@"
            fi
            exec ${packagedCli}/bin/digitizer "$@"
          ''}";
        };

        devShells = {
          # CPU-only shell — works on all platforms, used in CI
          default  = mkPyShell {};
          cpu-only = mkPyShell {};
        } // gpuShells;
      }
    );
}
