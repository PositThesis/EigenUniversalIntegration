{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-22.05";
    universal_src = {
      url = "github:stillwater-sc/universal";
      flake = false;
    };
    #eigen_universal_source = {
    #    url = "path:./";
    #    flake = false;
    #};
    universal_patch = {
        url = "path:./patch";
        flake = false;
    };
    # eigen_patch = {
    #     url = "path:./patch";
    #     flake = false;
    # };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, universal_src, universal_patch, flake-utils }:
    let
      # wrap this in another let .. in to add the hydra job only for a single architecture
      output_set = flake-utils.lib.eachDefaultSystem (system:
        let
            pkgs = nixpkgs.legacyPackages.${system};
            extra_patch = if (system == "aarch64-linux") then
                "patch -p1 < ${universal_patch}/custom_extractor.patch"
            else "";
            eigen_patch_file = builtins.readFile ./patch/eigen_svd.patch;
        in
        rec {
            packages = flake-utils.lib.flattenTree {
                universal = pkgs.stdenv.mkDerivation {
                    name = "universal";
                    src = universal_src;

                    nativeBuildInputs = [pkgs.cmake];

                    prePatch = ''
                        cat CMakeLists.txt | sed "s/VERSION 3.23/VERSION 3.22/g" > CMakeLists_2.txt
                        mv CMakeLists_2.txt CMakeLists.txt

                    ''; #'' + extra_patch;

                    cmakeFlags = [
                        "-DBUILD_DEMONSTRATION=OFF"
                    ];
                };

                eigen = pkgs.eigen.overrideAttrs(old: {
                    prePatch = ''
                        echo '${eigen_patch_file}' > svd_patch.patch
                        patch -p1 < svd_patch.patch
                    '';
                    #prePatch = ''
                    #    patch -p1 < ${eigen_patch_file}/eigwn_svd.patch
                    #'';
                });

                eigen_universal_integration = pkgs.gcc10Stdenv.mkDerivation {
                    name = "EigenUniversalIntegration";
                    src = ./.;

                    nativeBuildInputs = [pkgs.cmake];

                    buildInputs = [
                        packages.universal
                        packages.eigen
                        pkgs.llvmPackages.openmp
                    ];

                    checkPhase = ''
                        ctest
                    '';

                    cmakeFlags = [ "-DTEST=ON" ];

                    doCheck = true;
                };
            };

            defaultPackage = packages.eigen_universal_integration;

            devShell = pkgs.mkShell {
                buildInputs = [
                    packages.universal
                    pkgs.eigen
                    pkgs.cmake
                    # pkgs.gdbgui
                ];

                shellHook = ''
                    function run_cmake_build() {
                        cd project
                        cmake -B /mnt/RamDisk/build -DDEBUG=ON -DTESTS=ON -DSOLVERS=ON -DCIM=ON
                        cmake --build /mnt/RamDisk/build -j1
                        cd ..
                    }

                    function build_and_run() {
                        run_cmake_build
                        /mnt/RamDisk/build/ShermanLoad
                    }
                    function build_and_test() {
                        run_cmake_build
                        pushd /mnt/RamDisk/build
                        ctest
                        popd
                    }
                '';
            };


        }
    );
    in
        output_set // { hydraJobs.build."aarch64-linux" = output_set.defaultPackage."aarch64-linux"; };
    }
