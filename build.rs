fn main() {
    cfg_select! {
        feature = "cuda" => {
            use std::env;
            use std::path::PathBuf;
            use std::process::Command;

            // 1. Définir le fichier racine CUDA et le nom de sortie
            let kernel_file = "src/cuda/lib.cu"; // Votre fichier qui inclut les autres .cu
            let out_dir = env::var("OUT_DIR").unwrap();
            let ptx_file = PathBuf::from(&out_dir).join("kernels.ptx");

            // 2. Informer Cargo de recompiler si les fichiers .cu ou .cuh changent
            println!("cargo:rerun-if-changed=src/cuda/");

            // 3. Exécuter NVCC
            // On utilise les flags validés pour votre Fedora 43
            let status = Command::new("nvcc")
                .arg("-ptx")
                .arg("-ccbin")
                .arg("/usr/bin/gcc-15")
                .arg("-I")
                .arg("src/cuda")
                .arg("-o")
                .arg(&ptx_file)
                .arg(kernel_file)
                .status()
                .expect("Échec du lancement de nvcc");

            if !status.success() {
                panic!(
                    "La compilation CUDA a échoué. Vérifiez les erreurs de syntaxe dans vos fichiers .cu"
                );
            }

            // 4. Passer le chemin du PTX au programme via une variable d'environnement
            // Cela permet de faire include_str!(env!("KERNEL_PTX_PATH")) dans votre main.rs
            println!("cargo:rustc-env=KERNEL_PTX_PATH={}", ptx_file.display());
        }
        _ => {

        }
    }
}
