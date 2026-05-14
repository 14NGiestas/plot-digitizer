export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
nix develop -c digitizer generate --output-dir synthetic-data --count 1000 --workers $(nproc)

# 2. Desativa a colisão de threads na CPU para os workers do DataLoader
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
nix develop .#rocm -c digitizer train --execute --epochs 300 --dataset-dir synthetic-data --workers 2 --batch 32
echo "Pressione Enter para fechar..." ; read
