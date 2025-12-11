# MKL Hızlı Başlangıç

## En Kolay Yol: OpenBLAS Kurulumu

```bash
# OpenBLAS kur (MKL'ye alternatif, neredeyse aynı performans)
sudo apt install -y libopenblas-dev

# C++ extension'ı yeniden derle
cd mm_rec/cpp
python setup.py build_ext --inplace
```

## MKL Kurulumu (İsteğe Bağlı)

```bash
# MKL kur
sudo apt install -y intel-mkl libmkl-dev

# C++ extension'ı yeniden derle
cd mm_rec/cpp
python setup.py build_ext --inplace
```

## Kontrol

```bash
# Kurulum kontrolü
pkg-config --exists openblas && echo "✅ OpenBLAS kurulu" || echo "❌ OpenBLAS yok"
pkg-config --exists mkl && echo "✅ MKL kurulu" || echo "❌ MKL yok"
```

## Performans Testi

```bash
cd /home/onur/workspace/mm-rec
source venv/bin/activate
python -c "
import sys
sys.path.insert(0, 'mm_rec/cpp')
import mm_rec_blocks_cpu
print('✅ C++ extension yüklendi')
"
```

## Detaylı Bilgi

- `MKL_INSTALLATION_GUIDE.md` - Detaylı kurulum rehberi
- `install_mkl.sh` - Kurulum scripti
