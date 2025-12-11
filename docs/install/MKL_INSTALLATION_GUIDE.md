# MKL Kurulum Rehberi

**Tarih**: 2025-01-27  
**Durum**: MKL kurulum talimatları

## MKL Nedir?

Intel Math Kernel Library (MKL) - Yüksek performanslı matematik kütüphanesi
- C++ kütüphanesi
- Matrix operations için optimize edilmiş
- PyTorch'un da kullandığı kütüphane

## Kurulum Seçenekleri

### Seçenek 1: APT ile Kurulum (En Kolay) ✅ ÖNERİLEN

```bash
# Paket listesini güncelle
sudo apt update

# MKL runtime ve development paketlerini kur
sudo apt install -y intel-mkl libmkl-dev

# Kurulum sonrası kontrol
pkg-config --exists mkl && echo "✅ MKL kuruldu" || echo "❌ MKL bulunamadı"
```

**Avantajlar**:
- ✅ Kolay kurulum
- ✅ Sistem genelinde kullanılabilir
- ✅ Otomatik güncellemeler

**Dezavantajlar**:
- ⚠️ Ubuntu repository'lerinde olmayabilir (eski Ubuntu versiyonları)

### Seçenek 2: Intel oneAPI Base Toolkit (Tam Özellikli)

```bash
# 1. İndir (Intel web sitesinden)
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

# 2. Kurulum
chmod +x l_BaseKit_p_*.sh
sudo ./l_BaseKit_p_*.sh

# 3. Environment variables ayarla
source /opt/intel/oneapi/setvars.sh

# 4. Kalıcı yapmak için ~/.bashrc'ye ekle
echo 'source /opt/intel/oneapi/setvars.sh' >> ~/.bashrc
```

**Avantajlar**:
- ✅ En güncel versiyon
- ✅ Tüm Intel araçları dahil
- ✅ Resmi Intel desteği

**Dezavantajlar**:
- ⚠️ Büyük kurulum (~5GB)
- ⚠️ Daha karmaşık setup

### Seçenek 3: OpenBLAS (Alternatif - Daha Kolay) ✅ ÖNERİLEN ALTERNATİF

OpenBLAS, MKL'ye alternatif açık kaynak BLAS kütüphanesi:

```bash
# Kurulum
sudo apt install -y libopenblas-dev

# Kontrol
pkg-config --exists openblas && echo "✅ OpenBLAS kuruldu" || echo "❌ OpenBLAS bulunamadı"
```

**Avantajlar**:
- ✅ Çok kolay kurulum
- ✅ Açık kaynak (ücretsiz)
- ✅ MKL'ye yakın performans
- ✅ Ubuntu repository'lerinde mevcut

**Dezavantajlar**:
- ⚠️ MKL kadar optimize olmayabilir (ama çok yakın)

## Kurulum Sonrası Yapılacaklar

### 1. Environment Variables Ayarla

MKL için:
```bash
# ~/.bashrc veya ~/.zshrc'ye ekle
export MKLROOT=/usr/lib/x86_64-linux-gnu  # APT kurulumu için
# veya
export MKLROOT=/opt/intel/oneapi/mkl/latest  # oneAPI kurulumu için

export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH
export CPATH=$MKLROOT/include:$CPATH
```

OpenBLAS için:
```bash
# Genellikle otomatik, ama kontrol edin
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### 2. C++ Extension'ı Yeniden Derle

```bash
cd mm_rec/cpp
python setup.py build_ext --inplace
```

### 3. Test Et

```bash
python -c "
import sys
sys.path.insert(0, 'mm_rec/cpp')
import mm_rec_blocks_cpu
print('✅ C++ extension yüklendi')
"
```

## Hangi Seçeneği Seçmeliyim?

### MKL İstiyorsanız:
1. **APT ile kurulum** deneyin (en kolay)
2. Başarısız olursa **oneAPI Base Toolkit** kullanın

### Alternatif İstiyorsanız:
- **OpenBLAS** kullanın (daha kolay, neredeyse aynı performans)

## Kod Değişiklikleri

Kurulum sonrası, `setup.py` otomatik olarak MKL/OpenBLAS'i algılayacak ve kullanacak:

```python
# setup.py zaten hazır:
# - USE_MKL define edilirse MKL kullanır
# - USE_OPENBLAS define edilirse OpenBLAS kullanır
# - Yoksa manuel SIMD fallback kullanır
```

## Performans Beklentileri

MKL/OpenBLAS kurulumu sonrası:
- **Büyük problemler**: 10-50x speedup bekleniyor
- **Küçük problemler**: 2-5x speedup bekleniyor

## Sorun Giderme

### MKL bulunamıyor hatası:
```bash
# pkg-config ile kontrol
pkg-config --exists mkl || echo "MKL bulunamadı"

# Library path kontrol
ldconfig -p | grep mkl
```

### Derleme hatası:
```bash
# Include path kontrol
echo $CPATH
echo $MKLROOT

# Library path kontrol
echo $LD_LIBRARY_PATH
```

## Sonuç

**Öneri**: Önce **OpenBLAS** deneyin (en kolay), performans yeterli değilse **MKL** kurun.
