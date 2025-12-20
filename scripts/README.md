# Process Management Utilities

Bu dizin, `mm_rec` sunucunuzun kalıntı process'lerini temizlemek ve düzgün şekilde yönetmek için kullanışlı script'ler içerir.

## 🎯 Problem

Web arayüzü geliştirirken kalıntı process'ler kalıyor ve portlar bloke oluyor. Bu da:
- Port 8085'in meşgul olması
- Uygulamayı başlatamama
- Bilgisayarı yeniden başlatma zorunluluğu

gibi sorunlara yol açıyordu.

## ✅ Çözüm

### Çok Katmanlı Çözüm Sistemi

#### 1. **Graceful Shutdown (C++ Signal Handlers)**
- `mm_rec_cli.cpp` ve diğer C++ uygulamalara signal handler eklendi
- Ctrl+C (SIGINT), SIGTERM, SIGQUIT sinyalleri yakalanıyor
- Dashboard server düzgünce kapatılıyor
- Resource'lar temizleniyor

#### 2. **3-Fazlı Cleanup Stratejisi**
Script'ler üç farklı strateji kullanarak process'leri öldürüyor:

**Faz 1: Polite Shutdown (SIGTERM)**
- Process'e nazikçe sonlanma sinyali gönderilir
- 0.5 saniye beklenir

**Faz 2: Force Kill (SIGKILL)**
- Hala yaşayan process'ler `kill -9` ile zorla sonlandırılır
- 0.3 saniye beklenir

**Faz 3: Process Tree Cleanup**
- Tüm child process'ler öldürülür (`pkill -9 -P $pid`)
- Parent process öldürülür
- Binary adıyla da `killall` çalıştırılır

## 📝 Kullanım

### Hızlı Temizlik (Geliştirme İçin)
```bash
./scripts/quick_kill.sh
```
**Ne zaman kullanılır:** Geliştirme sırasında, hızlıca process'leri öldürüp yeniden başlatmak için.

### Detaylı Temizlik
```bash
./scripts/cleanup_processes.sh
```
**Ne zaman kullanılır:** Zombie process'ler varsa veya cleanup'ın başarılı olduğundan emin olmak için.

**Çıktı Örneği:**
```
========================================
  MM-REC Process Cleanup Utility
  [ENHANCED - Multi-Strategy Kill]
========================================

Phase 1: Killing processes by pattern...
⚠  Found demo_training_cpp processes (PIDs: 12345)
  → Attempting graceful shutdown (SIGTERM)...
  ✓ Gracefully terminated

Phase 2: Killing by binary name...
✓ Killed all demo_training_cpp instances

Phase 3: Freeing ports...
⚠  Port 8085 in use (PIDs: 12345)
  → Killing process 12345 using port 8085...
  ✓ Port 8085 freed successfully

Phase 4: Final verification...
========================================
  ✓ All Clean!
========================================

✓ All mm_rec processes terminated successfully
```

### Server Başlatma (Otomatik Cleanup ile)
```bash
./scripts/start_server.sh
```
**Ne zaman kullanılır:** Server'ı temiz bir şekilde başlatmak için. Otomatik olarak önce cleanup yapar.

**Özellikler:**
- Önceki process'leri otomatik temizler
- Port 8085'in müsait olduğunu kontrol eder
- Server'ı başlatır
- Ctrl+C ile durdurunca otomatik cleanup yapar

### Hızlı Yeniden Başlatma
```bash
./scripts/quick_restart.sh
```
**Ne zaman kullanılır:** Kod değişikliği yaptıktan sonra, tek komutla yeniden başlatmak için.

## 🛡️ Zombie Process'ler

Eğer script'ler bile process'leri öldüremezse (çok nadir):

```bash
# Sudo ile dene
sudo ./scripts/cleanup_processes.sh

# Eğer hala kalıyorsa, process state'ini kontrol et
ps aux | grep mm_rec

# D state (uninterruptible sleep) var mı bak
# D state varsa, I/O blocking var demektir
sudo iotop -o
```

## 🔧 Teknik Detaylar

### Signal Handler Implementasyonu

`mm_rec_cli.cpp` içinde:
```cpp
volatile std::sig_atomic_t g_shutdown_requested = 0;

void signal_handler(int signal) {
    if (g_shutdown_requested) {
        // İkinci sinyal - zorla çık
        std::_Exit(1);
    }
    
    g_shutdown_requested = 1;
    // Dashboard'u durdur
    DashboardManager::instance().stop();
    std::exit(0);
}
```

### Yakalanan Sinyaller
- `SIGINT` (Ctrl+C)
- `SIGTERM` (kill komutu)
- `SIGQUIT` (Ctrl+\)

## 🎨 Script Özellikleri

### Renkli Çıktı
- 🔵 Mavi: Başlıklar
- 🟢 Yeşil: Başarılı işlemler
- 🟡 Sarı: Uyarılar
- 🔴 Kırmızı: Hatalar
- 🟣 Mor: İşlem detayları

### Güvenli Hata Yönetimi
- Her komut `|| true` ile çalışır (hata olsa bile devam eder)
- Process bulunmazsa hata vermez
- Port zaten boşsa hata vermez

## 🚀 Önerilen Workflow

**Geliştirme sırasında:**
```bash
# 1. Kod değiştir
# 2. Hızlı restart
./scripts/quick_restart.sh
```

**Sorun yaşadığında:**
```bash
# 1. Detaylı cleanup
./scripts/cleanup_processes.sh

# 2. Manuel başlat
./scripts/start_server.sh
```

**Kalıcı zombie process:**
```bash
# Önce normal cleanup dene
./scripts/cleanup_processes.sh

# Çalışmazsa sudo ile
sudo ./scripts/cleanup_processes.sh

# Hala çalışmazsa reboot
sudo reboot
```

## 📊 Process Monitoring

Server çalışırken process'leri izlemek için:

```bash
# mm_rec process'lerini listele
ps aux | grep mm_rec

# Port 8085'i kullanan process
lsof -i :8085

# Tüm dinleyen portlar
ss -tulpn | grep LISTEN
```

## ⚡ Performans Notları

- **quick_kill.sh**: ~100ms (en hızlı, verbose çıktı yok)
- **cleanup_processes.sh**: ~1-2 saniye (detaylı, 3 fazlı)
- **start_server.sh**: ~2-3 saniye (cleanup + başlatma)
- **quick_restart.sh**: ~1 saniye (sessiz cleanup + başlatma)

## 🎁 Bonus: Alias Önerileri

`.bashrc` veya `.zshrc` dosyanıza ekleyin:

```bash
# mm_rec shortcuts
alias mmkill='~/workspace/mm-rec/scripts/quick_kill.sh'
alias mmstart='~/workspace/mm-rec/scripts/start_server.sh'
alias mmrestart='~/workspace/mm-rec/scripts/quick_restart.sh'
alias mmclean='~/workspace/mm-rec/scripts/cleanup_processes.sh'
```

Artık sadece `mmrestart` yazarak server'ı yeniden başlatabilirsiniz!
