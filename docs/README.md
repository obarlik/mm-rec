# Dokümantasyon Rehberi

Bu dizin, dağınık durumdaki Markdown dosyalarını konu başlıklarına göre toplar. Kısa özet:

- `architecture/`: Çekirdek formül, teknik gereksinimler, model dönüştürme ve yapı belgeleri.
- `performance/`: Benchmark sonuçları, optimizasyon raporları ve hız/performans analizleri.
- `training/`: Eğitim planları, metodoloji, veri seti entegrasyonu ve kalite rehberleri.
- `testing/`: Test planları, kılavuzlar, performans ve validasyon raporları.
- `install/`: Kurulum, MKL rehberleri ve hızlı başlangıç dokümanları.
- `cpp/`: C++/CPU kütüphane durumu, planlar ve optimizasyon raporları.
- `integration/`: DPG/HEM/UBOO/OpenAI entegrasyon notları.
- `plans/`: Yol haritaları ve iyileştirme planları.
- `status/`: Proje durum özetleri, kararlar ve final raporları.
- `analysis/`: Kapsamlı analizler, uyumluluk ve hazır oluş raporları.
- `mlops/`: MLOps özel spesifikasyonlar.
- `misc/`: Diğer yardımcı notlar.

---

### 🔥 JAX Migration (Current Active Architecture)
Projects has pivoted to JAX for performance (>100 it/s).
- **Setup**: [Environment Setup](setup/environment.md)
- **Deployment**: [Git Workflow](workflow/deployment.md)
- **Architecture**: [JAX Migration Specs](architecture/jax_migration.md)

---

**Legacy Note**: Files referring to `mm_rec` (PyTorch) directly are now reference material. Active development is in `mm_rec_jax/`.
