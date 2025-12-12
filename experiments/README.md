# Experiments (Isolated Prototypes)

Bu klasör ana koda dokunmadan iki küçük prototip içerir:

1) `clustered_batches/demo.py`
- Sentetik embedding/uzunluk üretir.
- Üç senaryoyu karşılaştırır: rastgele batch, uzunluk bucketing, embedding+uzunluk greedy kümeleme.
- Metrikler (padding oranı, token/s) `experiments/results/clustered_batches.json` içine yazılır.
- Çalıştırma: `python experiments/clustered_batches/demo.py`

2) `pcgrad/demo.py`
- İki başlıklı küçük MLP ile iki kayıplı senaryoda PCGrad benzeri projeksiyon ile baseline’ı kıyaslar.
- Metrikler (loss, adım süresi) `experiments/results/pcgrad.json` içine yazılır.
- Çalıştırma: `python experiments/pcgrad/demo.py`

Ek araç:
- `clustered_batches/histogram.py`: Uzunluk dağılımı ve padding tahmini (sentetik veya verilen dosya ile). Örnek: `python experiments/clustered_batches/histogram.py --batch-size 32 --lengths-file lengths.txt`

Notlar:
- Ek bağımlılık yok; mevcut PyTorch/NumPy yeterli.
- Sonuç dosyaları `experiments/results/` altına yazılır; sürüm kontrolüne alınması gerekmiyorsa `.gitignore` ile hariç tutabilirsiniz.
