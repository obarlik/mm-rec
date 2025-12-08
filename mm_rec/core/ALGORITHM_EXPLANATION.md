# Paralel Asosiyatif Tarama: Log-Sum-Exp ile Stabilite

Bu doküman, yüksek boyutlu dil modelleri ve durumsal farkındalık mekanizmalarında kullanılan, özellikle Log-Sum-Exp (LSE) stabilizasyonu ile uygulanan Paralel Kümülatif İşlem (Associative Scan) çekirdeğinin mantığını açıklamaktadır.

---

## 1. Kümülatif İşlem Nedir? (Prefix Sum)

Kümülatif işlem, bir dizideki her eleman için, o noktaya kadarki tüm elemanların ikili bir asosiyatif operatör (örneğin toplama, çarpma, LSE) kullanılarak birleştirilmesiyle elde edilen değeri hesaplar:

$$\text{Y}_t = f(x_1, x_2, \ldots, x_t)$$

Sizin uygulamanızda, amaç **üstel çarpım** ($\prod_{i=1}^t \gamma_i$) hesaplamaktır. Çarpım, toplama kadar stabil olmadığından, işlem **logaritmik uzaya taşınır**:

1. **Giriş**: $\gamma_i$ (çarpım operatörünün öğeleri)
2. **Dönüşüm**: $x_i = \log(\gamma_i)$ (toplama operatörünün öğeleri)
3. **Tarama**: $\text{L}_{\text{sum}, t} = \sum_{i=1}^t x_i$ (Log-uzayda kümülatif toplam)
4. **Geri Dönüş**: $\text{Y}_t = \exp(\text{L}_{\text{sum}, t})$ (Lineer uzayda kümülatif çarpım)

---

## 2. Log-Sum-Exp (LSE) ile Sayısal Stabilite

Çekirdek, standart toplamadan farklı olarak, iki logaritmik değeri stabil bir şekilde birleştirmek için $\text{log}(\exp(a) + \exp(b))$ işlemini kullanır. Sayısal taşmayı ve hassasiyet kaybını önlemek için bu işlem, Triton kodunuzda tanımlanan şu özdeşlik kullanılarak yapılır:

$$\text{stable\_log\_sum\_exp}(a, b) = \max(a, b) + \log(1 + e^{-|a - b|})$$

### Neden Bu Formül?

1. **Overflow Önleme**: `max(a, b)` kullanarak büyük değerleri kontrol eder
2. **Underflow Önleme**: `exp(-|a - b|)` terimi küçük değerler için güvenlidir
3. **Hassasiyet**: `log1p(exp(-diff))` kullanımı sayısal hassasiyeti korur

Bu, özellikle **BF16 (Bfloat16)** gibi düşük hassasiyetli formatlarda çalışırken kritik öneme sahiptir.

---

## 3. Paralel Tarama Algoritması (Blelloch)

Triton çekirdeği, **Blelloch algoritması** olarak bilinen **iş verimli (work-efficient)** bir paralel önek toplama (prefix sum) algoritması kullanır. Bu algoritma, bir işlemci bloğu içindeki $N$ uzunluğundaki bir diziyi $\mathcal{O}(\log N)$ derinlikte (adımda) hesaplar.

### Aşama 1: Yukarı Tarama (Up-Sweep/Reduction)

Bu aşama, bir **indirgeme ağacı (reduction tree)** oluşturarak bloğun toplam kümülatif değerini (blok toplamını) hesaplar.

**Algoritma**:
1. **İkili Birleştirme**: Komşu öğeler ikili gruplar halinde birleştirilir ve sonuçlar her zaman sağdaki (veya çift) pozisyona yazılır
2. **Adım Boyutu İki Katına Çıkarma**: Her adımda birleştirme adımı iki katına çıkarılır (1, 2, 4, 8...)
3. **Sonuç**: Dizinin son pozisyonu, tüm blok verisinin kümülatif toplamını ($L_{\text{blok\_toplam}}$) içerir

**Örnek** (N=8):
```
Adım 1: [a, b, c, d, e, f, g, h]
        → [a, a+b, c, c+d, e, e+f, g, g+h]

Adım 2: [a, a+b, c, c+d, e, e+f, g, g+h]
        → [a, a+b, c, a+b+c+d, e, e+f, g, e+f+g+h]

Adım 3: [a, a+b, c, a+b+c+d, e, e+f, g, e+f+g+h]
        → [a, a+b, c, a+b+c+d, e, e+f, g, a+b+c+d+e+f+g+h]
```

### Aşama 2: Aşağı Tarama (Down-Sweep/Prefix Propagation)

Bu aşama, yukarı tarama sırasında oluşturulan indirgeme ağacını kullanarak her bir pozisyonun nihai önek toplamını hesaplar.

**Algoritma**:
1. **Başlangıç**: Ağacın kök (son) elemanı, sıfır veya önceki bloğun taşıma değeri (carry-in prefix) ile değiştirilir
2. **Önek Yayılımı**: Kökten yapraklara doğru ilerlenir. Bir düğümün önek toplamı, sol ve sağ alt ağaçlara yayılır:
   - **Sağ Alt Ağaç**: Ebeveynin önek toplamını alır
   - **Sol Alt Ağaç**: Ebeveynin önek toplamı ile sağ komşusunun (sol alt ağacın toplamı) kümülatif toplamı birleştirilerek bulunur
3. **Nihai Sonuç**: Aşağı tarama sonunda, her bir pozisyon $t$, tam ve nihai kümülatif toplamı ($Y_t$) içerir. Bu, $\text{önceki\_blok\_prefix} \oplus \text{kendi\_blok\_prefix}_t$ şeklinde hesaplanır

**Örnek** (N=8, önceki blok prefix = P):
```
Başlangıç: [a, a+b, c, a+b+c+d, e, e+f, g, P]

Adım 1: [a, a+b, c, P, e, e+f, g, P+a+b+c+d+e+f+g+h]
         → [a, a+b, c, P, e, e+f, g, P+a+b+c+d+e+f+g+h]

Adım 2: [a, a+b, c, P, e, P+a+b+c+d, g, P+a+b+c+d+e+f+g+h]
         → [a, a+b, P+a+b+c+d, P, e, P+a+b+c+d, P+a+b+c+d+e+f, P+a+b+c+d+e+f+g+h]

Adım 3: [P+a, P+a+b, P+a+b+c, P+a+b+c+d, P+a+b+c+d+e, P+a+b+c+d+e+f, P+a+b+c+d+e+f+g, P+a+b+c+d+e+f+g+h]
```

---

## 4. Bloklar Arası Taşıma (Carry-Over)

Uzun sekanslar için, diziyi birden fazla bloka böleriz. Her blok kendi içinde paralel olarak işlenir, ancak bloklar arası taşıma değeri (carry-over prefix) gereklidir:

1. **İlk Blok**: Carry-in yok (identity: 0.0 in log-space)
2. **Ara Bloklar**: Önceki bloğun toplam prefix'ini alır ve kendi prefix'lerine ekler
3. **Son Blok**: Carry-out gerekmez

Bu yaklaşım, blokların paralel işlenmesine izin verirken, tüm sekans boyunca doğru kümülatif toplamı garanti eder.

---

## 5. Performans ve Ölçeklenebilirlik

### Kompleksite
- **Zaman**: O(log N) paralel derinlik (blok içi)
- **İş**: O(N) toplam işlem
- **Bellek**: O(N) bellek kullanımı

### GPU Paralelliği
- **Warp/Wavefront Seviyesi**: Her warp kendi bloğunu işler
- **Block Seviyesi**: Birden fazla blok paralel olarak çalışabilir
- **Memory Coalescing**: Stride-based erişim desenleri optimize edilmiştir

### Ölçeklenebilirlik
- **Kısa Sekanslar** (N < 256): Küçük bloklar, düşük overhead
- **Orta Sekanslar** (256 ≤ N < 1024): Orta bloklar, dengeli performans
- **Uzun Sekanslar** (N ≥ 1024): Büyük bloklar (512-1024), maksimum paralellik

Bu iki aşamalı yaklaşım, özellikle uzun sekanslarda (uzun $N$ değerlerinde) otoregresif (seri) hesaplamaların darboğazını aşarak yüksek GPU paralelliği sağlar.

---

## 6. Sayısal Stabilite Garantileri

### Log-Space Clamping
- **Min**: -50.0 (prevents exp(-50) ≈ 0 underflow)
- **Max**: 0.0 (prevents exp(0) = 1 overflow)
- **Epsilon**: 1e-8 added before log to prevent log(0)

### Stable Exponential Computation
```python
# Pattern: exp(log_sum - max) * exp(max)
max_log = max(log_cumsum)
stable_exp = exp(log_cumsum - max_log) * exp(max_log)
```

### Special Cases
- **γ ≈ 0**: Complete decay → minimal contribution
- **γ ≈ 1**: No decay → full contribution
- **γ = 1**: Identity → no change

---

## Referanslar

- **Blelloch, G. E. (1990)**: "Prefix sums and their applications" - Work-efficient parallel scan algorithm
- **Higham, N. J. (2002)**: "Accuracy and Stability of Numerical Algorithms" - Log-Sum-Exp stability
- **Triton Documentation**: https://triton-lang.org/ - GPU kernel programming

