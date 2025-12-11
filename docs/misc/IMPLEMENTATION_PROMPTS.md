# MM-Rec Implementation Prompts
## LLM-Assisted Development Guide

Bu doküman, MM-Rec projesinin eksik bileşenlerini LLM (Büyük Dil Modeli) ile adım adım geliştirmek için hazırlanmış detaylı prompt'ları içerir.

Her prompt, bir bileşenin tam implementasyonu için gerekli tüm bilgileri içerir.

---

## 🎯 Faz 1: Çekirdek Bileşenler (Core Components)

### 1. Memory State Management (Bellek Durum Yönetimi)

**Dosya:** `mm_rec/core/memory_state.py`

**Prompt:**

```
Şu an MM-Rec modelinin temel veri yapılarını oluşturuyoruz. Lütfen mm_rec/core/memory_state.py dosyasını oluştur. Bu dosya, modelin kısa ve uzun vadeli belleklerini yönetmelidir.

**Gereksinimler:**

1. **MemoryBank Sınıfı:** Tek bir bellek birimini (short-term veya long-term) temsil eden `MemoryBank` adında bir Python/PyTorch sınıfı oluştur.

   * **__init__**: `k_dim`, `v_dim`, `num_slots` (bellek yuvası sayısı) ve `dtype` (varsayılan olarak `torch.bfloat16`) parametrelerini almalı.
   * **self.k (Key)** ve **self.v (Value)** olmak üzere iki PyTorch tensörünü başlatmalı. Bunlar `k` ve `v` boyutlarına ve `num_slots` sayısına sahip olmalı. Tensörler CPU veya GPU'da tutulabilir (cihaz parametresi eklenebilir).
   * **Fonksiyon**: `initialize_bank(self, num_slots)`: Bankayı sıfır tensörlerle veya Gaussian dağılımıyla başlatmalı.

2. **MemoryState Sınıfı:** Modelin genel bellek durumunu yöneten `MemoryState` adında bir sınıf oluştur.

   * **__init__**: `short_term_config` ve `long_term_config` olmak üzere iki ayrı yapılandırma sözlüğünü almalı.
   * **self.short_term** ve **self.long_term** adında iki `MemoryBank` örneğini başlatmalı.
   * **Fonksiyon**: `get_state(self, bank_type: str) -> tuple[torch.Tensor, torch.Tensor]`: Belirtilen banka tipinin (örneğin 'short') `(k, v)` tensörlerini döndürmeli.
   * **Fonksiyon**: `update_state(self, bank_type: str, new_k: torch.Tensor, new_v: torch.Tensor)`: Belirtilen bankanın `k` ve `v` tensörlerini yeni tensörlerle değiştirmeli.
   * **Fonksiyon**: `to_device(self, device)`: Tüm bellek bankalarındaki tensörleri belirtilen cihaza taşımalı.

**Ek Notlar:**
- MemoryBank ve MemoryState sınıfları PyTorch'un `nn.Module`'ünden türetilmeli (eğer parametre içeriyorsa)
- Short-term memory: `[batch, seq_len, hidden_dim]` formatında
- Long-term memory: `[batch, num_memories, M, mem_dim]` formatında (M << seq_len)
- Referans: ENGINEERING_OUTPUTS.md bölüm 4.1 ve CODE_STRUCTURE.md
```

---

### 2. MDI (Memory Decay/Integration)

**Dosya:** `mm_rec/core/mdi.py`

**Prompt:**

```
MM-Rec modelinin bellek bozunumu ve entegrasyon mantığını uygulamamız gerekiyor. Lütfen mm_rec/core/mdi.py dosyasını oluştur. Bu modül, Associative Scan'dan gelen mantığı kullanarak bellek güncelleme kapılarını (gated integration) yönetecek.

**Gereksinimler:**

1. **MemoryDecayIntegration (MDI) Sınıfı:** Bir PyTorch `nn.Module` olarak `MemoryDecayIntegration` adında bir sınıf oluştur. Bu sınıf, modelin bir katmanındaki (layer) bellek güncelleme mekanizmasını temsil etmelidir.

   * **__init__**: `model_dim` ve `inner_dim` parametrelerini almalı.
   * **self.W_g (Gating Ağırlığı)**: Yeni gelen girdi (`z_t`) ile eski durumun (`h_{t-1}`) ne kadarının birleştirileceğini kontrol eden bir lineer katman (`nn.Linear`) tanımla.
   * **self.W_gamma (Decay Ağırlığı)**: Bozunum katsayısı γ'yı öğrenebilmek için `model_dim`'den `inner_dim`'e bir lineer katman tanımla.
   * **self.W_context (Modülasyon Ağırlığı)**: Gated entegrasyonu kontekste bağımlı hale getirmek için bir lineer katman tanımla (opsiyonel).

2. **İleri Geçiş (Forward) Metodu:** `forward(self, z_t: torch.Tensor, h_prev: torch.Tensor, context: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]` metodunu oluştur.

   * **Gated Entegrasyon Hesaplaması (z_t ve h_prev)**:
     * $g = \sigma(W_g \cdot [\text{z\_t}, \text{h\_prev}])$ (Burada $\sigma$ sigmoid fonksiyonudur ve $[\cdot, \cdot]$ birleştirmedir). Bu kapı, $\mathbf{g}$ adıyla yeni ve eski bilginin ağırlığını belirleyecek.
     * Yeni durum adayı: $\tilde{h} = (1 - g) \odot h_{\text{prev}} + g \odot z_t$.

   * **Bozunum Katsayısı (γ) Hesaplaması**:
     * $\gamma = \sigma(W_{\gamma} \cdot z_t)$

   * **Geri Dönüş**: Hesaplanan `h_new` (yeni durum) ve $\gamma$ tensörlerini döndür.

**Ek Notlar:**
- Core Formula: `h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}` ile uyumlu olmalı
- Referans: CORE_FORMULA_SPEC.md ve IMPLEMENTATION_SPEC.md bölüm 3
- γ değerleri [1e-6, 1-1e-6] aralığında clamp edilmeli
```

---

### 3. HDS (Hierarchical Data Structure)

**Dosya:** `mm_rec/core/hds.py`

**Prompt:**

```
MM-Rec modelinin en karmaşık kısmı olan Hiyerarşik Veri Yapısı'nı (HDS) uygulamamız gerekiyor. Lütfen mm_rec/core/hds.py dosyasını oluştur. Bu yapı, bellek erişimini O(M) karmaşıklığı ile yapabilmek için kritiktir.

**Gereksinimler:**

1. **HDS Sınıfı:** Bir PyTorch `nn.Module` olarak `HierarchicalDataStructure` (HDS) adında bir sınıf oluştur.

   * **__init__**: `memory_state: MemoryState`, `num_levels: int = 3` ve `level_dims: list` (her seviyenin bellek yuvası sayısı) parametrelerini almalı.
   * **self.levels** adında bir liste veya sözlük tutmalı.

2. **Hiyerarşi İnşa Fonksiyonu:** `construct_hierarchy(self, state: MemoryState)` metodunu oluştur.

   * Bu fonksiyon, `MemoryState` içindeki `long_term` belleği kullanarak bir dizi havuzlama (pooling) işlemi ile hiyerarşiyi sembolik olarak inşa etmeli.
   * **Basitleştirme**: Gerçek havuzlama yerine, her seviye için `long_term` Key/Value tensörlerinin bir alt kümesini temsil eden tensörler yarat. Örneğin, Level 1'in belleği, Level 0'ın belleğinin (Long-term) bir özetidir.
   * **Amaç**: Her seviyedeki bellek yuvalarını (slots) PyTorch tensörleri olarak temsil etmek.

3. **O(M) Sorgulama Fonksiyonu:** `query_memory(self, query: torch.Tensor, level: int = -1)` metodunu oluştur.

   * Bu, `Multi-Memory Attention` bileşeninin kullanacağı hızlı erişim arayüzüdür.
   * **query**: Geçerli durum temsilini (h_t) alır.
   * **level**: Hangi hiyerarşi seviyesinin sorgulanacağını belirtir (varsayılan olarak en üst seviye, en küçük bellek).
   * **Uygulama**: Belirtilen seviyedeki bellek bankasının **Key** tensörünü almalı ve `query` ile bu Key'ler arasında dikkat skorlarını hesaplamak için bir hazırlık yapmalıdır.
   * **Geri Dönüş**: Sorgulanacak bellek Key ve Value tensörlerini (`k_level`, `v_level`) döndürmeli.

**Ek Notlar:**
- Hierarchy levels: Level 0 (token), Level 1 (block), Level 2 (global), Level 3 (long-term M)
- Access cost: O(M), not O(N) where N is sequence length
- Referans: IMPLEMENTATION_SPEC.md bölüm 2 ve TECHNICAL_REQUIREMENTS.md
```

---

## 🏗️ Faz 2: Blok Entegrasyonu

### 4. MM-Rec Block

**Dosya:** `mm_rec/blocks/mm_rec_block.py`

**Prompt:**

```
Tüm temel bileşenleri (Associative Scan, MDI, HDS) birleştiren ana katman yapısını oluşturmamız gerekiyor. Lütfen mm_rec/blocks/mm_rec_block.py dosyasını oluştur ve MMRecBlock sınıfını PyTorch nn.Module olarak tanımla.

**Gereksinimler:**

1. **MMRecBlock Sınıfı:** PyTorch `nn.Module` olarak `MMRecBlock` sınıfını oluştur.

   * **__init__**: `model_dim`, `inner_dim`, `num_heads`, `num_memories`, `mem_dim` gibi parametreleri almalı.
   * **Bağımlılıklar**: Daha önce oluşturulan `AssociativeScanExponential` (`from mm_rec.core.associative_scan_triton import associative_scan_exponential`), `MemoryDecayIntegration` ve `HierarchicalDataStructure` örneklerini sınıf üyeleri olarak başlatmalı.
   * **Lineer Katmanlar**: Gerekli transformasyonlar için `nn.Linear` katmanları (`W_q`, `W_k`, `W_v`, `W_z` - z_t için) tanımlanmalı.
   * **Normalization**: RMSNorm katmanları eklenmeli.

2. **İleri Geçiş (Forward) Metodu:** `forward(self, x: torch.Tensor, state: MemoryState) -> tuple[torch.Tensor, MemoryState]` metodunu oluştur. Bu metod, tek bir MM-Rec katmanının tüm 7 adımını sıralamalıdır.

   * **Adımlar:**

     a. **Query, Key, Value, Z Transformasyonları**: Giriş `x`'ten `q`, `k`, `v`, `z_t` tensörlerini türet.

     b. **Associative Scan**: `k` tensöründen γ katsayılarını türet ve **Associative Scan** fonksiyonunu kullanarak kümülatif çarpımı hesapla: `cumprod = associative_scan_exponential(gamma)`

     c. **MDI (Memory Decay/Integration)**: `z_t` ve `h_{prev}`'yi kullanarak yeni bellek durumunu (`h_{t}` ve yeni γ) hesapla: `h_new, gamma_new = mdi(z_t, h_prev)`

     d. **Core Formula**: `h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}` formülünü uygula.

     e. **Multi-Memory Attention**: HDS'i kullanarak bellek sorgusunu yap: `mem_context = multi_mem_attention(h_t, hds, state)`

     f. **Residual ve Çıkış**: Nihai çıktı tensörünü ve güncellenmiş `MemoryState` nesnesini döndür.

**Ek Notlar:**
- Core Formula: CORE_FORMULA_SPEC.md'deki formülü takip et
- Referans: CODE_STRUCTURE.md bölüm 5 ve IMPLEMENTATION_SPEC.md bölüm 4
- 7 adımın sırası kritik: ENGINEERING_OUTPUTS.md bölüm 5.2
```

---

### 5. Multi-Memory Attention

**Dosya:** `mm_rec/blocks/attention.py`

**Prompt:**

```
MM-Rec modelinin Multi-Memory Attention mekanizmasını uygulamamız gerekiyor. Bu, O(L²) yerine O(M) karmaşıklığı ile uzun vadeli belleği sorgulayacak. Lütfen mm_rec/blocks/attention.py dosyasını oluştur.

**Gereksinimler:**

1. **MultiMemoryAttention Sınıfı:** Bir PyTorch `nn.Module` olarak `MultiMemoryAttention` sınıfını oluştur.

   * **__init__**: `model_dim`, `num_heads` ve `head_dim` parametrelerini almalı.
   * **Multi-head attention**: Her head için ayrı query/key/value transformasyonları tanımla.

2. **Sorgulama Fonksiyonu:** `forward(self, query: torch.Tensor, hds: HierarchicalDataStructure, state: MemoryState) -> torch.Tensor` metodunu oluştur.

   * **O(M) Erişim**: `hds.query_memory(query, level=-1)` çağrısını kullanarak hiyerarşik bellekten (örneğin en üst seviye) **Key** ve **Value** tensörlerini al. (Kullanılacak bellek boyutunun $M \ll L$ olduğunu varsay.)

   * **Dikkat Skorları Hesaplaması**: Sorgu (`query`, yani $h_t$) ile bellek Key'ler (`k_mem`) arasındaki dikkat skorlarını hesapla: $\text{Skorlar} = Q \cdot K_{\text{mem}}^T / \sqrt{d_k}$.

   * **Yumuşak Maksimum (Softmax)**: Skorlara `softmax` uygula.

   * **Bağlamsal Vektör**: Skorları bellek Value'larla (`v_mem`) çarp: $\text{Context} = \text{Softmax}(\text{Skorlar}) \cdot V_{\text{mem}}$.

   * **Geri Dönüş**: Hesaplanan `Context` tensörünü döndür. Bu, MMRecBlock'ta çıktıya eklenecektir.

**Ek Notlar:**
- Memory complexity: O(M) not O(N²)
- Referans: IMPLEMENTATION_SPEC.md bölüm 2.3 ve CODE_STRUCTURE.md bölüm 4
- Multi-head attention pattern kullanılmalı
```

---

## 👑 Faz 3: Model ve Eğitim

### 6. Complete Model

**Dosya:** `mm_rec/model.py`

**Prompt:**

```
Artık tüm bileşenlerimiz hazır. Lütfen mm_rec/model.py dosyasını oluşturarak MM-Rec modelinin tam mimarisini oluştur.

**Gereksinimler:**

1. **MMRecModel Sınıfı:** Bir PyTorch `nn.Module` olarak `MMRecModel` sınıfını oluştur.

   * **__init__**: `vocab_size`, `model_dim`, `num_layers: int = 24`, `num_heads`, `num_memories`, `mem_dim`, `seq_len` gibi parametreleri almalı.
   * **Embedding Layer**: Giriş belirteçleri (token) için `nn.Embedding` katmanını tanımla.
   * **MemoryState Başlatma**: Başlangıç bellek durumu için bir `MemoryState` örneği oluştur.
   * **MM-Rec Blokları**: `nn.ModuleList` kullanarak 24 adet `MMRecBlock` katmanını başlat.
   * **Normalization**: Final RMSNorm katmanı.
   * **Output Head**: Dil modelleme görevi için bir çıkış lineer katmanı (`nn.Linear`) tanımla.

2. **İleri Geçiş (Forward) Metodu:** `forward(self, input_ids: torch.Tensor) -> torch.Tensor` metodunu oluştur.

   * **Gömme (Embedding)**: `input_ids`'ı gömme katmanından geçir.
   * **Döngü**: Tüm MMRecBlock katmanları üzerinde döngü kur. Her katmanda, hem girdi tensörünü (`x`) hem de bellek durumunu (`state`) güncelle.
     * `x, state = block(x, state)`
   * Her katman için ayrı MemoryState kullanılmalı (veya paylaşılan state)
   * **Çıktı Başı**: Nihai `x` tensörünü normalize et ve çıkış katmanından geçir.
   * **Geri Dönüş**: Modelin `logits` çıktılarını döndür.

**Ek Notlar:**
- Model configuration: 24 layers, 4096 hidden_dim, 32K+ seq_len (REQUIRED)
- Referans: CODE_STRUCTURE.md bölüm 6 ve ENGINEERING_OUTPUTS.md bölüm 5
- Memory state management: Her layer için state güncellenmeli
```

---

## 📋 Prompt Kullanım Kılavuzu

### Adım Adım Kullanım

1. **Sıralama**: Prompt'ları sırayla kullanın (Faz 1 → Faz 2 → Faz 3)
2. **Bağımlılıklar**: Her prompt, önceki prompt'ların tamamlanmasını gerektirir
3. **Test**: Her bileşen oluşturulduktan sonra test edin
4. **Dokümantasyon**: Her bileşen için docstring ekleyin

### LLM'e Verilecek Format

Her prompt'u LLM'e şu şekilde verin:

```
[PROMPT İÇERİĞİ]

Lütfen bu prompt'u takip ederek [DOSYA_ADI] dosyasını oluştur.
Proje yapısına uygun olarak kod yaz.
Gerekli import'ları ekle.
Docstring'leri ekle.
Test edilebilir kod yaz.
```

### Doğrulama Checklist

Her bileşen için kontrol edin:

- [ ] Dosya doğru konumda oluşturuldu mu?
- [ ] Gerekli import'lar eklendi mi?
- [ ] Class/function signature'lar doğru mu?
- [ ] Docstring'ler eklendi mi?
- [ ] Test edilebilir mi?
- [ ] Referans dokümantasyonla uyumlu mu?

---

## 🔗 İlgili Dokümanlar

- **ENGINEERING_OUTPUTS.md**: Tüm çıktıların checklist'i
- **CODE_STRUCTURE.md**: API tasarımı ve kod örnekleri
- **IMPLEMENTATION_SPEC.md**: Algoritma detayları
- **CORE_FORMULA_SPEC.md**: Core formula spesifikasyonu
- **PROJECT_STATUS.md**: Mevcut durum ve ilerleme

---

**Son Güncelleme**: 2025-12-08
**Durum**: Prompt'lar hazır, implementasyon bekleniyor

