# 📚 Pre-Training'de İlk Öğrenilmesi Gereken Konular

**Tarih**: 2025-01-27  
**Hedef**: MM-Rec modelinin pre-training'de öğrenme sırası

---

## 🎯 Öğrenme Hiyerarşisi (Öncelik Sırası)

### Seviye 1: Temel Token ve Karakter Tanıma (İlk Hafta)

**Ne Öğrenmeli:**
1. **Token/Character Tanıma**
   - Alfabetik karakterler (a-z, A-Z)
   - Rakamlar (0-9)
   - Noktalama işaretleri (., !, ?, vb.)
   - Özel karakterler (space, newline, tab)

2. **Temel Kelime Tanıma**
   - Yaygın kelimeler (the, and, is, are, vb.)
   - Kısa kelimeler (1-4 karakter)
   - Yaygın kelime kombinasyonları

**Veri Özellikleri:**
- Basit metinler (çocuk kitapları, basit hikayeler)
- Tekrarlayan pattern'ler
- Kısa cümleler

**Beklenen Loss:**
- Başlangıç: 8-10
- 1 hafta sonra: 4-6

---

### Seviye 2: Dilbilgisi ve Syntax (İkinci Hafta)

**Ne Öğrenmeli:**
1. **Temel Dilbilgisi Kuralları**
   - Cümle yapısı (Subject-Verb-Object)
   - İsim-fiil uyumu
   - Zaman kullanımı (past, present, future)
   - Çoğul/tekil uyumu

2. **Syntax Pattern'leri**
   - Cümle başlangıçları (The, A, This, vb.)
   - Cümle bitişleri (., !, ?)
   - Bağlaçlar (and, but, or, because, vb.)
   - Edatlar (in, on, at, with, vb.)

3. **Kelime Sırası**
   - İngilizce kelime sırası kuralları
   - Sıfat-isim sırası
   - Fiil-zarf sırası

**Veri Özellikleri:**
- Düzenli cümle yapıları
- Basit hikayeler
- Eğitim metinleri

**Beklenen Loss:**
- Başlangıç: 4-6
- 2 hafta sonra: 2-3

---

### Seviye 3: Semantik ve Anlam (Üçüncü-Dördüncü Hafta)

**Ne Öğrenmeli:**
1. **Kelime Anlamları**
   - Eş anlamlılar (synonyms)
   - Zıt anlamlılar (antonyms)
   - Kelime ilişkileri (hyponymy, meronymy)

2. **Bağlam Anlama**
   - Kelimelerin bağlama göre anlamı
   - Çok anlamlılık (polysemy)
   - İfade anlamları (idioms, phrases)

3. **Temel İlişkiler**
   - Neden-sonuç ilişkileri
   - Zaman ilişkileri (before, after, during)
   - Mekan ilişkileri (in, on, under, vb.)

**Veri Özellikleri:**
- Çeşitli konular (bilim, tarih, edebiyat)
- Farklı yazım stilleri
- Zengin kelime dağarcığı

**Beklenen Loss:**
- Başlangıç: 2-3
- 1 ay sonra: 1.5-2.0

---

### Seviye 4: Dünya Bilgisi (Birinci-İkinci Ay)

**Ne Öğrenmeli:**
1. **Temel Gerçekler**
   - Coğrafya (ülkeler, şehirler, nehirler)
   - Tarih (önemli olaylar, tarihler)
   - Bilim (temel kavramlar, elementler)
   - Kültür (gelenekler, bayramlar)

2. **İlişkisel Bilgi**
   - "Paris is the capital of France"
   - "Water boils at 100°C"
   - "Shakespeare wrote Hamlet"

3. **Kategoriler**
   - Hayvanlar, bitkiler, nesneler
   - Meslekler, roller
   - Soyut kavramlar

**Veri Özellikleri:**
- Wikipedia
- Ansiklopedi içerikleri
- Eğitim kitapları
- Bilimsel metinler

**Beklenen Loss:**
- Başlangıç: 1.5-2.0
- 2 ay sonra: 1.0-1.5

---

### Seviye 5: Mantık ve Akıl Yürütme (İkinci-Üçüncü Ay)

**Ne Öğrenmeli:**
1. **Mantıksal İlişkiler**
   - Eğer-o zaman (if-then)
   - Neden-sonuç (cause-effect)
   - Karşılaştırma (comparison)
   - Çıkarım (inference)

2. **Problem Çözme**
   - Adım adım düşünme
   - Mantıksal sıralama
   - Analiz ve sentez

3. **Soyut Düşünme**
   - Metaforlar
   - Analojiler
   - Genellemeler

**Veri Özellikleri:**
- Felsefe metinleri
- Mantık problemleri
- Bilimsel makaleler
- Edebi eserler

**Beklenen Loss:**
- Başlangıç: 1.0-1.5
- 3 ay sonra: 0.8-1.2

---

## 📊 Öğrenme İlerlemesi (Loss Bazlı)

### Loss 8-10: Temel Token Tanıma
- **Öğrenilen**: Karakterler, basit kelimeler
- **Veri**: Basit metinler, tekrarlayan pattern'ler
- **Süre**: 1-2 hafta

### Loss 4-6: Dilbilgisi ve Syntax
- **Öğrenilen**: Cümle yapısı, temel kurallar
- **Veri**: Düzenli cümleler, basit hikayeler
- **Süre**: 2-3 hafta

### Loss 2-3: Semantik ve Anlam
- **Öğrenilen**: Kelime anlamları, bağlam
- **Veri**: Çeşitli konular, zengin kelime dağarcığı
- **Süre**: 1 ay

### Loss 1.5-2.0: Dünya Bilgisi
- **Öğrenilen**: Gerçekler, ilişkiler, kategoriler
- **Veri**: Wikipedia, ansiklopediler, eğitim kitapları
- **Süre**: 1-2 ay

### Loss 1.0-1.5: Mantık ve Akıl Yürütme
- **Öğrenilen**: Mantıksal ilişkiler, problem çözme
- **Veri**: Felsefe, bilim, edebiyat
- **Süre**: 2-3 ay

### Loss <1.0: İleri Seviye
- **Öğrenilen**: Karmaşık akıl yürütme, yaratıcılık
- **Veri**: Çok çeşitli, yüksek kaliteli içerik
- **Süre**: Sürekli

---

## 🎯 MM-Rec İçin Özel Öneriler

### 1. Long Context Öğrenme (32K+)

**Öncelik**: Yüksek (MM-Rec'in temel özelliği)

**Ne Öğrenmeli:**
- Uzun metinlerde tutarlılık
- Uzun mesafeli bağımlılıklar
- Paragraf/bolüm arası ilişkiler

**Veri:**
- Uzun kitaplar (bölümler halinde)
- Akademik makaleler
- Uzun hikayeler

### 2. Bellek Yönetimi

**Öncelik**: Yüksek (MM-Rec'in dual memory sistemi)

**Ne Öğrenmeli:**
- Önemli bilgileri hatırlama
- İlgisiz bilgileri unutma
- Uzun vadeli bağlam koruma

**Veri:**
- Tekrarlayan referanslar içeren metinler
- Uzun hikayeler (karakter isimleri, olaylar)
- Bilimsel metinler (kavramlar, tanımlar)

### 3. Hiyerarşik Yapı Anlama

**Öncelik**: Orta (HDS sistemi için)

**Ne Öğrenmeli:**
- Paragraf yapısı
- Bölüm/alt başlık ilişkileri
- Hiyerarşik bilgi organizasyonu

**Veri:**
- Yapılandırılmış metinler (Wikipedia, ansiklopediler)
- Akademik makaleler (bölümler, alt bölümler)
- Teknik dokümantasyon

---

## 📋 Veri Kaynağı Öncelikleri

### İlk Hafta: Basit ve Tekrarlayan
1. **Tiny Shakespeare** (gerçek text, küçük)
2. **Basit hikayeler** (çocuk kitapları)
3. **Eğitim metinleri** (basit cümleler)

### İkinci-Üçüncü Hafta: Dilbilgisi Odaklı
1. **WikiText-103** (Wikipedia, düzenli yapı)
2. **Basit kitaplar** (düzenli cümle yapısı)
3. **Eğitim içerikleri** (grammar-focused)

### Birinci Ay: Çeşitlilik
1. **OpenWebText** (çeşitli konular)
2. **Wikipedia** (geniş kapsam)
3. **BookCorpus** (farklı yazım stilleri)

### İkinci Ay: Bilgi Odaklı
1. **C4** (Colossal Clean Crawled Corpus)
2. **Wikipedia** (detaylı bilgi)
3. **Akademik metinler** (derin bilgi)

### Üçüncü Ay: Mantık ve Akıl Yürütme
1. **Felsefe metinleri**
2. **Bilimsel makaleler**
3. **Edebi eserler** (karmaşık yapı)

---

## 🎓 Öğrenme Stratejisi

### Progressive Curriculum Learning

**Aşama 1: Basit → Karmaşık**
- Önce basit metinler
- Sonra karmaşık metinler

**Aşama 2: Kısa → Uzun**
- Önce kısa sequence'ler (512-1024)
- Sonra uzun sequence'ler (32K+)

**Aşama 3: Tek Konu → Çok Konu**
- Önce tek konu (tutarlılık)
- Sonra çok konu (genelleme)

### Örnek Eğitim Planı

**Hafta 1-2: Temel**
- Dataset: Tiny Shakespeare + basit metinler
- Sequence length: 512
- Loss hedefi: 8 → 4

**Hafta 3-4: Dilbilgisi**
- Dataset: WikiText-103
- Sequence length: 1024
- Loss hedefi: 4 → 2

**Ay 2: Semantik**
- Dataset: OpenWebText subset
- Sequence length: 2048
- Loss hedefi: 2 → 1.5

**Ay 3: Bilgi**
- Dataset: Wikipedia + C4 subset
- Sequence length: 4096
- Loss hedefi: 1.5 → 1.0

**Ay 4+: İleri**
- Dataset: Full C4 + çeşitli kaynaklar
- Sequence length: 8192-32768
- Loss hedefi: 1.0 → 0.8

---

## 💡 MM-Rec İçin Özel Notlar

### 1. Long Context Öğrenme
- MM-Rec 32K+ sequence destekliyor
- Uzun metinlerde tutarlılık öğrenmeli
- Paragraf/bolüm arası bağlantılar önemli

### 2. Bellek Yönetimi
- Dual memory sistemi (h_t + M)
- Önemli bilgileri uzun vadede hatırlamalı
- İlgisiz bilgileri unutmalı

### 3. Hiyerarşik Yapı
- HDS sistemi hiyerarşik bilgi kullanıyor
- Paragraf/bölüm yapısını öğrenmeli
- Seviye bazlı bilgi organizasyonu

---

## ✅ Sonuç

### İlk Öğrenilmesi Gerekenler (Öncelik Sırası)

1. **Temel Token/Karakter Tanıma** (Loss 8-10)
   - Alfabetik karakterler
   - Basit kelimeler
   - Noktalama

2. **Dilbilgisi ve Syntax** (Loss 4-6)
   - Cümle yapısı
   - Temel kurallar
   - Kelime sırası

3. **Semantik ve Anlam** (Loss 2-3)
   - Kelime anlamları
   - Bağlam anlama
   - Temel ilişkiler

4. **Dünya Bilgisi** (Loss 1.5-2.0)
   - Gerçekler
   - İlişkiler
   - Kategoriler

5. **Mantık ve Akıl Yürütme** (Loss 1.0-1.5)
   - Mantıksal ilişkiler
   - Problem çözme
   - Soyut düşünme

### MM-Rec İçin Özel
- ✅ Long context öğrenme (32K+)
- ✅ Bellek yönetimi (dual memory)
- ✅ Hiyerarşik yapı anlama (HDS)

**Durum**: Öğrenme hiyerarşisi belirlendi, progressive curriculum learning önerildi.

---

**Tarih**: 2025-01-27  
**Durum**: Pre-training öğrenme sırası belirlendi
