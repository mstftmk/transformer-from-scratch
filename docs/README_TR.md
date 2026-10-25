> **Not:** Bu belge projenin ilk sürümünü anlatır. Güncel sürümde varsayılan görev `pointer`'dır ve varsayılan eğitim 20.000 örnek / 20 epoch / lr 1e-3'tür. Orijinal `mod_sum` görevi hâlâ seçilebilir (bkz. [README.md](../README.md)).

# Transformer From Scratch (PyTorch) — Eğitici Proje

Bu README, projedeki tüm açıklamaları **tek yerde** saklar. Her aşamada **kod önce**, **açıklama sonra** gelir. Ayrıca kritik noktalar (gradient flow, shape, numerical stability, memory footprint) özel olarak vurgulanır.

---

## Aşama 1 — Transformer Neden Ortaya Çıktı?

```text
# Conceptual pseudocode (intuition for attention)
scores = Q @ K.T / sqrt(d_k)
weights = softmax(scores)
output = weights @ V
```

Açıklama:
- CNN’ler lokal alana güçlüdür ama uzun menzilli ilişkiler için çok derinleşmeleri gerekir.
- RNN’ler sıralı çalışır; paralelleşme zordur ve uzun bağımlılıklarda gradyan sorunları yaşar.
- Attention sezgisi: Her token, diğer tüm token’lara bakar (benzerlik), onları ağırlıklandırır ve değerlerin ağırlıklı toplamını alır. Bu, **uzun menzilli ilişkiyi tek adımda** ve **paralel** yakalamayı sağlar.

**Kritik noktalar:**
- **Numerical stability:** Softmax öncesi skoru ölçeklemek ve max çıkarmak taşmayı azaltır.
- **Memory footprint:** `QK^T` matrisi `(B, H, N, N)` ölçeğinde büyür; uzun dizilerde bellek maliyeti kritiktir.

---

## Aşama 2 — Temel Yapı Taşları

```python
# Embedding + Positional Encoding + Self-Attention + FFN
# Bu yapı taşları model.py içinde tek tek, elle implement edilmiştir.
```

Açıklama:
- **Embedding:** Token’ları vektör uzayına taşır.
- **Positional Encoding:** Sıra bilgisini ekler (sinusoidal).
- **Self-Attention:** Her token’ın tüm diğer token’ları “okumasını” sağlar.
- **Multi-Head Attention:** Farklı ilişki tipleri paralel öğrenilir.
- **FFN:** Her token üzerinde konum-bağımsız, doğrusal olmayan dönüşüm.

**Kritik noktalar:**
- **Gradient flow:** Residual bağlantılar ve pre-norm mimari gradyanı korur.
- **Shape kontrolü:** Her adımda `(B, N, C)` şeklini takip etmek hata riskini azaltır.
- **Numerical stability:** Softmax öncesi max çıkarımı, AMP kullanımında daha güvenlidir.

---

## Aşama 3 — Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout, causal):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, causal)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Residual + Attention
        x = x + self.ffn(self.ln2(x))   # Residual + FFN
        return x
```

Açıklama:
- **Residual bağlantılar**, öğrenmeyi kolaylaştırır ve derin ağlarda gradyan sönmesini azaltır.
- **LayerNorm**, aktivasyon ölçeğini stabilize eder.
- **Pre-norm** (LayerNorm önce), derinlik arttıkça eğitim stabilitesini artırır.

**Kritik noktalar:**
- **Gradient flow:** Residual yol “kimlik” sinyali sağlar.
- **Numerical stability:** LayerNorm ölçeklenmiş aktivasyon sağlar.

---

## Aşama 4 — Backbone Tasarımı

```python
class TransformerBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = SinusoidalPositionalEncoding(cfg.seq_len, cfg.embed_dim, cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.ffn_hidden_dim, cfg.dropout, cfg.causal)
             for _ in range(cfg.num_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.embed_dim)

    def forward(self, input_ids):
        x = self.token_embed(input_ids)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)
```

Açıklama:
- Blokları üst üste koymak **kapasiteyi** artırır.
- Derinlik seçimi veri büyüklüğü, görev karmaşıklığı ve hesap bütçesiyle dengelenmelidir.

**Kritik noktalar:**
- **Memory footprint:** Katman sayısı arttıkça aktivasyon belleği büyür.
- **Gradient flow:** Daha derin ağlarda pre-norm tercih edilmesi önemlidir.

---

## Aşama 5 — Training Pipeline

```python
# dataset.py -> synthetic dataset
# loss.py -> CrossEntropyLoss
# main.py  -> optimizer AdamW + cosine scheduler
```

Açıklama:
- Dataset: Sentetik ama yapılı bir görev seçildi (son iki token toplamı mod V).
- Loss: Next-token prediction için CrossEntropy.
- Optimizer: AdamW (weight decay’i düzgün uygular).
- Scheduler: Cosine annealing (LR’yi yumuşakça azaltır).

**Kritik noktalar:**
- **Reproducibility:** Seed sabitleme, deney tekrarlarını güvenilir kılar.
- **Numerical stability:** AdamW + uygun LR, eğitim patlamalarını azaltır.

---

## Aşama 6 — Kodlama Kuralları

```python
# Shape notları ve okunabilir forward
# x: (B, N, C)
```

Açıklama:
- Her class tek sorumlulukta tutuldu.
- Forward pass okunabilir ve lineer akışla yazıldı.
- Shape yorumları `(B, N, C)` biçiminde eklendi.

**Kritik noktalar:**
- **Debuggability:** Shape takip etmek, attention hatalarını hızla yakalar.
- **Maintainability:** Modüler yapı genişletmeyi kolaylaştırır.

---

## Aşama 7 — Training Loop

```python
# trainer.py içinde train_epoch ve evaluate
# Gradient clipping uygulanır.
```

Açıklama:
- Forward → backward → gradient clipping → optimizer step sırası net.
- Logging ve checkpointing eğitim izlenebilirliğini artırır.

**Kritik noktalar:**
- **Gradient clipping:** Büyük gradyanlar patlamayı engeller.
- **Checkpointing:** Eğitim kesilse bile devam edilebilir.
- **AMP:** Opsiyonel karışık hassasiyet eğitimi hızlandırabilir.

---

## Aşama 8 — ViT’e Dönüşüm ve Optimizasyon

```python
# Patch embedding + CLS token + 2D positional embedding
# Attention/LayerNorm/FFN fused kernels ile hızlandırılır.
```

Açıklama:
- ViT için giriş görüntüsü patch’lere ayrılır, her patch token’a çevrilir.
- `CLS` token eklenir ve sınıflandırma onun üzerinden yapılır.

**Kritik noktalar:**
- **CUDA kernel kritik:** QK^T, softmax, LayerNorm, GELU.
- **Memory footprint:** FlashAttention gibi yöntemlerle düşürülebilir.
- **Speed:** Fused operations büyük hız kazancı sağlar.

---

## Dosya Yapısı

- `dataset.py`  — Küçük ama gerçekçi sentetik veri
- `model.py`    — Transformer backbone ve attention
- `loss.py`     — CrossEntropyLoss wrapper
- `trainer.py`  — Training loop + logging + checkpoint
- `main.py`     — Pipeline giriş noktası

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
python main.py
```

Açıklama:
- `venv` izolasyon sağlar; bağımlılıklar projeye özel kalır.
- `pip install torch` CPU için yeterlidir. GPU kullanacaksan uygun CUDA paketini seçmelisin.
- `python main.py` tüm pipeline’ı uçtan uca çalıştırır.

---

## Çalıştırma

```bash
python main.py
```

Bu komut `dataset.py`, `model.py`, `loss.py`, `trainer.py` ve `main.py` ile tanımlanan tam eğitim akışını başlatır.

## Doğrulama

```bash
python -m py_compile dataset.py model.py loss.py trainer.py main.py
```

---

## Örnek Çıktı

```text
2026-02-08 19:12:03 - INFO - epoch=1 train_loss=3.92 val_loss=3.85
2026-02-08 19:12:11 - INFO - epoch=2 train_loss=3.41 val_loss=3.30
2026-02-08 19:12:19 - INFO - epoch=3 train_loss=2.88 val_loss=2.75
```

Açıklama:
- Bu değerler **temsili**dir. Donanım, seed ve hiperparametrelere göre değişir.
- Loss’un düşmesi, modelin hedef kuralı öğrenmeye başladığını gösterir.

---

## Metrik Hesaplama (Örnek)

```python
# logits: (B, V), targets: (B,)
preds = logits.argmax(dim=-1)           # (B,)
accuracy = (preds == targets).float().mean().item()

# Cross-entropy loss -> perplexity dönüşümü (opsiyonel)
perplexity = float(torch.exp(loss))
```

Açıklama:
- **Accuracy**, doğru sınıf tahmin oranını verir ve yorumlaması kolaydır.
- **Perplexity**, özellikle dil modellemede kullanılır; loss azaldıkça perplexity düşer.
- Bu metrikleri `trainer.py` içinde `evaluate()` veya `train_epoch()` sonuna ekleyebilirsin.

---

## Backbone Tasarım Rehberi

```text
Token Embedding  -> Tokenları vektör uzayına taşır.
Positional Enc.  -> Sıra bilgisini ekler.
Self-Attention   -> Global bağımlılıkları yakalar.
FFN              -> Her token için güçlü doğrusal olmayan dönüşüm.
LayerNorm        -> Aktivasyon ölçeklerini dengeler.
Residual         -> Gradyan akışını korur.
```

Açıklama:
1. **Katman ne zaman eklenir?**  
   Model **underfit** ise (train loss yüksek, val ile benzer), derinlik (`num_layers`) veya genişlik (`embed_dim`) artırılır. Veri karmaşıklığı yüksekse daha derin stack gerekir.  
   Etkisi: Kapasite ve temsil gücü artar, ama eğitim süresi ve bellek tüketimi yükselir.
2. **Katman ne zaman çıkarılır?**  
   **Overfit** varsa (train iyi, val kötü), derinliği azalt, dropout artır veya daha küçük FFN kullan. Bellek/latency kısıtlıysa katman sayısını düşür.  
   Etkisi: Genelleme iyileşebilir, hız artar, kapasite azalır.
3. **Hangi katman ne yapar?**  
   **Attention** tokenlar arası ilişkiyi yakalar ve uzun menzilli bağımlılık sağlar.  
   **FFN** temsil gücünü artırır, parametrelerin büyük kısmı buradadır.  
   **LayerNorm** aktivasyonları dengeler, eğitimi stabil kılar.  
   **Residual** gradyan akışını korur ve derin ağları eğitilebilir yapar.
4. **Num_heads nasıl seçilir?**  
   `embed_dim % num_heads == 0` olmalı. Head sayısı artarsa farklı ilişki tipleri öğrenilir ama hesap maliyeti ve bellek tüketimi artar.  
   Pratikte `embed_dim / num_heads` 32–64 aralığında başlamak güvenlidir.
5. **FFN boyutu nasıl seçilir?**  
   Genelde `ffn_hidden_dim ≈ 4 * embed_dim` güvenli başlangıçtır.  
   Daha küçük FFN bellek kazandırır ama kapasiteyi düşürür; daha büyük FFN öğrenmeyi güçlendirir ama overfit ve maliyet riski artar.
6. **Sequence length etkisi:**  
   Attention maliyeti `O(N^2)` olduğu için uzun dizilerde hız ve bellek kritik olur.  
   Etkisi: `N` iki katına çıkarsa attention maliyeti yaklaşık dört kat artar.
7. **Pre-norm vs Post-norm ne zaman?**  
   Pre-norm (bizdeki gibi) derin modellerde daha stabil ve gradyan dostudur.  
   Post-norm, sığ modellerde bazen daha iyi doğruluk verebilir ama derinlik arttıkça eğitim zorlaşabilir.
8. **Dropout ne zaman artırılır?**  
   Overfit varsa dropout artırılır. Küçük veri ve büyük model kombinasyonunda özellikle etkilidir.  
   Etkisi: Genelleme artar, ama çok yüksek dropout öğrenmeyi yavaşlatır.
9. **Residual ölçekleme veya Stochastic Depth ne zaman?**  
   Çok derin ağlarda residual’ı ölçeklemek veya katmanları rastgele atlamak (stochastic depth) stabiliteyi artırabilir.  
   Etkisi: Genelleme artar, eğitim daha kararlı hale gelir.

### Örnek Backbone Konfigürasyonları

```python
# Küçük / hızlı (CPU veya küçük GPU)
ModelConfig(embed_dim=64, num_heads=4, num_layers=2, ffn_hidden_dim=256)

# Orta seviye (daha iyi kalite)
ModelConfig(embed_dim=128, num_heads=4, num_layers=4, ffn_hidden_dim=512)

# Daha güçlü (daha fazla kapasite, daha yüksek maliyet)
ModelConfig(embed_dim=256, num_heads=8, num_layers=8, ffn_hidden_dim=1024)
```

Açıklama:
1. Küçük konfigürasyon hızlıdır ama karmaşık örüntüleri sınırlı öğrenir.  
2. Orta konfigürasyon pratik dengedir.  
3. Büyük konfigürasyon yüksek kapasite sağlar ama bellek ve süre maliyeti yüksektir.

### Kullanım Senaryoları ve Etkileri

1. **Küçük veri + hızlı prototip**  
   Daha az katman ve daha küçük `embed_dim` kullan. Etkisi: hızlı eğitim, düşük maliyet, sınırlı performans.
2. **Orta veri + dengeli kalite**  
   4–6 katman ve orta `embed_dim` seç. Etkisi: iyi doğruluk ve makul maliyet.
3. **Büyük veri + yüksek kalite**  
   Derinlik ve genişliği artır. Etkisi: yüksek doğruluk, yüksek maliyet ve daha zor eğitim.
4. **Uzun sequence**  
   `seq_len` büyüdükçe attention maliyeti patlar. Etkisi: bellek ve hız darboğazı.  
   Gerekirse `seq_len` azalt, chunking yap veya FlashAttention gibi teknikler düşün.

### Backbone İnce Noktalar (Sahada İşe Yarayan Detaylar)

1. **Causal mask ne zaman gerekir?**  
   Next-token prediction için zorunludur. Sınıflandırma gibi tüm diziye bakan görevlerde gerekmez.  
   Etkisi: Gelecek bilgisi sızmasını engeller ama ek maskeleme maliyeti getirir.
2. **CLS token mı, pooling mi?**  
   Sınıflandırmada CLS token kullanılırsa model “özet” vektörü öğrenir.  
   Ortalama pooling, basit ve bazen daha stabil sonuç verir ama görevden göreve değişir.
3. **Embedding ile head ağırlıklarını bağlamak (weight tying)**  
   Dil modelleme gibi görevlerde embedding ile çıktı katmanını paylaşmak parametreyi azaltır ve genellikle performansı artırır.  
   Etkisi: Model boyutu düşer, genelleme iyileşebilir.
4. **GELU vs ReLU**  
   GELU daha yumuşak geçiş sağlar ve Transformer’larda standarttır.  
   ReLU daha hızlıdır ama bazen performans düşer.
5. **Positional Encoding seçimi**  
   Sinusoidal, farklı uzunluklara genelleme sağlar.  
   Öğrenilebilir embedding, sabit uzunlukta daha iyi olabilir ama extrapolation zayıftır.
6. **LayerNorm epsilon ve stabilite**  
   Epsilon çok küçükse sayısal sorun, çok büyükse aşırı yumuşatma olur.  
   Varsayılan `1e-5` genelde güvenlidir.
7. **Attention head dimension**  
   Head boyutu çok küçükse kapasite düşer; çok büyükse head sayısı azalır.  
   Uygulamada `head_dim` 32–64 dengelidir.

---

## Backbone Yazarken Olmazsa Olmazlar

1. **Shape disiplinini koru**  
   Her modülde giriş/çıkış şekli net olmalı: `(B, N, C)` standardını bozma. Yanlış reshape/permute sessiz hatalara yol açar.
2. **Scaled dot-product şart**  
   `QK^T / sqrt(dk)` ölçeklemesi yapılmazsa softmax aşırı keskinleşir ve gradyanlar bozulur.
3. **Causal mask görevle uyumlu olmalı**  
   Next-token prediction için zorunlu, sınıflandırmada gereksizdir. Yanlış kullanım doğrudan bilgi sızıntısı yaratır.
4. **Pre-norm derinlikte hayat kurtarır**  
   Derin stacklerde (L>=6) pre-norm genellikle daha stabil öğrenir. Post-norm kullanacaksan küçük derinlikte dene.
5. **Dropout yerleşimi önemlidir**  
   Attention ağırlıklarına ve FFN çıkışına dropout koymak genelleme sağlar. Çok fazla dropout öğrenmeyi yavaşlatır.
6. **LayerNorm epsilon ayarı**  
   Çok küçük epsilon sayısal sorun yaratabilir; `1e-5` iyi bir başlangıçtır.
7. **Init + dtype uyumu**  
   Varsayılan PyTorch init çoğu durumda yeterlidir. AMP kullanıyorsan softmax/mask değerlerini `finfo(min)` ile güvenceye al.
8. **Attention maliyetini unutma**  
   `N` büyüdükçe bellek ve hız kuadratik artar. Uzun dizilerde FlashAttention veya chunking planla.
9. **Residual bağlantılar eksik olamaz**  
   Residual yoksa derin ağ eğitimi dramatik şekilde zorlaşır.
10. **FFN kapasitesi toplam parametrenin çoğunu taşır**  
    `ffn_hidden_dim` küçülürse kapasite hızla düşer; büyürse overfit ve maliyet artar.
11. **Layer sayısını tek seferde büyütme**  
    Önce küçük bir modelle öğrenip öğrenmediğini doğrula, sonra derinliği artır.
12. **Sanity check yap**  
    Küçük veri üzerinde hızla overfit edemiyorsa mimari veya loss akışında hata vardır.

---

## Sorun → Çözüm (Gerçekçi Örnekler)

| Sorun (Belirti) | Olası Sebep | Çözüm | Etki |
|---|---|---|---|
| Train loss düşmüyor, val de düşmüyor | LR çok düşük veya model çok küçük | LR artır veya `embed_dim/num_layers` artır | Öğrenme başlar, kapasite artar |
| Train loss düşüyor, val yükseliyor | Overfit | Dropout artır, layer sayısını azalt, FFN küçült | Genelleme iyileşir |
| Loss dalgalı ve kararsız | LR çok yüksek veya gradient patlıyor | LR düşür, gradient clipping ekle | Eğitim stabil olur |
| NaN/Inf loss | Softmax taşması, mask hatası | `scale` ekle, `finfo(min)` kullan, mask doğru mu kontrol et | Sayısal stabilite |
| Çok yavaş eğitim | `seq_len` büyük | `seq_len` azalt, FlashAttention kullan | Hız ve bellek kazanımı |
| Accuracy düşük ama loss düşüyor | Head yanlış veya pooling hatası | CLS veya pooling seçimini gözden geçir | Doğruluk artar |
| Model hiç overfit olamıyor | Pipeline hatası | Küçük bir batch’te aşırı overfit test et | Hata erken yakalanır |

Notlar:
1. Bu tablo **teşhis için** hızlı bir kontrol listesi sağlar.  
2. Çözüm adımlarını tek tek uygula; aynı anda çok değişiklik yapma.

---

## Örnek Backbone Tasarımı (Açıklamalı)

```python
# Orta seviye, dengeli bir backbone
cfg = ModelConfig(
    vocab_size=50,
    seq_len=16,
    embed_dim=128,     # Daha geniş temsil
    num_heads=4,       # head_dim=32
    num_layers=4,      # Orta derinlik
    ffn_hidden_dim=512,# FFN kapasitesi (4x)
    dropout=0.1,
    causal=True,
)
```

Bu backbone’daki katmanlar ve etkileri:
1. **Token Embedding** → Token’ları `C` boyutlu vektöre taşır.  
   Etki: Temsil gücü artar; `C` büyüdükçe model kapasitesi artar.
2. **Positional Encoding** → Sıra bilgisini ekler.  
   Etki: Attention “hangi sıra” bilgisini öğrenebilir.
3. **Self-Attention** → Tokenlar arası global ilişkiyi modeller.  
   Etki: Uzun menzilli bağımlılıkları yakalar.
4. **FFN** → Token başına güçlü dönüşüm yapar.  
   Etki: Parametrelerin çoğu burada; kapasiteyi belirler.
5. **LayerNorm** → Aktivasyonları dengeler.  
   Etki: Eğitim stabilitesini artırır.
6. **Residual** → Gradyan akışını korur.  
   Etki: Derin ağları eğitilebilir kılar.

---

## Layer Sayısını Neye Göre Ayarlamalı?

1. **Veri miktarı**  
   Büyük veri → daha derin model. Küçük veri → daha sığ model.
2. **Görev karmaşıklığı**  
   Basit kural öğrenimi → az layer. Karmaşık bağlam → daha fazla layer.
3. **Overfit/Underfit sinyali**  
   Underfit → layer artır. Overfit → layer azalt veya dropout artır.
4. **Donanım bütçesi**  
   Derinlik arttıkça bellek ve süre büyür. Kısıtlı GPU’da sığ model seç.
5. **Deneme stratejisi**  
   Önce küçük bir `L` ile başla, öğreniyorsa kademeli artır.

---

## Parametre Sayısı ve FLOPs Tahmini

```text
Semboller:
C = embed_dim
F = ffn_hidden_dim
V = vocab_size
L = num_layers
B = batch_size
N = seq_len
```

### Parametre Sayısı (Yaklaşık)

```text
Embedding params ≈ V * C
Block params     ≈ 4*C^2 + 2*C*F + 9*C + F
Backbone params  ≈ L * (block params)
Final LayerNorm  ≈ 2*C
Head params      ≈ C*V + V
Total params     ≈ V*C + L*(4*C^2 + 2*C*F + 9*C + F) + 2*C + (C*V + V)
```

Açıklama:
1. Bu formül **bias** terimlerini ve LayerNorm ağırlık/bias’larını içerir.  
2. Sinusoidal positional encoding **parametre eklemez**.  
3. Eğer **learned positional embedding** kullanırsan `N*C` parametre eklenir.  
4. **Weight tying** (embedding ve head paylaşımı) kullanırsan `C*V` kadar parametre düşer.

### FLOPs / MACs (Yaklaşık)

```text
MACs per layer ≈ B * [ N*(4*C^2 + 2*C*F) + 2*N*N*C ]
Total MACs     ≈ L * MACs per layer
FLOPs          ≈ 2 * MACs  (mul+add ayrı sayılırsa)
```

Açıklama:
1. QKV projeksiyonları ve FFN, **N*C^2** ve **N*C*F** terimlerini belirler.  
2. Attention skorları ve ağırlıklı toplam **N^2*C** terimiyle büyür.  
3. `N` büyüdükçe FLOPs ve bellek maliyeti **kuadratik** artar.  
4. Softmax ve LayerNorm ek maliyet getirir ama genelde ikincil kalır.

### Örnek Hesap (Varsayılan Konfig)

```text
V=50, C=64, F=256, L=4
Toplam parametre ≈ 206,514

B=64, N=16
Toplam MACs (forward) ≈ 0.210 GMACs
Toplam FLOPs (forward) ≈ 0.420 GFLOPs
```

Notlar:
1. Training sırasında forward+backward yaklaşık 2–3x daha pahalıdır.  
2. Donanım ve framework optimizasyonları gerçek ölçümü değiştirir.

---

## Ek Kritik Notlar (Özet)

- **Gradient flow:** Residual + pre-norm ile korunur.
- **Shape değişimleri:** `(B, N, C)` standardı tüm modüllerde korunur.
- **Numerical stability:** Softmax öncesi max çıkarımı, `scale = 1/sqrt(dk)`.
- **Memory footprint:** Attention matrisi `(B, H, N, N)` en pahalı kısımdır.
- **Masking:** Causal mask geleceğe bilgi sızmasını engeller.

---

## İleride Genişletme Fikirleri

- Öğrenilebilir positional embedding eklemek.
- Daha büyük seq_len ve daha derin model denemek.
- FlashAttention veya xFormers ile attention hızlandırmak.
