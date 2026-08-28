# CoSQ Proje Hafıza Dosyası
**Proje No:** 126E534 — TÜBİTAK 1002-A Hızlı Destek Programı
**Proje Başlığı:** Chain-of-Self-Questioning (CoSQ): Büyük Dil Modellerinde Halüsinasyonu Azaltmaya Yönelik Pilot Bir Sorgulama Çerçevesi
**Yürütücü:** Doç. Dr. Ali Şenol — Tarsus Üniversitesi, Bilgisayar Mühendisliği
**Son Güncelleme:** 28 Ağustos 2026

> Bu dosya, projenin tüm geçmişini, alınan kararları ve kod altyapısının son halini özetler. Yeni bir oturumda bu dosyayı paylaşarak konuşmaya kaldığınız yerden devam edebilirsiniz.

---

## 1. PROJE DURUMU

| Aşama | Durum |
|---|---|
| Başvuru hazırlığı | ✅ Tamamlandı (5 revizyon turu) |
| TÜBİTAK değerlendirmesi | ✅ **KABUL EDİLDİ** (Karar yazısı: 18/06/2026, Sayı E-24395168-115.02-219519) |
| Sözleşme süreci | ✅ Tamamlandı; proje 15/08/2026 itibarıyla başladı |
| Kod altyapısı (pipeline) | ✅ Güncel Python paketi/CLI; ASU backend, açık-uçlu sunum, graded CoSQ ve rapor üretimi hazır |
| Deneylerin çalıştırılması | 🔄 Devam ediyor — deneysel çalışmalar yürütülüyor |

---

## 2. BAŞVURU GEÇMİŞİ — Alınan Kararlar

### 2.1. Scope Düzeltmeleri
- **İlk taslak:** 3 model (Llama 3-8B, GPT-4o, Claude 3.5) × 4 dataset — 1002 için aşırı kapsamlı bulundu.
- **Düzeltme:** Tek model (Llama 3-8B-Instruct), tek ana dataset (TruthfulQA MC1, 100 soru), GSM8K tamamlayıcı.
- **Sonuç:** Pilot/proof-of-concept niteliği netleşti.

### 2.2. Hipotez Netleştirmesi
- **H1 (nihai):** CoSQ uygulanan Llama 3-8B'nin TruthfulQA halüsinasyon oranı, CoT'ye kıyasla p<0.05 ve Cliff's Delta ≥ 0.3 (orta etki) ile daha düşük olacaktır.
- İlk versiyondaki "kayda değer iyileşme" gibi belirsiz ifadeler somut istatistiksel eşiklere dönüştürüldü.

### 2.3. Bütçe Düzeltmeleri (kritik hatalar giderildi)
- Bursiyer ücreti: 7.000 TL/ay (TÜBİTAK tablosuna aykırı) → **22.500 TL/ay, 3 ay = 67.500 TL** olarak düzeltildi.
- Genel Bütçe Tablosu toplam hatası (107.500 → 82.500, sonra GPU kalemi eklenince 92.500) düzeltildi.
- TRUBA'ya (ücretsiz kaynak) bütçe ayrılması hatası giderildi; ticari bulut GPU (Colab Pro+/AWS, 10.000 TL) B planı olarak eklendi.
- **Nihai bütçe kalemleri:** Sarf 15.000 + Hizmet (Bulut GPU) 10.000 + Bursiyer 67.500 = **92.500 TL**

### 2.4. Metodolojik Düzeltmeler
- TruthfulQA değerlendirmesi: GPT-judge gerektiren orijinal script yerine **MC1 (multiple-choice) formatı** kullanılarak judge bağımlılığı kaldırıldı.
- HR formülü: `HR = Yanlış / Toplam` (sıfır bölen riski not edildi).
- AR (Abstention Rate) metriği eklendi: aşırı çekimserlik riskine karşı denge analizi.
- Cliff's Delta etki büyüklüğü ölçütü eklendi (H1'e somut eşik kazandırdı).

### 2.5. İş Paketi (İP) Yapısı Düzeltmesi
- TÜBİTAK formu "makale yazımı, sonuç raporu iş paketi olarak sunulmamalı" uyarısı nedeniyle İP3 yeniden adlandırıldı: **"CoSQ Açık Kaynak Yazılım Paketi"** (teknik çıktı odaklı).
- Yayın/raporlama ayrı bir İP4 ("Yayın Hazırlığı ve Raporlama") olarak ayrıştırıldı.

### 2.6. Literatür Güçlendirmesi
Özgün değer bölümüne eklenen kritik referanslar (CoSQ'nun özgünlüğünü netleştirmek için):
- Kuhn vd. (2023) — Semantic Uncertainty
- Manakul vd. (2023) — SelfCheckGPT
- Wan vd. (2024) — CoT Rerailer
- Mir (2025) — Layer-wise Semantic Dynamics (LSD)
- Zhang (2024) — R-tuning
- Li vd. (2026) — I-CALM

### 2.7. Final İş Paketleri (Onaylanmış Hali)
| İP | Ad | Ay | Önem (%) |
|---|---|---|---|
| İP1 | CoSQ Prototip Geliştirme | 1-4 | 40 |
| İP2 | Deneysel Değerlendirme | 5-10 | 45 |
| İP3 | CoSQ Açık Kaynak Yazılım Paketi | 11-12 | 10 |
| İP4 | Yayın Hazırlığı ve Raporlama | 11-12 | 5 |

---

## 3. KOD ALTYAPISI — Güncel Durum

Proje artık eski `main.py` tabanlı tek dosya akışıyla değil, paketlenmiş `cosq` CLI üzerinden yürütülüyor. Kod GitHub'a push edilmiş durumda ve deneysel çalışmalar aktif olarak yapılıyor.

**Kaynak / GitHub dizini:**
`E:\AcademicWorks\Makaleler\Papers After PostDoc\1002\1002-github`

**Aktif deney dizini:**
`E:\AcademicWorks\Makaleler\Papers After PostDoc\1002\cosq_cluade\cosq`

**GitHub:**
https://github.com/senolali/1002

**Güncel altyapı özellikleri:**

1. `cosq run`, `cosq evaluate`, `cosq analyze`, `cosq report`, `cosq probe`, `cosq list` CLI komutları
2. ASU CreateAI backend ve model config dosyaları
3. TruthfulQA MC1 veri hattı; HuggingFace dataset adı `truthfulqa/truthful_qa` uyumluluğu
4. Closed-book MC sunum ve açık-uçlu (`open_ended`) sunum
5. Direct, CoT, CoT+Abstain, CoSQ, CoSQ-Gate ve CoSQ-Graded koşulları
6. Çoklu model kampanya çalıştırıcısı; anlamlı model setiyle karşılaştırma için hazır
7. Rapor üretimi: HR, AR, AA, Coverage, overall Accuracy, Phi, unparseable, coverage-matched risk
8. Açık-uçlu scoring düzeltmesi: negation korunur, option-unique tokenlar öncelenir, şüpheli eşleşme `unparseable` kalır

**Doğrulanmış durum:**

- Mock smoke run çalıştı.
- ASU `gpt4o` probe çalıştı.
- GPT-4o TruthfulQA pilot ve tau sweep sonuçları üretildi.
- `llama3-8b` açık-uçlu pilot sonucu üretildi ve scorer düzeltmesinden sonra API çağrısı yapmadan yeniden raporlandı.
- 28/08/2026 itibarıyla deneysel çalışmalar aktif olarak yürütülüyor.

---

## 4. PROJE YAPISI — Güncel Paket

```
cosq/
├── configs/
│   ├── experiment/         # smoke, pilot, open-ended ve model kampanya configleri
│   └── model/              # ASU/OpenAI/mock model tanımları
├── src/cosq/
│   ├── backends/           # mock, ASU CreateAI ve model yükleme
│   ├── data/               # TruthfulQA/JSONL veri yükleyicileri
│   ├── decision/           # binary threshold ve graded confidence karar kuralları
│   ├── eval/               # scorer, metrics, selective/coverage-matched analiz
│   ├── parsing/            # MC parsing, abstention detection, open-ended matcher
│   ├── report/             # markdown table ve istatistiksel analiz üretimi
│   ├── runner/             # deney yürütme ve kayıt yazımı
│   └── strategies/         # direct, cot, cot_abstain, cosq, cosq_gate, cosq_graded
├── tests/                  # pytest testleri
└── results/
    ├── runs/               # manifest/config ve ham kayıtlar
    └── tables/             # citable markdown raporları
```

---

## 5. ÇALIŞTIRMA TALİMATLARI — Güncel Akış

```bash
# Aktif dizin
cd "E:\AcademicWorks\Makaleler\Papers After PostDoc\1002\cosq_cluade\cosq"

# Sanal ortam
.venv\Scripts\activate

# Komutları gör
cosq --help
cosq list

# Model bağlantısını ucuz test et
cosq probe --model configs/model/asu_gpt4o.yaml
cosq probe --model configs/model/asu_llama3_8b.yaml

# Kod/pipeline smoke test
cosq run --config configs/experiment/smoke_mock.yaml --backend mock --allow-dirty

# Mevcut run'dan rapor üret / yeniden skorla (API çağrısı yapmaz)
cosq report results/runs/<RUN_DIZINI>

# Deney çalıştırma örneği
cosq run --config configs/experiment/pilot_truthfulqa_open.yaml --allow-dirty
```

**Beklenen çıktılar:**

- `results/runs/<run>/manifest.json` — model, config, revision, maliyet ve run özeti
- `results/runs/<run>/config.resolved.yaml` — koşulan config'in dondurulmuş hali
- `results/runs/<run>/records.jsonl` — ham cevap kayıtları; büyük olduğu için genellikle Git'e alınmaz
- `results/tables/<run>.md` — raporlanabilir markdown sonuç tablosu ve analiz JSON'u

---
## 6. SONRAKİ ADIMLAR

1. ✅ PYS sözleşme süreci tamamlandı; proje 15/08/2026 itibarıyla başladı
2. ⏳ Bursiyer (YL öğrencisi) belirlemek ve göreve başlatmak
3. ✅ ASU CreateAI erişimi ve model probe akışı doğrulandı
4. ✅ `cosq` CLI smoke/probe/report hattı doğrulandı
5. 🔄 Çoklu model açık-uçlu/graded deney kampanyalarını yürütmek ve raporlamak (İP1-İP2: Ay 1-10)
6. ⏳ Sonuçlara göre makale taslağına geçmek (İP4: Ay 11-12)

---

## 7. ÖNEMLİ NOTLAR / HATIRLATMALAR

- Deneyler şu anda yerel GPU yerine ASU CreateAI hosted API üzerinden yürütülüyor; hosted model revision bilgisi manifestlerde kayıt altına alınmalı.
- TruthfulQA değerlendirmesi GPT-judge gerektirmiyor (MC1 formatı kullanılıyor) — ek API maliyeti yok.
- CoSQ kural motorunda binary ve graded confidence varyantları mevcut; sonuçlar AA + Coverage, HR, Phi ve coverage-matched risk ile birlikte okunmalı.
- TruthfulQA örneklemi ve her run manifest/config ile izlenebilir; sample size istatistiksel olarak soru sayısıdır, API query sayısı değildir.
- Sözleşme süreci tamamlandı; proje resmi başlangıç tarihi 15/08/2026. Şu an deneysel çalışmalar yürütülüyor.

---

*Bu dosyayı yeni bir Claude oturumuna yükleyerek "Bu projeye devam ediyorum, hafıza dosyasını okudun mu?" diyebilirsiniz.*
---

## 8. 2026-08-28 Guncel metodoloji notu: CoSQ accuracy nasil okunacak?

- CoSQ abstaining bir sistemdir. Bu nedenle CoSQ'un "cevap verdiginde dogrulugu" icin ana metrik `AA = correct / (correct + wrong)` olmalidir.
- `Coverage = (correct + wrong) / N` mutlaka AA ile birlikte raporlanmalidir. Yuksek AA, cok dusuk coverage ile tek basina yeterli bir basari iddiasi degildir.
- `Accuracy = correct / N` yine raporlanir ama bu "overall / coverage-inclusive accuracy"dir. CoSQ bilmedigi soruya bilincli olarak cevap vermedigi icin bu metrik abstention'i paydada cezalandirir.
- H1 icin ana okuma hala HR dususu + AR + Phi'dir. H2 icin asil test coverage-matched risk ve CoSQ'un answered subset'inde CoT'a gore daha dusuk risk verip vermedigidir.
- Acik-uclu TruthfulQA sonuclarinda otomatik lexical scoring bir olcum katmanidir. 2026-08-28 denetiminde bazi cevaplarin soru kelimeleri veya negation kaybi yuzunden yanlis secenege eslestirilebildigi goruldu. Kod negation'i koruyacak, option-unique token'lari onceleyecek ve emin olmadigi eslesmeleri `unparseable` birakacak sekilde guncellendi.
- Eski `records.jsonl` dosyalari gecerlidir; API tekrar calistirmadan `cosq report` ile yeniden skorlanabilir. Acik-uclu sayilar final rapora girmeden once stratified human audit veya ayrica valide edilmis semantic judge ile kontrol edilmelidir.




---

## 9. 2026-08-28 SCI ana kosum plani

- GitHub README/docs tarafinda ASU/CreateAI adi acikca kullanilmayacak; public metinlerde "hosted provider" dili korunacak.
- Operasyonel calisma ASU altyapisi uzerinden yerel aktif dizinde yapiliyor: `E:\AcademicWorks\Makaleler\Papers After PostDoc\1002\cosq_cluade\cosq`.
- Tam SCI kosumu icin yerel script eklendi: `scripts/run_sci_campaign.py`.
- Varsayilan model paneli: `llama3-8b`, `llama3-70b`, `llama4_scout-17b`, `gpt-oss-20b`, `gpt-oss-120b`, `gemma3_12b_it`, `gemma4_31b_it`, `mistral-7b`, `gpt5_4_mini`.
- Panelin mantigi: cogu acik kaynak/acik agirlik model; Llama 3 -> Llama 4 jenerasyon kontrolu; Llama 3 8B -> 70B ve GPT-OSS 20B -> 120B olcek karsilastirmasi; Gemma 3 -> Gemma 4 ikinci acik model ailesi; Mistral pre-registered sensitivity; GPT-5.4 Mini yeni frontier proprietary referans.
- Ana protokol: acik-uclu TruthfulQA, `n=300`, `repeats=3`, stratejiler `direct`, `cot`, `cot_abstain`, `cosq`, `cosq_graded_min080`, `cosq_graded_mean080`.
- Script her model icin probe -> run -> analyze -> report akisini calistirir; `results/campaigns/sci_core/sci_campaign_summary.md`, `.csv` ve SVG grafikler uretir.
- Mock smoke test basariyla calisti: run/analyze/report/summary/figures uretildi; `ruff check scripts/run_sci_campaign.py` ve `py_compile` temiz.
- Onerilen tam kosum komutu:

```bash
cd "E:\AcademicWorks\Makaleler\Papers After PostDoc\1002\cosq_cluade\cosq"
.venv\Scripts\activate
python scripts\run_sci_campaign.py --n 300 --repeats 3 --keep-going
```

- Hakem riski azaltilmis okuma: CoSQ icin accuracy iddiasi AA + Coverage olarak verilecek; H1 HR dususu, H2 coverage-matched risk ile yorumlanacak; acik-uclu scorer sonuclari final rapordan once stratified human audit ile kontrol edilecek.

