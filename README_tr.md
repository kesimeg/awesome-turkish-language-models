# awesome-turkish-language-models [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
![Awesome turkish language models image](https://github.com/user-attachments/assets/12b48069-b177-4278-b225-f29b503f60b0)
Türkçe yapay zeka modelleri, veri setleri ve makalelerden oluşan özenle derlenmiş bir liste

Bu reponun amacı, Türkçe yapay zeka modelleri, veri setleri ve makaleler hakkındaki bilgileri paylaşmak ve yaygınlaştırmaktır. Bu tür Türkçe kaynakların sayısı az ve dağınık durumda. Bu repo, bu kaynaklardan özenle seçilmiş bir derlemeyi bir araya getirmeyi amaçlamaktadır. Bu, tüm Türkçe NLP/LLM modellerinin veya veri setlerinin listesi değil, bir seçkidir. Dolayısıyla, BERT veya LLaMA tabanlı tüm modeller buraya dahil edilmeyecektir. Aynı durum düşük kaliteli Google Translate çevirilerine sahip verisetleri için de geçerlidir. Buradaki her kaynağın kendine özgü bir unsuru olmasını hedefliyoruz. Bu kendine özgünlük, model performansı, modelin belirli bir işi yapan az sayıda modelden biri olması, grupları/şirketleri öne çıkarmak (herkes açık kaynak olacak şekilde paylaşım yapmıyor, o halde neden takdir etmeyelim!) vb. olabilir. Eklemek istediğiniz bir şey varsa, isteklerinizi memnuniyetle kabul ederiz :smirk: , lütfen katkı bölümüne göz atın.
## Table of Contents


* **[Modeller](#modeller)** 

* **[Verisetleri](#verisetleri)** 

* **[Canlı Liderlik tabloları](#canlı-liderlik-tabloları)**  

* **[Kıyaslama verisetleri](#kıyaslama-verisetleri)**  

* **[Makaleler](#makaleler)**  

* **[Eğitimler ve Kodlar](#eğitimler-ve-kodlar)**  

* **[Araçlar ve APIler](#araçlar-ve-apiler)** 

* **[MCPler](#mcpler)** 

* **[YZ'nin Türkiyedeki Durumu (Projeler,ürünler, gruplar vs.)](#yznin-türkiyedeki-durumu)** 

* **[Çeşitli Kaynaklar](#çeşitli-kaynaklar)**  

* **[Katkıda Bulunma](#katkıda-bulunma)**  


### Modeller

#### LLMs
- [ytu-ce-cosmos/Turkish-Llama](https://huggingface.co/ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1)
- [Trendyol/Llama-3-Trendyol-LLM-8b-chat-v2.0](https://huggingface.co/Trendyol/Llama-3-Trendyol-LLM-8b-chat-v2.0)
- [Trendyol/Trendyol-LLM-7B-chat-v4.1.0](https://huggingface.co/Trendyol/Trendyol-LLM-7B-chat-v4.1.0)
- [TURKCELL/Turkcell-LLM-7b-v1](https://huggingface.co/TURKCELL/Turkcell-LLM-7b-v1)
- [KOCDIGITAL/Kocdigital-LLM-8b-v0.1](https://huggingface.co/KOCDIGITAL/Kocdigital-LLM-8b-v0.1)
- [WiroAI/OpenR1-Qwen-7B-Turkish](https://huggingface.co/WiroAI/OpenR1-Qwen-7B-Turkish) Akıl yürütme modeli
- [WiroAI/wiroai-turkish-llm-9b](https://huggingface.co/WiroAI/wiroai-turkish-llm-9b)
- [ytu-ce-cosmos/Turkish-Gemma-9b-v0.1](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-v0.1)
- [Trendyol/Trendyol-LLM-8B-T1](https://huggingface.co/Trendyol/Trendyol-LLM-8B-T1) Qwen3'e ince ayar yapılmış, düşünme modu var
- [ytu-ce-cosmos/Turkish-Gemma-9b-T1](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-T1)
- [vngrs-ai/Kumru-2B](https://huggingface.co/vngrs-ai/Kumru-2B) Kumru modeli Mistral mimarisine sahip. [Sıfırdan](https://medium.com/vngrs/kumru-llm-34d1628cfd93) (ince ayar yapılmış bir modeli değil) eğitilmiş bir model.
- [Trendyol/Trendyol-LLM-Asure-12B](https://huggingface.co/Trendyol/Trendyol-LLM-Asure-12B) Çok kipli model
- [ytu-ce-cosmos/Turkish-Gemma-4b-T1-Scout](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-4b-T1-Scout) Akıl yürütme, internet aramaları için eğitilmiş

#### VLMs
- [ytu-ce-cosmos/Turkish-LLaVA](https://huggingface.co/ytu-ce-cosmos/Turkish-LLaVA-v0.1)
- [Trendyol/Trendyol-Vision-Flash](https://huggingface.co/Trendyol/Trendyol-Vision-Flash) Trendyol'un e-ticaret odaklı VLM modeli (küçük versiyon)
- [Trendyol/Trendyol-Vision-Master](https://huggingface.co/Trendyol/Trendyol-Vision-Master) Trendyol'un e-ticaret odaklı VLM modeli (büyük versiyon)

#### NLP
- [Trendyol/tybert](https://huggingface.co/Trendyol/tybert)
- [Trendyol/tyroberta](https://huggingface.co/Trendyol/tyroberta)
- [ytu-ce-cosmos/turkish-base-bert-uncased](https://huggingface.co/ytu-ce-cosmos/turkish-base-bert-uncased)
- [ytu-ce-cosmos/turkish-colbert](https://huggingface.co/ytu-ce-cosmos/turkish-colbert)
- [ytu-ce-cosmos/turkish-gpt2-large](https://huggingface.co/ytu-ce-cosmos/turkish-gpt2-large)
- [dbmdz/bert-base-turkish-128k-uncased](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased)
- [TURKCELL/bert-offensive-lang-detection-tr](https://huggingface.co/TURKCELL/bert-offensive-lang-detection-tr)
- [asafaya/kanarya-2b](https://huggingface.co/asafaya/kanarya-2b)
- [boun-tabi-LMG/TURNA](https://huggingface.co/boun-tabi-LMG/TURNA)
- [Helsinki-NLP group](https://huggingface.co/Helsinki-NLP) Türkçe için pek çok çeviri modeli var
- [VRLLab/TurkishBERTweet](https://huggingface.co/VRLLab/TurkishBERTweet) Tweetlerde duygu analizi
- [akdeniz27/bert-base-turkish-cased-ner](https://huggingface.co/akdeniz27/bert-base-turkish-cased-ner)
- [Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0](https://huggingface.co/Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0) Türkçe ve çok dilli embedding modeli
- [artiwise-ai/modernbert-base-tr-uncased](https://huggingface.co/artiwise-ai/modernbert-base-tr-uncased)
- [ytu-ce-cosmos/turkish-e5-large](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large) Türkçe retrieval modeli
- [ModernBERT-TR Collection](https://huggingface.co/collections/ytu-ce-cosmos/modernbert-tr) Türkçe için bir ModerBERT modeli. Bu çeşitli farklı modellerden oluşan bir koleksiyondur.  İçerisinde ModernBERT, reranker, PII, guardrail ve farklı pek çok modeli bulundurmaktadır.

#### Ses modelleri
- [Trendyol/Trendyol-TTS](https://huggingface.co/Trendyol/Trendyol-TTS) Türkçe metinden ses üreten model

#### Çok-kipli modeller
- [kesimeg/lora-turkish-clip](https://huggingface.co/kesimeg/lora-turkish-clip) Türkçe veri ile ince ayar yapılmış CLIP modeli


### Verisetleri

#### Sadece metinden oluşanlar
- [merve/turkish_instructions](https://huggingface.co/datasets/merve/turkish_instructions) Instruction veriseti
- [BrewInteractive/alpaca-tr](https://huggingface.co/datasets/BrewInteractive/alpaca-tr/viewer/default/train?p=2&views%5B%5D=train) Instruction veriseti
- [Metin/WikiRAG-TR](https://huggingface.co/datasets/Metin/WikiRAG-TR)
- [MBZUAI/Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X/viewer/tr?views%5B%5D=tr)
- [Helsinki-NLP group](https://huggingface.co/Helsinki-NLP) Türkçe için pek çok çeviri verisetlerini içeriyor
- [turkish-nlp-suite/turkish-wikiNER](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER)
- [turkish-nlp-suite/InstrucTurca](https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca)
- [WiroAI/dolphin-r1-turkish](https://huggingface.co/datasets/WiroAI/dolphin-r1-turkish) Akıl yürütme verisetleri
- [allenai/c4](https://huggingface.co/datasets/allenai/c4) İnternetten toplanmış veriler
- [HPLT/HPLT2.0_cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned/viewer/tur_Latn) İnternetten toplanmış veriler
- [unimelb-nlp/wikiann](https://huggingface.co/datasets/unimelb-nlp/wikiann) Varlık ismi tanıma
- [TUR2SQL](https://github.com/alibugra/TUR2SQL) Yazıdan SQL sorgularına çeviri veriseti
- [emre/ct_tree_of_thought_turkish](https://huggingface.co/datasets/emre/ct_tree_of_thought_turkish) Türkçe Tree of Thoughts (ToT) veriseti 
- [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) ~95 milyon Türkçe metin var
- [TURSpider](https://github.com/alibugra/TURSpider) Metinden SQL sorgulamaya çeviri veriseti
- [vngrs-ai/vngrs-web-corpus](https://huggingface.co/datasets/vngrs-ai/vngrs-web-corpus) İnternetten toplanmış farklı farklı verisetlerinden oluşturulmuş bir pretraining veriseti
- [HuggingFaceFW/finetranslations](https://huggingface.co/datasets/HuggingFaceFW/finetranslations) Çeviri için 58 milyon Türkçe-İngilizce metin bulunmaktadır. Çeviriler, Gemma3-27B kullanılarak oluşturulmuş (orijinal Türkçe veri kümesinden İngilizceye çevrilmiş)
- [ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0](https://huggingface.co/datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0) İnternetten toplanan pretraining verileri
- [alibayram/diyalog-dataset](https://huggingface.co/datasets/alibayram/diyalog-dataset) Multi turn konuşma veri seti
- [BTX24/turkish-privacy-pii-ner](https://huggingface.co/datasets/BTX24/turkish-privacy-pii-ner) Sentetik kişisel veriler için  varlık tanıma veri seti
- [Tuguberk/turkish-hermes-function-calling](https://huggingface.co/datasets/Tuguberk/turkish-hermes-function-calling) Hermes function calling verisetining Türkçe çevirisi
- [OnerAYTAS/Turkish_prompt_injection_jailbreak_dataset]( https://huggingface.co/datasets/OnerAYTAS/Turkish_prompt_injection_jailbreak_dataset) Türkçe jailbreak kategorizasyon veriseti
- [berhaan/Turkish-CodeAlpaca-20k](https://huggingface.co/datasets/berhaan/Turkish-CodeAlpaca-20k) Kodlama üzerine bir instruction following veriseti
- [OnerAYTAS/Turkish_self_harm_dataset](https://huggingface.co/datasets/OnerAYTAS/Turkish_self_harm_dataset) Kendine Zarar Verme konulu metinlerden oluşan bir veri seti. Veri seti farklı İngilizce veri setlerinin Türkçeye çevrilmesi ile oluşturulmuştur.

#### Metin & Görsel/Videolar
- [ytu-ce-cosmos/Turkish-LLaVA-Finetune](https://huggingface.co/datasets/ytu-ce-cosmos/Turkish-LLaVA-Finetune)
- [ytu-ce-cosmos/Turkish-LLaVA-Pretrain](https://huggingface.co/datasets/ytu-ce-cosmos/Turkish-LLaVA-Pretrain)
- [ytu-ce-cosmos/turkce-kitap](https://huggingface.co/datasets/ytu-ce-cosmos/turkce-kitap)
- [99eren99/LLaVA1.5-Data-Turkish](https://huggingface.co/datasets/99eren99/LLaVA1.5-Data-Turkish)
- [TasvirEt](https://www.kaggle.com/datasets/begum302553/tasviret-flickr8k-turkish)
- [nezahatkorkmaz/turkish-medical-vqa-evaluated](https://huggingface.co/datasets/nezahatkorkmaz/turkish-medical-vqa-evaluated) Medikal görüntülerden soru yanıtlama veri seti
- [nezahatkorkmaz/unsloth-pmc-vqa-tr](https://huggingface.co/datasets/nezahatkorkmaz/unsloth-pmc-vqa-tr) Medikal görüntüler için soru-cevap veri seti. PMC-VQA verisetinin çevirisi. Orijinal verisetindeki görüntülere erişim gerektiriyor.
- [BosphorusSign22k](https://ogulcanozdemir.github.io/bosphorussign22k/) İşaret dili
- [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) 1,7 milyon Türkçe örnek içerir. Pretraining ve RAG uygulamaları için son derece uygun bir PDF veri kümesi.
- [ituperceptron/image-captioning-turkish](https://huggingface.co/datasets/ituperceptron/image-captioning-turkish) Image captioning veriseti. 200k uzun, 100k kısa caption örneği var

#### Metin & Ses
- [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) Bu verisetining daha eski versiyonları da mevcut v16,v15, vs.


### Canlı Liderlik tabloları
- [malhajar/OpenLLMTurkishLeaderboard_v0.2](https://huggingface.co/spaces/malhajar/OpenLLMTurkishLeaderboard_v0.2)
- [KUIS-AI/Cetvel](https://huggingface.co/spaces/KUIS-AI/Cetvel)
- [kesimeg/Turkish-rewardbench](https://huggingface.co/spaces/kesimeg/Turkish-rewardbench) Reward modellerinin karşılaştırması
- [TurkBench/TurkBench](https://huggingface.co/spaces/TurkBench/TurkBench)
- [newmindai/Mezura](https://huggingface.co/spaces/newmindai/Mezura) RAG, İnsan değerlendirmesi (ELO skoru) ve değerlendirme metrikleri var. Ayrıca malhajar/OpenLLMTurkishLeaderboard_v0.2 canlı liderlik tablosunu da içermekte.
- [newmindai/Mizan](https://huggingface.co/spaces/newmindai/Mizan) Embedding modelleri için bir canlı liderlik tablosu. Embedding modellerinin retrieval, kümeleme ve benzeri problemlerdeki performanslarını karşılaştırıyor.

### Kıyaslama verisetleri

#### Sadece metinden oluşanlar
- [AYueksel/TurkishMMLU](https://huggingface.co/datasets/AYueksel/TurkishMMLU)
- [alibayram/turkish_mmlu](https://huggingface.co/datasets/alibayram/turkish_mmlu)
- [ytu-ce-cosmos/gsm8k_tr](https://huggingface.co/datasets/ytu-ce-cosmos/gsm8k_tr)
- [Holmeister's Collections](https://huggingface.co/collections/Holmeister/turkish-llm-multi-prompt-evaluation-datasets-676994cd18391bb6e813bec3) 17 verisetinden oluşan 11 farklı konuda performans ölçen bir koleksiyon(Truthfulness, fairness, summarization vb.). Daha fazlası için [makale](https://www.sciencedirect.com/science/article/abs/pii/S0957417425010437) 
- [CohereLabs/Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) Türkçe de dahil olmak üzere pek çok dil için MMLU verisetinin çevirileri var
- [mrlbenchmarks/global-piqa-nonparallel](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel) Kültürel değerler için bir karşılaştırma veri seti
- [ytu-ce-cosmos/gpqa-extended_tr](https://huggingface.co/datasets/ytu-ce-cosmos/gpqa-extended_tr) Lisansüstü seviye bilim soruları
- [CohereLabsCommunity/multilingual-reward-bench](https://huggingface.co/datasets/CohereLabsCommunity/multilingual-reward-bench) Reward kıyaslama veriseti (terchih/beğeni tahmini)
- [boun-tabilab's Tabilab Collection](https://huggingface.co/collections/boun-tabilab/tabibench) Klasik NLP problemleri için oluşturulmuş bir koleksiyon (NER, özetleme, sınıflandırma, etc.)
- [AIM-Intelligence/XL-SafetyBench](https://huggingface.co/datasets/AIM-Intelligence/XL-SafetyBench) Jailbreak ve kültürel hassasiyeti ölçen kıyaslama veri seti

#### Görsel ve Metinlerden oluşanlar
- [CohereLabs/m-WildVision](https://huggingface.co/datasets/CohereLabs/m-WildVision)
- [CohereLabs/AyaVisionBench](https://huggingface.co/datasets/CohereLabs/AyaVisionBench)
- [kesimeg/MMStar_tr](https://huggingface.co/datasets/kesimeg/MMStar_tr)
- [metu-yks/yksbench](https://huggingface.co/datasets/metu-yks/yksbench) Üniversite giriş sınavından oluşturulmuş bir görsel veriseti. Sorular matematik, geometri, fizik, kimya, biyoloji ve coğrafya gibi alanları kapsamakta
- [ytu-ce-cosmos/tubitak-science-olympiad-tr](https://huggingface.co/datasets/ytu-ce-cosmos/tubitak-science-olympiad-tr) TÜBITAK Bilim Olimpiyatları sorularından oluşuyor.Görseller soruların ekran görüntüleri 
- [Berkesule/Turkish-Visual-Reasoning-Dataset](https://huggingface.co/datasets/Berkesule/Turkish-Visual-Reasoning-Dataset) Soyut düşünmeyi ölçen görsel soru cevap veriseti

### Makaleler
- [Cosmos-LLaVA: Chatting with the Visual](https://arxiv.org/pdf/2412.02760)
- [Introducing cosmosGPT: Monolingual Training for Turkish Language Models](https://arxiv.org/pdf/2404.17336)
- [TurkishMMLU: Measuring Massive Multitask Language Understanding in Turkish](https://arxiv.org/abs/2407.12402)
- [TURSpider: A Turkish Text-to-SQL Dataset and LLM-Based Study](https://ieeexplore.ieee.org/document/10753591)
- [How do LLMs perform on Turkish? A multi-faceted multi-prompt evaluation](https://www.sciencedirect.com/science/article/abs/pii/S0957417425010437) Farklı LLMlerin TÜrkçe performansları
- [Evaluating the Quality of Benchmark Datasets for Low-Resource Languages: A Case Study on Turkish](https://arxiv.org/abs/2504.09714)
- [YKSBench: Stress-Testing Multimodal Models with Exam-Style Questions](https://openreview.net/pdf?id=qgAjoo3cJE) YKSBench kıyaslama verisetinin makalesi.
- [TurkBench: A Benchmark for Evaluating Turkish Large Language Models](https://www.arxiv.org/pdf/2601.07020) TurkBench kıyaslama verisetinin makalesi.

### Eğitimler ve Kodlar
- [METU NLP Lab Git repo](https://github.com/metunlp)
- [wikipedia ToT data generation notebook](https://colab.research.google.com/drive/1mHOtErnLLoifkm0ySLc3_F9UqRa3SkSV?usp=sharing)


### Araçlar ve APIler
- [Glosbe](https://tr.glosbe.com/)
- [Wiktionary](https://tr.wiktionary.org/wiki/Vikis%C3%B6zl%C3%BCk:Anasayfa)
- [Zemberek](https://github.com/ahmetaa/zemberek-nlp) Türkçe doğal dil işleme araçları
- [3rt4nm4n/turkish-apis](https://github.com/3rt4nm4n/turkish-apis) Türkçe APIların listesi

### MCPler
- [THY-MCP](https://mcp.turkishtechlab.com/)
- [borsa-mcp](https://github.com/saidsurucu/borsa-mcp) Türk borsaları için MCP Server 
- [yargi-mcp](https://github.com/saidsurucu/yargi-mcp) Türk hukuk kaynaklarına erişim için MCP Server 
- [mevzuat-mcp](https://github.com/saidsurucu/mevzuat-mcp) Adalet Bakanlığı'na ait Mevzuat Bilgi Sistemine erişim için bir MCP Server
- [yoktez-mcp](https://github.com/saidsurucu/yoktez-mcp) YÖKTEZ'e erişim için MCP Server 
- [yokatlas-mcp](https://github.com/saidsurucu/yokatlas-mcp) YOK Atlas'a erişim için MCP Server

### YZ'nin Türkiyedeki Durumu
- [KUIS-AI Youtube channel](https://www.youtube.com/@kuisaicenter)
- [TR-AI Youtube channel](https://www.youtube.com/c/T%C3%BCrkiyeYapayZeka%C4%B0nisiyatifi)
- [Trendyol Tech Youtube channel](https://www.youtube.com/@TrendyolTech) İş akışlarına ve ürünlerine AI modellerinin nasıl entegre edildiğini anlatan videolar var

### Çeşitli Kaynaklar
- [Mukayese: Turkish NLP Strikes Back](https://mukayese.tdd.ai/#/)
- [Mukayese github repo](https://github.com/alisafaya/mukayese)
- [Wikipedia dumps](https://dumps.wikimedia.org/) Veriseti olarak kullanılabilir
- [Turkish Encoder-only Models List](https://huggingface.co/collections/atasoglu/turkish-encoder-only-models-65e2604fa0a2f649da9477e8) Sadece kodlayıcılardan oluşan Türkçe modellerin bir listesi
- [Turkish Instruction Datasets List](https://huggingface.co/collections/atasoglu/turkish-instruction-datasets-6601d92fa6d901e554d98979) Türkçe Instruction verisetlerinin bir koleksiyonu
- [Turkish Vision-Language Datasets List](https://huggingface.co/collections/atasoglu/turkish-vision-language-datasets-66e7c563750d486e30732dd4) Türkçe VLM modellerinin bir koleksiyonu
- [Cosmos App](https://play.google.com/store/apps/details?id=com.cosmos.cosmos&hl=tr-TR) Cosmos AI araştırma grubunun modelinin yer aldığı uygulama (iOS versiyonu da mevcut)
- [ITU NLP Research Tools and Resources](https://ddi.itu.edu.tr/en/toolsandresources)

### Katkıda Bulunma
Buraya eklenmesi gerektiğini düşündüğünüz bir şey varsa, pull request gönderin! Pull request göndermeden önce, modelin/veri kümesinin/vb. yeterli kaliteye ve özgünlüğe sahip olup olmadığını lütfen değerlendirin. Huggingfacete bir sürü LLama ve BERT’in ince ayarları olan model mevcut, aynı durum veri kümeleri için de geçerli. Birçok veri kümesinin birden fazla makine çeviri versiyonu bulunuyor. Bu durum, kaliteli kaynakları bulmayı zorlaştırıyor. Bu listeyi olabildiğince özenle derlenmiş halde tutmak, ancak yine de yeterli sayıda kaynağı kapsayabilmek istiyoruz.