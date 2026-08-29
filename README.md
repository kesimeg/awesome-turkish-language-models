# awesome-turkish-language-models [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
![Awesome turkish language models image](https://github.com/user-attachments/assets/12b48069-b177-4278-b225-f29b503f60b0)
A curated list of Turkish AI models, datasets, papers

The purpose of this repo is to share and spread the information of Turkish AI models, datasets and papers. The amount of these Turkish resources are low and spread across the web. This repo aims to bring a curated selection of these resources together. This is not a list of all Turkish NLP/LLM models or datasets but a selection. So not all BERT or LLaMA based models are gonna make it here. The same applies to low quality Google translate translations of datasets. We aim each entry to have some kind of unique element to its own. This can be model performance, uniqueness in the task, highlighting the groups/companies (not everyone share their stuff so why not appreciate it!) etc. If you want to add anything you are welcome :smirk: , please check out the contributing section.

## Table of Contents


* **[Models](#models)** 

* **[Datasets](#datasets)** 

* **[Live Leaderboards](#live-leaderboards)**  

* **[Benchmark Datasets](#benchmark-datasets)**  

* **[Papers](#papers)**  

* **[Tutorials and Codes](#tutorials-and-codes)**  

* **[Tools and APIs](#tools-and-apis)** 

* **[MCPs](#mcps)** 

* **[State of AI in Türkiye(Projects, products, groups etc.)](#state-of-ai-in-türkiye)** 

* **[Miscellaneous](#miscellaneous)**  

* **[Contributing](#contributing)**  


### Models

#### LLMs
- [ytu-ce-cosmos/Turkish-Llama](https://huggingface.co/ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1)
- [Trendyol/Llama-3-Trendyol-LLM-8b-chat-v2.0](https://huggingface.co/Trendyol/Llama-3-Trendyol-LLM-8b-chat-v2.0)
- [Trendyol/Trendyol-LLM-7B-chat-v4.1.0](https://huggingface.co/Trendyol/Trendyol-LLM-7B-chat-v4.1.0)
- [TURKCELL/Turkcell-LLM-7b-v1](https://huggingface.co/TURKCELL/Turkcell-LLM-7b-v1)
- [KOCDIGITAL/Kocdigital-LLM-8b-v0.1](https://huggingface.co/KOCDIGITAL/Kocdigital-LLM-8b-v0.1)
- [WiroAI/OpenR1-Qwen-7B-Turkish](https://huggingface.co/WiroAI/OpenR1-Qwen-7B-Turkish) Reasoning model
- [WiroAI/wiroai-turkish-llm-9b](https://huggingface.co/WiroAI/wiroai-turkish-llm-9b)
- [ytu-ce-cosmos/Turkish-Gemma-9b-v0.1](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-v0.1)
- [Trendyol/Trendyol-LLM-8B-T1](https://huggingface.co/Trendyol/Trendyol-LLM-8B-T1) Qwen3 finetune, has thinking mode
- [ytu-ce-cosmos/Turkish-Gemma-9b-T1](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-T1)
- [vngrs-ai/Kumru-2B](https://huggingface.co/vngrs-ai/Kumru-2B) Kumru model has the architecture of Mistral. Its a model trained from [scratch](https://medium.com/vngrs/kumru-llm-34d1628cfd93) (not a finetune).
- [Trendyol/Trendyol-LLM-Asure-12B](https://huggingface.co/Trendyol/Trendyol-LLM-Asure-12B) Mutimodal model
- [ytu-ce-cosmos/Turkish-Gemma-4b-T1-Scout](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-4b-T1-Scout) Trained for reasoning and web-search tool calling

#### VLMs
- [ytu-ce-cosmos/Turkish-LLaVA](https://huggingface.co/ytu-ce-cosmos/Turkish-LLaVA-v0.1)
- [Trendyol/Trendyol-Vision-Flash](https://huggingface.co/Trendyol/Trendyol-Vision-Flash) Trendyol's e-commerce focused VLM (small version)
- [Trendyol/Trendyol-Vision-Master](https://huggingface.co/Trendyol/Trendyol-Vision-Master) Trendyol's e-commerce focused VLM (large version)

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
- [Helsinki-NLP group](https://huggingface.co/Helsinki-NLP) Lots of translation models for turkish
- [VRLLab/TurkishBERTweet](https://huggingface.co/VRLLab/TurkishBERTweet) Tweet sentiment analysis
- [akdeniz27/bert-base-turkish-cased-ner](https://huggingface.co/akdeniz27/bert-base-turkish-cased-ner)
- [Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0](https://huggingface.co/Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0) Turkish and multilingual embeddings
- [artiwise-ai/modernbert-base-tr-uncased](https://huggingface.co/artiwise-ai/modernbert-base-tr-uncased)
- [ytu-ce-cosmos/turkish-e5-large](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large) Turkish retrieval model
- [ModernBERT-TR Collection](https://huggingface.co/collections/ytu-ce-cosmos/modernbert-tr) ModernBERT model for Turkish. This is a collection of many models. It includes ModernBERT, reranker, PII, guardrail and various other models.

#### Speech models
- [Trendyol/Trendyol-TTS](https://huggingface.co/Trendyol/Trendyol-TTS) Turkish Text-to-Speech model

#### Multi-modal models
- [kesimeg/lora-turkish-clip](https://huggingface.co/kesimeg/lora-turkish-clip) CLIP model finetuned on turkish dataset


### Datasets

#### Text only
- [merve/turkish_instructions](https://huggingface.co/datasets/merve/turkish_instructions) Instruction tuning dataset
- [BrewInteractive/alpaca-tr](https://huggingface.co/datasets/BrewInteractive/alpaca-tr/viewer/default/train?p=2&views%5B%5D=train) Instruction tuning dataset
- [Metin/WikiRAG-TR](https://huggingface.co/datasets/Metin/WikiRAG-TR)
- [MBZUAI/Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X/viewer/tr?views%5B%5D=tr)
- [Helsinki-NLP group](https://huggingface.co/Helsinki-NLP) Lots of translation models datasets for turkish
- [turkish-nlp-suite/turkish-wikiNER](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER)
- [turkish-nlp-suite/InstrucTurca](https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca)
- [WiroAI/dolphin-r1-turkish](https://huggingface.co/datasets/WiroAI/dolphin-r1-turkish) Reasoning dataset
- [allenai/c4](https://huggingface.co/datasets/allenai/c4) Web scrape
- [HPLT/HPLT2.0_cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned/viewer/tur_Latn) Web scrape
- [unimelb-nlp/wikiann](https://huggingface.co/datasets/unimelb-nlp/wikiann) NER
- [TUR2SQL](https://github.com/alibugra/TUR2SQL) Text to SQL query dataset
- [emre/ct_tree_of_thought_turkish](https://huggingface.co/datasets/emre/ct_tree_of_thought_turkish)
Turkish Tree of Thoughts (ToT) dataset 
- [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) Has ~95 million turkish text
- [TURSpider](https://github.com/alibugra/TURSpider) Text-to-SQL dataset
- [vngrs-ai/vngrs-web-corpus](https://huggingface.co/datasets/vngrs-ai/vngrs-web-corpus) Pretraining data which is a collection of different datasets crawled from the internet
- [HuggingFaceFW/finetranslations](https://huggingface.co/datasets/HuggingFaceFW/finetranslations) Has 58 Million Turkish-English text pairs for translation. Translations were generated with Gemma3-27B (From original Turkish dataset to English)
- [ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0](https://huggingface.co/datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0) Pretraining data crawled from the internet
- [alibayram/diyalog-dataset](https://huggingface.co/datasets/alibayram/diyalog-dataset) Multi turn conversation data
- [BTX24/turkish-privacy-pii-ner](https://huggingface.co/datasets/BTX24/turkish-privacy-pii-ner) Synthetic PII NER dataset
- [Tuguberk/turkish-hermes-function-calling](https://huggingface.co/datasets/Tuguberk/turkish-hermes-function-calling) Turkish translation of Hermes function calling dataset
- [OnerAYTAS/Turkish_prompt_injection_jailbreak_dataset]( https://huggingface.co/datasets/OnerAYTAS/Turkish_prompt_injection_jailbreak_dataset) Turkish jailbreak categorization dataset
- [berhaan/Turkish-CodeAlpaca-20k](https://huggingface.co/datasets/berhaan/Turkish-CodeAlpaca-20k) An instruction following dataset for coding.
- [OnerAYTAS/Turkish_self_harm_dataset](https://huggingface.co/datasets/OnerAYTAS/Turkish_self_harm_dataset) The dataset contains texts with the topic of self harm. It was created by translating various English datasets into Turkish

#### Text & Images/Videos
- [ytu-ce-cosmos/Turkish-LLaVA-Finetune](https://huggingface.co/datasets/ytu-ce-cosmos/Turkish-LLaVA-Finetune)
- [ytu-ce-cosmos/Turkish-LLaVA-Pretrain](https://huggingface.co/datasets/ytu-ce-cosmos/Turkish-LLaVA-Pretrain)
- [ytu-ce-cosmos/turkce-kitap](https://huggingface.co/datasets/ytu-ce-cosmos/turkce-kitap)
- [99eren99/LLaVA1.5-Data-Turkish](https://huggingface.co/datasets/99eren99/LLaVA1.5-Data-Turkish)
- [TasvirEt](https://www.kaggle.com/datasets/begum302553/tasviret-flickr8k-turkish)
- [nezahatkorkmaz/turkish-medical-vqa-evaluated](https://huggingface.co/datasets/nezahatkorkmaz/turkish-medical-vqa-evaluated) Medical image question and answer dataset
- [nezahatkorkmaz/unsloth-pmc-vqa-tr](https://huggingface.co/datasets/nezahatkorkmaz/unsloth-pmc-vqa-tr) Medical image question answering dataset. Translated from PMC-VQA dataset. Requires access to images from original dataset.
- [BosphorusSign22k](https://ogulcanozdemir.github.io/bosphorussign22k/) Sign recognition
- [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) Has 1.7 million Turkish entries. A PDF dataset that can be great for pretraining, RAG benchmark curation.
- [ituperceptron/image-captioning-turkish](https://huggingface.co/datasets/ituperceptron/image-captioning-turkish) Image captioning dataset. 200k long, 100k short captions

#### Text & Speech
- [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) This dataset also has older versions v16,v15, etc.


### Live Leaderboards
- [malhajar/OpenLLMTurkishLeaderboard_v0.2](https://huggingface.co/spaces/malhajar/OpenLLMTurkishLeaderboard_v0.2)
- [KUIS-AI/Cetvel](https://huggingface.co/spaces/KUIS-AI/Cetvel)
- [kesimeg/Turkish-rewardbench](https://huggingface.co/spaces/kesimeg/Turkish-rewardbench) Reward model comparison
- [TurkBench/TurkBench](https://huggingface.co/spaces/TurkBench/TurkBench)
- [newmindai/Mezura](https://huggingface.co/spaces/newmindai/Mezura) Has RAG, Human evaluation (ELO score) and other benchmark scores. It also includes benchmarks in  malhajar/OpenLLMTurkishLeaderboard_v0.2 
- [newmindai/Mizan](https://huggingface.co/spaces/newmindai/Mizan) Embedding model leaderboard. Compares abilities of embedding models on tasks such as retrieval, clustering etc.

### Benchmark Datasets

#### Text only
- [AYueksel/TurkishMMLU](https://huggingface.co/datasets/AYueksel/TurkishMMLU)
- [alibayram/turkish_mmlu](https://huggingface.co/datasets/alibayram/turkish_mmlu)
- [ytu-ce-cosmos/gsm8k_tr](https://huggingface.co/datasets/ytu-ce-cosmos/gsm8k_tr)
- [Holmeister's Collections](https://huggingface.co/collections/Holmeister/turkish-llm-multi-prompt-evaluation-datasets-676994cd18391bb6e813bec3) A collection of 17 datasets for 11 different tasks (Truthfulness, fairness, summarization etc.). For more see the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425010437) 
- [CohereLabs/Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) MMLU for multiple languages including Turkish
- [mrlbenchmarks/global-piqa-nonparallel](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel) Cultural commonsense benchmark.
- [ytu-ce-cosmos/gpqa-extended_tr](https://huggingface.co/datasets/ytu-ce-cosmos/gpqa-extended_tr) Graduate level science questions.
- [CohereLabsCommunity/multilingual-reward-bench](https://huggingface.co/datasets/CohereLabsCommunity/multilingual-reward-bench) Reward benchmark (preference prediction)
- [boun-tabilab's Tabilab Collection](https://huggingface.co/collections/boun-tabilab/tabibench) A collection of benchmarks for classical NLP tasks (NER, summarization, classification, etc.)
- [AIM-Intelligence/XL-SafetyBench](https://huggingface.co/datasets/AIM-Intelligence/XL-SafetyBench) Jailbreak and cultural sensitivity benchmark

#### Visual and Text
- [CohereLabs/m-WildVision](https://huggingface.co/datasets/CohereLabs/m-WildVision)
- [CohereLabs/AyaVisionBench](https://huggingface.co/datasets/CohereLabs/AyaVisionBench)
- [kesimeg/MMStar_tr](https://huggingface.co/datasets/kesimeg/MMStar_tr)
- [metu-yks/yksbench](https://huggingface.co/datasets/metu-yks/yksbench) A visual benchmark based on university entrance exam. Questions include  visuals related to mathematics, geometry, physics, chemistry, biology, and geography
- [ytu-ce-cosmos/tubitak-science-olympiad-tr](https://huggingface.co/datasets/ytu-ce-cosmos/tubitak-science-olympiad-tr) A visual benchmark based on TUBITAK Science Olympiad questions. Images are screenshots of questions. 
- [Berkesule/Turkish-Visual-Reasoning-Dataset](https://huggingface.co/datasets/Berkesule/Turkish-Visual-Reasoning-Dataset) Visual question answering benchmark evaluating abstract reasoning
 

### Papers
- [Cosmos-LLaVA: Chatting with the Visual](https://arxiv.org/pdf/2412.02760)
- [Introducing cosmosGPT: Monolingual Training for Turkish Language Models](https://arxiv.org/pdf/2404.17336)
- [TurkishMMLU: Measuring Massive Multitask Language Understanding in Turkish](https://arxiv.org/abs/2407.12402)
- [TURSpider: A Turkish Text-to-SQL Dataset and LLM-Based Study](https://ieeexplore.ieee.org/document/10753591)
- [How do LLMs perform on Turkish? A multi-faceted multi-prompt evaluation](https://www.sciencedirect.com/science/article/abs/pii/S0957417425010437) Performances of various LLMs in Turkish
- [Evaluating the Quality of Benchmark Datasets for Low-Resource Languages: A Case Study on Turkish](https://arxiv.org/abs/2504.09714)
- [YKSBench: Stress-Testing Multimodal Models with Exam-Style Questions](https://openreview.net/pdf?id=qgAjoo3cJE) Paper of YKSBench benchmark.
- [TurkBench: A Benchmark for Evaluating Turkish Large Language Models](https://www.arxiv.org/pdf/2601.07020) Paper of TurkBench benchmark

### Tutorials and Codes
- [METU NLP Lab Git repo](https://github.com/metunlp)
- [wikipedia ToT data generation notebook](https://colab.research.google.com/drive/1mHOtErnLLoifkm0ySLc3_F9UqRa3SkSV?usp=sharing)


### Tools and APIs
- [Glosbe](https://tr.glosbe.com/)
- [Wiktionary](https://tr.wiktionary.org/wiki/Vikis%C3%B6zl%C3%BCk:Anasayfa)
- [Zemberek](https://github.com/ahmetaa/zemberek-nlp) Some turkish NLP tools
- [3rt4nm4n/turkish-apis](https://github.com/3rt4nm4n/turkish-apis) A list of turkish-apis

### MCPs
- [THY-MCP](https://mcp.turkishtechlab.com/)
- [borsa-mcp](https://github.com/saidsurucu/borsa-mcp) MCP Server for Istanbul Stock Exchange and Turkish Investment Fund Data 
- [yargi-mcp](https://github.com/saidsurucu/yargi-mcp) MCP Server For Turkish Legal Databases 
- [mevzuat-mcp](https://github.com/saidsurucu/mevzuat-mcp) MCP Server for Searching Turkish Legislation 
- [yoktez-mcp](https://github.com/saidsurucu/yoktez-mcp) MCP Server for Turkish Thesis Database 
- [yokatlas-mcp](https://github.com/saidsurucu/yokatlas-mcp) MCP Server for YOK Atlas

### State of AI in Türkiye
- [KUIS-AI Youtube channel](https://www.youtube.com/@kuisaicenter)
* [TR-AI Youtube channel](https://www.youtube.com/c/T%C3%BCrkiyeYapayZeka%C4%B0nisiyatifi)
- [Trendyol Tech Youtube channel](https://www.youtube.com/@TrendyolTech) Has videos related to their AI products and how they integrate AI

### Miscellaneous
- [Mukayese: Turkish NLP Strikes Back](https://mukayese.tdd.ai/#/)
- [Mukayese github repo](https://github.com/alisafaya/mukayese)
- [Wikipedia dumps](https://dumps.wikimedia.org/) Can be used as a dataset
- [Turkish Encoder-only Models List](https://huggingface.co/collections/atasoglu/turkish-encoder-only-models-65e2604fa0a2f649da9477e8) A collection of encoder only turkish models
- [Turkish Instruction Datasets List](https://huggingface.co/collections/atasoglu/turkish-instruction-datasets-6601d92fa6d901e554d98979) A collection of turkish instruction datasets
- [Turkish Vision-Language Datasets List](https://huggingface.co/collections/atasoglu/turkish-vision-language-datasets-66e7c563750d486e30732dd4) A collection of turkish vision language datasets
- [Cosmos App](https://play.google.com/store/apps/details?id=com.cosmos.cosmos&hl=tr-TR) The app of Cosmos AI Research group hosting their cosmos model. (Also has an iOS version)
- [ITU NLP Research Tools and Resources](https://ddi.itu.edu.tr/en/toolsandresources)

### Contributing
If you got anything to be added here just make a pull request! Before making a pull request please consider if a model/dataset/etc. has enough quality/uniqueness. Huggingface is crowded with finetuning of LLama and BERT, same applies to dataset. Many datasets have multiple machine translation version. This makes it hard to find good quality sources. We want to keep this list as curated as possible but still be able to cover enough sources.