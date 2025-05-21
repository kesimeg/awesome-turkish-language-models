# awesome-turkish-language-models [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of Turkish AI models, datasets, papers

The purpose of this repo to share and spread the information of Turkish AI models, datasets and papers. The amount of these Turkish resources are low and spread across the web. This repo aims to bring a curated selection of these resources together. This is not a list of all Turkish NLP/LLM models or datasets but a selection. So not all BERT or LLaMA based models are gonna make it here. The same applies to low quality Google translate translations of datasets. We aim each entry to have some kind of unique element to its own. This can be model performance, uniqueness in the task, highlighting the groups/companies (not everyone share their stuff so why not appreciate it!) etc. If you want to add anything you are welcomed :smirk: , please check out the contributing section.

## Table of Contents


* **[Models](#models)** 

* **[Datasets](#datasets)** 

* **[Papers](#papers)**  

* **[Benchmarks](#benchmarks)**  

* **[Tutorials and Codes](#tutorials-and-codes)**  

* **[Tools and APIs](#tools-and-apis)** 

* **[State of AI in Türkiye(Projects, products, groups etc.)](#state-of-ai-in-türkiye)** 

* **[Miscellaneous](#miscellaneous)**  

* **[Contributing](#contributing)**  


### Models

#### LLMs
1. [ytu-ce-cosmos/Turkish-Llama](https://huggingface.co/ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1)
2. [Trendyol/Llama-3-Trendyol-LLM-8b-chat-v2.0](https://huggingface.co/Trendyol/Llama-3-Trendyol-LLM-8b-chat-v2.0)
3. [Trendyol/Trendyol-LLM-7B-chat-v4.1.0](https://huggingface.co/Trendyol/Trendyol-LLM-7B-chat-v4.1.0)
4. [TURKCELL/Turkcell-LLM-7b-v1](https://huggingface.co/TURKCELL/Turkcell-LLM-7b-v1)
5. [KOCDIGITAL/Kocdigital-LLM-8b-v0.1](https://huggingface.co/KOCDIGITAL/Kocdigital-LLM-8b-v0.1)
6. [WiroAI/OpenR1-Qwen-7B-Turkish](https://huggingface.co/WiroAI/OpenR1-Qwen-7B-Turkish) Reasoning model
7. [WiroAI/wiroai-turkish-llm-9b](https://huggingface.co/WiroAI/wiroai-turkish-llm-9b)

#### VLMs
1. [ytu-ce-cosmos/Turkish-LLaVA](https://huggingface.co/ytu-ce-cosmos/Turkish-LLaVA-v0.1)

#### NLP
1. [Trendyol/tybert](https://huggingface.co/Trendyol/tybert)
2. [Trendyol/tyroberta](https://huggingface.co/Trendyol/tyroberta)
3. [ytu-ce-cosmos/turkish-base-bert-uncased](https://huggingface.co/ytu-ce-cosmos/turkish-base-bert-uncased)
4. [ytu-ce-cosmos/turkish-colbert](https://huggingface.co/ytu-ce-cosmos/turkish-colbert)
5. [ytu-ce-cosmos/turkish-gpt2-large](https://huggingface.co/ytu-ce-cosmos/turkish-gpt2-large)
6. [dbmdz/bert-base-turkish-128k-uncased](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased)
7. [TURKCELL/bert-offensive-lang-detection-tr](https://huggingface.co/TURKCELL/bert-offensive-lang-detection-tr)
8. [asafaya/kanarya-2b](https://huggingface.co/asafaya/kanarya-2b)
9. [boun-tabi-LMG/TURNA](https://huggingface.co/boun-tabi-LMG/TURNA)
10. [Helsinki-NLP group](https://huggingface.co/Helsinki-NLP) Lots of translation models for turkish
11. [VRLLab/TurkishBERTweet](https://huggingface.co/VRLLab/TurkishBERTweet) Tweet sentiment analysis
12. [akdeniz27/bert-base-turkish-cased-ner](https://huggingface.co/akdeniz27/bert-base-turkish-cased-ner)
13. [Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0](https://huggingface.co/Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0) Turkish and multilingual embeddings
14. [artiwise-ai/modernbert-base-tr-uncased](https://huggingface.co/artiwise-ai/modernbert-base-tr-uncased)

#### Speech models
To be added

#### Multi-modal models
1. [kesimeg/lora-turkish-clip](https://huggingface.co/kesimeg/lora-turkish-clip) CLIP model finetuned on turkish dataset


### Datasets

#### Text only
1. [merve/turkish_instructions](https://huggingface.co/datasets/merve/turkish_instructions) Instruction tuning dataset
2. [BrewInteractive/alpaca-tr](https://huggingface.co/datasets/BrewInteractive/alpaca-tr/viewer/default/train?p=2&views%5B%5D=train) Instruction tuning dataset
3. [AYueksel/TurkishMMLU](https://huggingface.co/datasets/AYueksel/TurkishMMLU)
4. [Metin/WikiRAG-TR](https://huggingface.co/datasets/Metin/WikiRAG-TR)
5. [MBZUAI/Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X/viewer/tr?views%5B%5D=tr)
6. [alibayram/turkish_mmlu](https://huggingface.co/datasets/alibayram/turkish_mmlu)
7. [Helsinki-NLP group](https://huggingface.co/Helsinki-NLP) Lots of translation models datasets for turkish
8. [ytu-ce-cosmos/gsm8k_tr](https://huggingface.co/datasets/ytu-ce-cosmos/gsm8k_tr)
9. [turkish-nlp-suite/turkish-wikiNER](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER)
10. [turkish-nlp-suite/InstrucTurca](https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca)
11. [WiroAI/dolphin-r1-turkish](https://huggingface.co/datasets/WiroAI/dolphin-r1-turkish) Reasoning dataset
12. [allenai/c4](https://huggingface.co/datasets/allenai/c4) Web scrape
13. [HPLT/HPLT2.0_cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned/viewer/tur_Latn) Web scrape
14. [unimelb-nlp/wikiann](https://huggingface.co/datasets/unimelb-nlp/wikiann) NER
15. [TUR2SQL](https://github.com/alibugra/TUR2SQL) Text to SQL query dataset
16. [dolphin-r1-turkish](https://huggingface.co/datasets/WiroAI/dolphin-r1-turkish) Reasoning dataset
17. [Holmeister's Collections](https://huggingface.co/collections/Holmeister/turkish-llm-multi-prompt-evaluation-datasets-676994cd18391bb6e813bec3) A collection of 17 datasets for 11 different tasks (Truthfulness, fairness, summarization etc.). For more see the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425010437) 
18. [CohereLabs/Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) MMLU for multiple languages including Turkish
19. [emre/ct_tree_of_thought_turkish](https://huggingface.co/datasets/emre/ct_tree_of_thought_turkish)
Turkish Tree of Thoughts (ToT) dataset 

#### Text & Images
1. [ytu-ce-cosmos/Turkish-LLaVA-Finetune](https://huggingface.co/datasets/ytu-ce-cosmos/Turkish-LLaVA-Finetune)
2. [ytu-ce-cosmos/Turkish-LLaVA-Pretrain](https://huggingface.co/datasets/ytu-ce-cosmos/Turkish-LLaVA-Pretrain)
3. [ytu-ce-cosmos/turkce-kitap](https://huggingface.co/datasets/ytu-ce-cosmos/turkce-kitap)
4. [99eren99/LLaVA1.5-Data-Turkish](https://huggingface.co/datasets/99eren99/LLaVA1.5-Data-Turkish)
5. [TasvirEt](https://www.kaggle.com/datasets/begum302553/tasviret-flickr8k-turkish)
6. [Cohere For AI](https://huggingface.co/CohereForAI) Has various dataset for VLM benchmarking
#### Text & Speech
1. [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) This dataset also has older versions v16,v15, etc.

### Papers
1. [Cosmos-LLaVA: Chatting with the Visual](https://arxiv.org/pdf/2412.02760)
2. [Introducing cosmosGPT: Monolingual Training for Turkish Language Models](https://arxiv.org/pdf/2404.17336)
3. [TurkishMMLU: Measuring Massive Multitask Language Understanding in Turkish](https://arxiv.org/abs/2407.12402)
4. [TURSpider: A Turkish Text-to-SQL Dataset and LLM-Based Study](https://ieeexplore.ieee.org/document/10753591)
5. [How do LLMs perform on Turkish? A multi-faceted multi-prompt evaluation](https://www.sciencedirect.com/science/article/abs/pii/S0957417425010437) Performances of various LLMs in Turkish
6. [Evaluating the Quality of Benchmark Datasets for Low-Resource Languages: A Case Study on Turkish](https://arxiv.org/abs/2504.09714)
### Benchmarks
1. [malhajar/OpenLLMTurkishLeaderboard_v0.2](https://huggingface.co/spaces/malhajar/OpenLLMTurkishLeaderboard_v0.2)
2. [KUIS-AI/Cetvel](https://huggingface.co/spaces/KUIS-AI/Cetvel)
3. [kesimeg/Turkish-rewardbench](https://huggingface.co/spaces/kesimeg/Turkish-rewardbench) Reward model comparison
### Tutorials and Codes
1. [METU NLP Lab Git repo](https://github.com/metunlp)
2. [wikipedia ToT data generation notebook](https://colab.research.google.com/drive/1mHOtErnLLoifkm0ySLc3_F9UqRa3SkSV?usp=sharing)

### Tools and APIs
1. [Glosbe](https://tr.glosbe.com/)
2. [Wiktionary](https://tr.wiktionary.org/wiki/Vikis%C3%B6zl%C3%BCk:Anasayfa)
3. [Zemberek](https://github.com/ahmetaa/zemberek-nlp) Some turkish NLP tools
4. [3rt4nm4n/turkish-apis](https://github.com/3rt4nm4n/turkish-apis) A list of turkish-apis

### State of AI in Türkiye
1. [KUIS-AI Youtube channel](https://www.youtube.com/@kuisaicenter)
2. [TR-AI Youtube channel](https://www.youtube.com/c/T%C3%BCrkiyeYapayZeka%C4%B0nisiyatifi)
3. [Trendyol Tech Youtube channel](https://www.youtube.com/@TrendyolTech) Has videos related to their AI products and how they integrate AI

### Miscellaneous
1. [Mukayese: Turkish NLP Strikes Back](https://mukayese.tdd.ai/#/)
2. [Mukayese github repo](https://github.com/alisafaya/mukayese)
3. [Wikipedia dumps](https://dumps.wikimedia.org/) Can be used as a dataset

### Contributing
If you got anything to be added here just make a pull request! Before making a pull request please consider if a model/dataset/etc. has enough quality/uniqueness. Huggingface is crowded with finetuning of LLama and BERT, same applies to dataset. Many datasets have multiple machine translation version. This makes it hard to find good quality sources. We want to keep this list as curated as possible but still be able to cover enough sources.

