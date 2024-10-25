# Awesome-LLM-Causal-Reasoning


<p align="center">
  <img src='https://raw.githubusercontent.com/chendl02/Awesome-LLM-Causal-Reasoning/main/misc/intro.jpg' width=500>
</p>


🔥🔥🔥 **Improving Causal Reasoning in Large Language Models: A Survey** **[[Paper]](https://arxiv.org/abs/2410.16676)**



We provide a comprehensive review of research aimed at enhancing LLMs for causal reasoning (CR). We categorize existing methods based on the role of LLMs: either as *<u>reasoning engines</u>* or as *<u>helpers</u>* providing knowledge or data to traditional CR methods, followed by a detailed discussion of the methodologies in each category. We then evaluate the performance of LLMs on various causal reasoning tasks, providing key findings and in-depth analysis. Finally, we provide insights from current studies and highlight promising directions for future research.





 

---

<font size=5><center><b> Table of Contents </b> </center></font>
- [LLMs as Reasoning Engines](#llms-as-reasoning-engines)
- [LLMs as Helper](#llms-as-helper)
- [Datasets](#datasets)
  - [Causality Discovery](#causality-discovery)
  - [Causal Inference](#causal-inference)
  - [Additional Causal Tasks](#additional-causal-tasks)
  
---



## LLMs as Reasoning Engines

<p align="center">
  <img src='https://raw.githubusercontent.com/chendl02/Awesome-LLM-Causal-Reasoning/main/misc/engine.jpg' width=1000>
</p>

**[C2P: Featuring Large Language Models with Causal Reasoning](https://arxiv.org/abs/2407.18069)**

*Abdolmahdi Bagheri, Matin Alinejad, Kevin Bello, Alireza Akhondi-Asl.* Preprint'24

**[Large Language Model Cascades with Mixture of Thoughts Representations for Cost-efficient Reasoning.](https://arxiv.org/abs/2310.03094)** 

*Murong Yue, Jie Zhao, Min Zhang, Liang Du, Ziyu Yao.*  ICLR'2024

**[Large Language Model for Causal Decision Making.](https://arxiv.org/abs/2312.17122)** 

*Jiang, Haitao, Lin Ge, Yuhe Gao, Jianian Wang, and Rui Song.*  COLM'2024



**[Towards CausalGPT: A Multi-Agent Approach for Faithful Knowledge Reasoning via Promoting Causal Consistency in LLMs](https://arxiv.org/abs/2308.11914)**

*Ziyi Tang, Ruilin Wang, Weixing Chen, Keze Wang, Yang Liu, Tianshui Chen, Liang Lin.* Preprint'2023

**[CLadder: Assessing Causal Reasoning in Language Models](https://arxiv.org/abs/2312.04350)**

*Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojasv Kamal, Zhiheng Lyu, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, Bernhard Schölkopf.* NeurIPS'2023


**[Causal Reasoning of Entities and Events in Procedural Texts](https://aclanthology.org/2023.findings-eacl.31/)**

*Li Zhang, Hainiu Xu, Yue Yang, Shuyan Zhou, Weiqiu You, Manni Arora, Chris Callison-Burch.* ACL'2023

**[Preserving Commonsense Knowledge from Pre-trained Language Models via Causal Inference](https://aclanthology.org/2023.acl-long.509/)**

*Junhao Zheng, Qianli Ma, Shengjie Qiu, Yue Wu, Peitian Ma, Junlong Liu, Huawen Feng, Xichen Shang, Haibin Chen.* ACL'23


**[Answering Causal Questions with Augmented LLMs](https://openreview.net/pdf?id=ikLvibXZid)**

*Nick Pawlowski, James Vaughan, Joel Jennings, Cheng Zhang.* ICML Workshop'2023

**[Neuro-Symbolic Procedural Planning with Commonsense Prompting](https://arxiv.org/abs/2206.02928)**

*Yujie Lu, Weixi Feng, Wanrong Zhu, Wenda Xu, Xin Eric Wang, Miguel Eckstein, William Yang Wang.* ICLR'2023

**[Faithful Reasoning Using Large Language Models.](https://arxiv.org/abs/2208.14271v1)** 

*Antonia Creswell, Murray Shanahan.*  Preprint'2022

**[Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning.](https://arxiv.org/abs/2205.09712)** 

*Antonia Creswell, Murray Shanahan, Irina Higgins.*  Preprint'2022


**[CausalBERT: Injecting Causal Knowledge Into Pre-trained Models with Minimal Supervision.](https://arxiv.org/abs/2107.09852)** 

*Zhongyang Li, Xiao Ding, Kuo Liao, Bing Qin, Ting Liu.*  Preprint'2021





## LLMs as Helper
<p align="center">
  <img src='https://raw.githubusercontent.com/chendl02/Awesome-LLM-Causal-Reasoning/main/misc/helper.jpg' width=500>
</p>

**[LLM-Enhanced Causal Discovery in Temporal Domain from Interventional Data](https://arxiv.org/abs/2404.14786v1)**

*Peiwen Li, Xin Wang, Zeyang Zhang, Yuan Meng, Fang Shen, Yue Li, Jialong Wang, Yang Li, Wenweu Zhu.* Preprint'2024

**[Faithful Explanations of Black-box NLP Models Using LLM-generated Counterfactuals](https://openreview.net/pdf?id=UMfcdRIotC)**

*Yair Ori Gat, Nitay Calderon, Amir Feder, Alexander Chapanin, Amit Sharma, Roi Reichart.* ICLR'2024

**[Causal Structure Learning Supervised by Large Language Model](https://arxiv.org/abs/2311.11689v1)**

*Taiyu Ban, Lyuzhou Chen, Derui Lyu, Xiangyu Wang, Huanhuan Chen.* Preprint'2023

**[Neuro-Symbolic Integration Brings Causal and Reliable Reasoning Proofs](https://arxiv.org/abs/2311.09802v2)**

*Sen Yang, Xin Li, Leyang Cui, Lidong Bing, Wai Lam.* Preprint'2023

**[Extracting Self-Consistent Causal Insights from Users Feedback with LLMs and In-context Learning](https://arxiv.org/abs/2312.06820v1)**

*Sara Abdali, Anjali Parikh, Steve Lim, Emre Kiciman.* Preprint'2023

**[Improving Commonsense Causal Reasoning by Adversarial Training and Data Augmentation](https://arxiv.org/abs/2101.04966)**

*Ieva Staliūnaitė, Philip John Gorinski, Ignacio Iacobacci.* Preprint'2021


# Datasets
<p align="center">
  <img src='https://raw.githubusercontent.com/chendl02/Awesome-LLM-Causal-Reasoning/main/misc/RadarChart.png' width=400>
</p>

We first categorize the end tasks into three groups: causal discovery, causal inference, and additional causal tasks. 
For each category, we evaluate recent LLMs using pass@1 accuracy with strategies such as zero-shot, few-shot, direct I/O prompting, and Chain-of-Thought (CoT) reasoning.

## Causality Discovery

**[Can large language models infer causation from correlation](https://arxiv.org/abs/2306.05836)**

*Zhijing Jin, Jiarui Liu, Zhiheng Lyu, Spencer Poff, Mrinmaya Sachan, Rada Mihalcea, Mona Diab, Bernhard Schölkopf.* ICLR'2024


**[CausalQA: A Benchmark for Causal Question Answering](https://aclanthology.org/2022.coling-1.291.pdf)**

*Alexander Bondarenko, Magdalena Wolska, Stefan Heindorf, Lukas Blübaum, Axel-Cyrille Ngonga Ngomo, Benno Stein, Pavel Braslavski, Matthias Hagen, Martin Potthast.* ACL'2022

**[e-CARE: a New Dataset for Exploring Explainable Causal Reasoning](https://aclanthology.org/2022.acl-long.33.pdf)**

* Li Du, Xiao Ding, Kai Xiong, Ting Liu, and Bing Qin.* ACL'2022

**[CausaLM: Causal Model Explanation Through Counterfactual Language Models](https://aclanthology.org/2021.cl-2.13.pdf)**

* Amir Feder, Nadav Oved, Uri Shalit, Roi Reichart.* ACL'2021

## Causal Inference

**[CRAB:Assessing the Strength of Causal Relationships Between Real-World Events](https://aclanthology.org/2023.emnlp-main.940.pdf)**

*Angelika Romanou, Syrielle Montariol, Debjit Paul, Léo Laugier, Karl Aberer, Antoine Bosselut.* EMNLP'2023

**[CLadder: Assessing Causal Reasoning in Language Models](https://arxiv.org/abs/2312.04350)**

*Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojasv Kamal, Zhiheng Lyu, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, Bernhard Schölkopf.* NeurIPS'2023

**[COLA: Contextualized Commonsense Causal Reasoning from the Causal Inference Perspective](https://aclanthology.org/2023.acl-long.288.pdf)**

*Zhaowei Wang, Quyet V. Do, Hongming Zhang, Jiayao Zhang, Weiqi Wang, Tianqing Fang, Yangqiu Song, Ginny Wong, Simon See.* ACL'2023

**[Abductive Commonsense Reasoning](https://arxiv.org/abs/1908.05739)**

*Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Scott Wen-tau Yih, Yejin Choi.* ICLR'2020



## Additional Causal Tasks

**[TRAM: Benchmarking Temporal Reasoning for Large Language Models](https://aclanthology.org/2024.findings-acl.382.pdf)**

*Yuqing Wang, Yun Zhao.* ACL'2024



**[MoCa: Measuring Human-Language Model Alignment on Causal and Moral Judgment Tasks](https://arxiv.org/abs/2310.19677)**

*Allen Nie, Yuhui Zhang, Atharva Amdekar, Chris Piech, Tatsunori Hashimoto, Tobias Gerstenberg.* NeurIPS'2023


**[CRASS: A Novel Data Set and Benchmark to Test Counterfactual Reasoning of Large Language Models](https://aclanthology.org/2022.lrec-1.229.pdf)**

*Jörg Frohberg, Frank Binder.* LREC'2022



# Citation
```
@article{xiong2024improving,
  title={Improving Causal Reasoning in Large Language Models: A Survey},
  author={Xiong, Siheng and Chen, Delin and Wu, Qingyang and Yu, Longxuan and Liu, Qingzhen and Li, Dawei and Chen, Zhikai and Liu, Xiaoze and Pan, Liangming},
  journal={arXiv preprint arXiv:2410.16676},
  year={2024}
}
```
