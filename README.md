  

# 总体概述

* 框架语义解析（Frame Semantic Parsing，FSP）是自然语言处理领域中的一项重要任务，其目标是从句中提取框架语义结构<sup>[1]</sup>，实现对句子中涉及到的事件或情境的深层理解。FSP在阅读理解<sup>[2-3]</sup>、文本摘要<sup>[4-5]</sup>、关系抽取<sup>[6]</sup>等下游任务有着重要意义。

* 在自然语言中，大部分情况下以词为单位传达含义，但也存在很多词汇意义聚合现象，即组成的短语出现了新的含义。如:“爱买不买”，整个短语表示说话者对另一方是否要购买某物不在乎或不感兴趣。在框架语义分析中，该短语应该作为一个整体激活“情感反应”框架。如果以“爱”、“买”等单个动词作为目标词，激活喜欢、购买等框架，则无法捕捉到其独特的情感色彩。

* 构式语法主张语言是由固定的、有意义的单位组成，这些单位被称为构式<sup>[7-8]</sup>，既可以是简单的词或短语，也可以是复杂的句子或话语。如，“爱买不买”对应的构式是“爱V不V”，该构式是一个表达语义的整体，表示对某行为不在意或无所谓，应该整体作为目标词激起相应的框架。

* 为了提升框架语义解析能力，进一步实现对语言的深度理解，我们增加了以构式为“目标词”的框架语义解析数据，推出了第二届汉语框架语义解析评测。

* 本次评测设置了开放和封闭两个赛道，其中开放赛道的参赛队伍可以使用ChatGPT等大模型进行推理，但禁止对其进行微调，且需提交所使用的提示模板；封闭赛道中，参赛模型的参数量将会被限制。

# 任务介绍

汉语框架语义解析（Chinese FSP，CFSP）是基于汉语框架网(Chinese FrameNet, CFN)的语义解析任务，本次标注数据格式如下：

1. 标注数据的字段信息如下：
+ sentence_id：例句id
+ cfn_spans：框架元素标注信息
+ frame：例句所激活的框架名称
	+ target：目标词或构式的相关信息
	+ start：目标词或构式在句中的起始位置
	+ end：目标词或构式在句中的结束位置
	+ pos：对应分词的词性
+ text：标注例句
+ word：例句的分词结果及其词性信息

数据样例：

```json
[{
   "sentence_id": 2611,
   "cfn_spans":[
      { "start": 0, "end": 2, "fe_abbr": " time ", "fe_name": "时间" },
      { "start": 4, "end": 15, "fe_abbr": " ent ", "fe_name": "实体" },
      { "start": 17, "end": 26, "fe_abbr": " ini_v ", "fe_name": "初值" },
      { "start": 28, "end": 38, "fe_abbr": " fin_v ", "fe_name": "终值" }
   ],
   "frame": "量变",
   "target": [{ "start": 16, "end": 16, "pos": "p" },{ "start": 27, "end": 27, "pos": "p" }],
   "text": "近五年，新注册登记新能源汽车数量从2017年的65万辆到2021年的295万辆，呈高速增长态势。",
   "word": [
      {"start": 0, "end": 0, "pos": "a"},
      {"start": 1, "end": 1, "pos": "m"},
      {"start": 2, "end": 2, "pos": "q"},
      {"start": 3, "end": 3, "pos": "wp"},
      {"start": 4, "end": 4, "pos": "d"},
      {"start": 5, "end": 6, "pos": "v"},
      ...
   ]
}]
```

2. 框架信息在`frame_info.json`中，框架数据的字段信息如下：
+ frame_name：框架名称
+ frame_ename：框架英文名称
+ frame_def：框架定义
+ fes：框架元素信息
	+ fe_name：框架元素名称
	+ fe_abbr：框架元素缩写
	+ fe_ename：框架元素英文名称
	+ fe_def：框架元素定义

数据样例：

```json
[{
  "frame_name": "量变",
  "frame_ename": " Change_position_on_a_scale ",
  "frame_def": "该框架表示实体在某个维度上（即某属性）的相对位置发生变化，其属性值从初值变至终值。",
  "fes": [
      { "fe_name": "实体", "fe_abbr": "ent", "fe_ename": "Entity", "fe_def": "在某属性上具有一定量值的事物。" },
      { "fe_name": "属性", "fe_abbr": "attr", "fe_ename": "Attribute", "fe_def": "实体的有数量变化的属性" },
      { "fe_name": "初值", "fe_abbr": "ini_v", "fe_ename": "Initial_value ", "fe_def": "实体的属性值变化的起点。" },
      { "fe_name": "终值", "fe_abbr": "fin_v", "fe_ename": "Final_value", "fe_def": "实体最后达到的量值。" },
      { "fe_name": "初始状态", "fe_abbr": "i_state", "fe_ename": "Initial_state ", "fe_def": "实体经历属性值变化之前的状态。注意，初值仅仅表示某一值，而初始状态用一些描述性话语来表达，在表达中可能包含属性的值，也可能并没有一个明确的数值。" },
      { "fe_name": "终状态", "fe_abbr": "finis", "fe_ename": "Final_state", "fe_def": "实体经历属性值的变化之后所达到的状态。" },
      { "fe_name": "变幅", "fe_abbr": "diff", "fe_ename": "Difference", "fe_def": "实体在某维度上变动的幅度。" },
      { "fe_name": "值区间", "fe_abbr": "val_range", "fe_ename": "Value_range", "fe_def": "属性值的变动范围。" }
   ]
}]

```
 

本次评测共分为以下三个子任务：


* 子任务1: 框架识别（Frame Identification），识别句子中给定目标词激活的框架。
* 子任务2: 论元范围识别（Argument Identification），识别句子中给定目标词所支配论元的边界范围。
* 子任务3: 论元角色识别（Role Identification），预测子任务2所识别论元的语义角色标签。

  
  

## 子任务1: 框架识别（Frame Identification）

### 1. 任务描述

  

框架识别任务是框架语义学研究中的核心任务，其要求根据给定句子中目标词的上下文语境，为其寻找一个可以激活的框架。框架识别任务是自然语言处理中非常重要的任务之一，它可以帮助计算机更好地理解人类语言，并进一步实现语言处理的自动化和智能化。具体来说，框架识别任务可以帮助计算机识别出句子中的关键信息和语义框架，从而更好地理解句子的含义。这对于自然语言处理中的许多任务都是至关重要的。

  

### 2. 任务说明


该任务给定一个包含目标词或目标构式的句子，需要根据其语境识别出激活的框架，并给出识别出的框架名称。

1. 输入：句子相关信息（id和文本内容）及目标词或构式。
2. 输出：句子id及目标词所激活框架的识别结果，数据为json格式，<font  color="red">所有例句的识别结果需要放在同一list中</font>,样例如下：

```json
[
	[2611, "量变"],
	[2612, "等同"],
	...
]
```
  

### 3. 评测指标
框架识别采用正确率作为评价指标：

```math
$$task1\_acc = correct / total$$
```

其中，correct为模型预测正确的数量，total为待识别框架总量。

## 子任务2: 论元范围识别（Argument Identification）  

### 1. 任务描述

  

给定一条汉语句子及目标词，在目标词已知的条件下，从句子中自动识别出目标词所支配的语义角色的边界。该任务的主要目的是确定句子中每个目标词所涉及的论元（即框架元素）在句子中的位置。论元范围识别任务对于框架语义解析任务来说非常重要，因为正确识别谓词和论元的范围可以帮助系统更准确地识别论元的语义角色，并进一步分析句子的语义结构。

### 2. 任务说明


论元范围识别任务是指，在给定包含目标词的句子中，识别出目标词所支配的语义角色的边界。

1. 输入：句子相关信息（id和文本内容）及目标词。
2. 输出：句子id，及所识别出所有论元角色的范围，每组结果包含例句id：`task_id`, `span`起始位置, `span`结束位置，<font  color="red">每句包含的论元数量不定，识别出多个论元需要添加多个元组，所有例句识别出的结果共同放存在一个list中</font>，样例如下：
	
```json
[
	[ 2611, 0, 2 ],
	[ 2611, 4, 15],
	...
	[ 2612, 5, 7],
	...
]
```
 

### 3. 评测指标
  

论元范围识别采用P、R、F1作为评价指标：
```math
$${\rm{precision}} = \frac{{{\rm{InterSec(gold,pred)}}}}{{{\rm{Len(pred)}}}}$$
```
```math
$${\rm{recall}} = \frac{{{\rm{InterSec(gold,pred)}}}}{{{\rm{Len(gold)}}}}$$
```
```math
$${\rm{task2\_f1}} = \frac{{{\rm{2*precision*recall}}}}{{{\rm{precision}} + {\rm{recall}}}}$$
```
其中：gold 和 pred 分别表示真实结果与预测结果，InterSec(\*)表示计算二者共有的token数量， Len(\*)表示计算token数量。
  
## 子任务3: 论元角色识别（Role Identification）
  
### 1. 任务描述
框架语义解析任务中，论元角色识别任务是非常重要的一部分。该任务旨在确定句子中每个论元对应的框架元素，即每个论元在所属框架中的语义角色。例如，在“我昨天买了一本书”这个句子中，“我”是“商业购买”框架中的“买方”框架元素，“一本书”是“商品”框架元素。论元角色识别任务对于许多自然语言处理任务都是至关重要的，例如信息提取、关系抽取和机器翻译等。它可以帮助计算机更好地理解句子的含义，从而更准确地提取句子中的信息，进而帮助人们更好地理解文本。
  

### 2. 任务说明
  
论元角色识别任务是指，在给定包含目标词的句子中，识别出目标词所支配语义角色的边界及其角色名称。
框架及其框架元素的所属关系在`frame_info.json`文件中。

1. 输入：句子相关信息（id和文本内容）、目标词、框架信息以及论元角色范围。
2. 输出：句子id，及论元角色识别的结果，示例中“实体集”和“施动者”是“等同”框架中的框架元素。<font  color="red">注意所有例句识别出的结果应共同放存在一个list中</font>，样例如下：
```json
[
	[ 2611, 0, 2, "时间" ],
	[ 2611, 4, 15, "实体" ],
	...
	[ 2612, 5, 7, "时间" ],
	...
]
```

  

### 3. 评测指标

论元角色识别采用P、R、F1作为评价指标：
```math
$${\rm{precision}} = \frac{{{\rm{Count(gold \cap pred)}}}} {{{\rm{Count(pred)}}}}$$
```
```math
$${\rm{recall}} = \frac{{{\rm{Count(gold \cap pred)}}}} {{{\rm{Count(gold)}}}}$$
```
```math
$${\rm{task3\_f1}} = \frac{{{\rm{2*precision*recall}}}}{{{\rm{precision}} + {\rm{recall}}}}$$
```
其中，gold 和 pred 分别表示真实结果与预测结果， Count(\*) 表示计算集合元素的数量。
 
  
# 结果提交

本次评测结果在阿里天池平台上进行提交和排名。参赛队伍需要在阿里天池平台的“提交结果”界面提交预测结果，提交的压缩包命名为submit.zip，其中包含三个子任务的预测文件。

+ submit.zip
	+ A_task1_test.json
	+ A_task2_test.json
	+ A_task3_test.json
  

<font  color="red">1. 三个任务的提交结果需严格命名为A_task1_test.json、A_task2_test.json和A_task3_test.json。 2. 请严格使用`zip submit.zip A_task1_test.json A_task2_test.json A_task3_test.json` 进行压缩，即要求解压后的文件不能存在中间目录。</font> 选⼿可以只提交部分任务的结果，如只提交“框架识别”任务：`zip submit.zip A_task1_test.json`，未预测任务的分数默认为0。
 
# 系统排名

1. 所有评测任务均采用百分制分数显示，小数点后保留2位。
2. 系统排名取各项任务得分的加权和（三个子任务权重依次为 0.3，0.3，0.4），即： 
$${\rm{task\_score=0.3*task1\_acc+0.3*task2\_f1+0.4*task3\_f1}} $$
4. 如果某项任务未提交，默认分数为0，仍参与到系统最终得分的计算。
  

# Baseline
Baseline下载地址：[Github](https://github.com/SXUCFN/The-2nd-Chinese-Frame-Semantic-Parsing)
Baseline表现：

|task1_acc|task2_f1|task3_f1|task_score|
|---------|--------|--------|----------|
|70.16|84.79|58.51|70.19|
  
  

# 评测数据

数据由json格式给出，数据集包含以下内容：
  
+ CFN-train.json: 训练集标注数据。
+ CFN-dev.json: 验证集标注数据。
+ CFN-test-A.json: A榜测试集。
+ CFN-test-B.json: B榜测试集。B榜开赛前开放下载。
+ frame_info.json: 框架信息。
+ submit.zip：提交示例。
	+ A_task1_test.json：task1子任务提交示例。
	+ A_task2_test.json：task2子任务提交示例。
	+ A_task3_test.json：task3子任务提交示例。
 
# 数据集信息
  
* 数据集提供方：山西大学智能计算与中文信息处理教育部重点实验室，山西太原 030000。
* 负责人：谭红叶 tanhongye@sxu.edu.cn。
* 联系人：闫智超 202022408073@email.sxu.edu.cn、李俊材 202312407010@email.sxu.edu.cn。


# 赛程安排

本次大赛分为报名组队、A榜、B榜三个阶段，具体安排和要求如下：

1.  <font  color="red">报名时间：2024年3月22日-5月2日</font>
    
2.  训练、验证数据及baseline发布：2024年3月22日
    
3.  测试A榜数据发布：2024年3月23日
    
4.  <font  color="red">测试A榜评测截止：2024年5月4日 17:59:59</font>
    
5.  测试B榜数据发布：2024年5月6日
    
6.  <font  color="red">测试B榜最终测试结果：2024年5月8日 17:59:59</font>
    
7.  公布测试结果：2024年5月15日前
    
8.  提交中文或英文技术报告：2024年5月31日
    
9.  中文或英文技术报告反馈：2024年6月5日
    
10.  正式提交中英文评测论文：2024年6月10日
    
11.  公布获奖名单：2024年6月15日
    
12.  评测研讨会：2024年7月25日-7月28日


<font  color="red">**注意：报名组队与实名认证（2024年3月22日—5月2日）**</font>

# 报名方式

1.  4月1日阿里天池平台([https://tianchi.aliyun.com/](https://tianchi.aliyun.com/))将开放本次比赛的报名组队、登录比赛官网，完成个人信息注册，即可报名参赛；选手可以单人参赛，也可以组队参赛。组队参赛的每个团队不超过5人，每位选手只能加入一支队伍；选手需确保报名信息准确有效，组委会有权取消不符合条件队伍的参赛资格及奖励；选手报名、组队变更等操作截止时间为5月2日23：59：59；各队伍（包括队长及全体队伍成员）需要在5月2日23：59：59前完成实名认证（认证入口：天池官网-右上角个人中心-认证-支付宝实名认证），未完成认证的参赛团队将无法进行后续的比赛；
    
2.  向赛题举办方发送电子邮件进行报名，以获取数据解压密码。邮件标题为：“CCL2024-第二届汉语框架语义解析评测-参赛单位”，例如：“CCL2024-汉语框架语义解析评测-复旦大学”；附件内容为队伍的参赛报名表，报名表[点此下载](https://github.com/SXUCFN/The-2nd-Chinese-Frame-Semantic-Parsing/blob/main/%E7%AC%AC%E4%BA%8C%E5%B1%8A%E6%B1%89%E8%AF%AD%E6%A1%86%E6%9E%B6%E8%AF%AD%E4%B9%89%E8%A7%A3%E6%9E%90%E8%AF%84%E6%B5%8B%E6%8A%A5%E5%90%8D%E8%A1%A8.docx)，同时<font  color="red">报名表应更名为“参赛队名+参赛队长信息+参赛单位名称”</font>。请参加评测的队伍发送邮件至  [202312407010@email.sxu.edu.cn](mailto:202312407010@email.sxu.edu.cn)，报名成功后赛题数据解压密码会通过邮件发送给参赛选手，选手在天池平台下载数据即可。
    

# 赛事规则
  
1.  由于版权保护问题，CFN数据集只免费提供给用户用于非盈利性科学研究使用，参赛人员不得将数据用于任何商业用途。如果用于商业产品，请联系柴清华老师，联系邮箱  [charles@sxu.edu.cn](mailto:charles@sxu.edu.cn)。
2.  每名参赛选手只能参加一支队伍，一旦发现某选手以注册多个账号的方式参加多支队伍，将取消相关队伍的参赛资格。
3.  数据集的具体内容、范围、规模及格式以最终发布的真实数据集为准。验证集不可用于模型训练，针对测试集，参赛人员不允许执行任何人工标注。
4.  参赛队伍可在参赛期间随时上传测试集的预测结果，阿里天池平台A榜阶段每天可提交3次、B榜阶段每天可提交5次，系统会实时更新当前最新榜单排名情况，严禁参赛团队注册其它账号多次提交。
5.  允许使用公开的代码、工具、外部数据（从其他渠道获得的标注数据）等，但需要保证参赛结果可以复现。
6.  参赛队伍可以自行设计和调整模型，但需注意模型参数量最多不超过1.5倍BERT-Large（510M）。
7.  算法与系统的知识产权归参赛队伍所有。要求最终结果排名前10的队伍提供算法代码与系统报告（包括方法说明、数据处理、参考文献和使用的开源工具、外部数据等信息）。提交完毕将采用随机交叉检查的方法对各个队伍提交的模型进行检验，如果在排行榜上的结果无法复现，将取消获奖资格。
8.  参赛团队需保证提交作品的合规性，若出现下列或其他重大违规的情况，将取消参赛团队的参赛资格和成绩，获奖团队名单依次递补。重大违规情况如下：  
     a. 使用小号、串通、剽窃他人代码等涉嫌违规、作弊行为；  
     b. 团队提交的材料内容不完整，或提交任何虚假信息；  
     c. 参赛团队无法就作品疑义进行足够信服的解释说明；
9.  <font  color="red">获奖队伍必须注册会议并在线下参加（如遇特殊情况，可申请线上参加）</font>。

  
# 奖项信息
本次评测将评选出如下奖项。
由中国中文信息学会计算语言学专委会（CIPS-CL）为获奖队伍提供荣誉证书。
|奖项|一等奖|二等奖|三等奖|
|----|----|----|----|
|数量|0-2名|0-2名|0-2名|
|奖励合计|笔记本电脑2台|奖金2400元|奖金1600元|

# 赞助情况
笔记本电脑由百信信息技术有限公司提供赞助；  
评测奖金由思腾合力（天津）科技有限公司高教负责人宋肖敏和太原市杰辉科技共同赞助。


# 数据集协议


该数据集遵循协议： [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/?spm=5176.12282016.0.0.7a0a1517bGbbHL)。

  

由于版权保护问题，CFN数据集只免费提供给用户用于非盈利性科学研究使用，参赛人员不得将数据用于任何商业用途。如果用于商业产品，请联系柴清华老师，联系邮箱 charles@sxu.edu.cn。

  
  

# FAQ

  

* Q：比赛是否有技术交流群？

* A：请加钉钉群 73770004096 。

* Q：数据集解压密码是什么？

* A：请阅读“如何报名”，发送邮件报名成功后接收解压邮件。

* Q：状态返回失败，如何查看失败信息？

* A：鼠标点击模型列，查看具体报错信息。![enter image description here](https://img.alicdn.com/imgextra/i4/O1CN01QIluKC21aVcHROmwx_!!6000000007001-2-tps-750-217.png)

* Q：验证集可否用于模型训练？

* A：不可以。

  
  
  
  
  

# 参考文献

[1] Daniel Gildea and Daniel Jurafsky. 2002. Automatic labeling of semantic roles. Computational linguistics,28(3):245–288.

[2] Shaoru Guo, Ru Li*, Hongye Tan, Xiaoli Li, Yong Guan. A Frame-based Sentence Representation for Machine Reading Comprehension[C]. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistic (ACL), 2020: 891-896.

[3] Shaoru Guo, Yong Guan, Ru Li*, Xiaoli Li, Hongye Tan. Incorporating Syntax and Frame Semantics in Neural Network for Machine Reading Comprehension[C]. Proceedings of the 28th International Conference on Computational Linguistics (COLING), 2020: 2635-2641.

[4] Yong Guan, Shaoru Guo, Ru Li*, Xiaoli Li, and Hu Zhang. Integrating Semantic Scenario and Word Relations for Abstractive Sentence Summarization[C]. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP) 2021: 2522-2529.

[5] Yong Guan, Shaoru Guo, Ru Li*, Xiaoli Li, and Hongye Tan, 2021. Frame Semantic-Enhanced Sentence Modeling for Sentence-level Extractive Text Summarization[C]. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP) 2021: 404-4052.

[6] Hongyan Zhao, Ru Li*, Xiaoli Li, Hongye Tan. CFSRE: Context-aware based on frame-semantics for distantly supervised relation extraction[J]. Knowledge-Based Systems, 2020, 210: 106480.

[7] Willich, A. (2022) Introducing Construction Semantics (CxS): a frame-semantic extension of Construction Grammar and constructicography. Linguistics Vanguard, Vol. 8 (Issue 1), pp. 139-149.

[8] Boas, Hans Christian. “Construction Grammar and Frame Semantics.” The Routledge Handbook of Cognitive Linguistics (2021): n. pag.
