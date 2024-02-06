# 任务内容

&emsp;&emsp;框架语义解析（Frame Semantic Parsing，FSP）是基于框架语义学的细粒度语义分析任务，其目标是从句中提取框架语义结构，实现对句子中事件或情境的深层理解。框架语义解析对阅读理解、文本摘要、关系抽取等下游任务具有重要意义。
&emsp;&emsp;然而，传统框架语义解析无法解释意义不能直接从其组成部分中预测出来的语言现象。如“爱买不买”，表示说话者对另一方是否要购买某物感到不在乎或是不感兴趣，传统方法以词为单位对该短语进行解析，以“爱”、“买”等动词作为目标词，激活喜欢、购买等场景，无法表达该短语的真实含义。
&emsp;&emsp;构式语法主张语言知识是由固定的、有意义的单位组成，这些单位被称为构式，既可以是简单的词或短语，也可以是复杂的句子或话语。由此，短语“爱买不买”是一个表达语义的整体，激活“情感反应”框架，样例如下所示：

<table align="center">
    <tr>
        <td><b>例句</b></td>
        <td colspan="2">社员来买东西只能看看样子，如果要求换换样，营业员就说：爱买不买，就这玩艺。</td>
    </tr>
    <tr>
        <td><b>目标词</b></td>
        <td colspan="2">爱…不…</td>
    </tr>
    <tr>
        <td><b>框架</b></td>
        <td colspan="2">情感反应</td>
    </tr>
    <tr>
        <td rowspan="3"><b>标注内容</b></td>
        <td><b>角色</b></td>
        <td><b>标注范围</b></td>
    </tr>
    <tr>
        <td>认知者</td>
        <td>营业员</td>
    </tr>
    <tr>
        <td>刺激物</td>
        <td>买</td>
    </tr>
</table>
<tfoot align="center">注：此处只展示以“爱…不…”进行框架语义解析的内容</tfoot>

&emsp;&emsp;为了提升框架语义解析能力，进一步实现对语言的深度理解，我们增加了以构式为“目标词”进行框架语义解析的数据，推出了第二届汉语框架语义解析评测。

&emsp;&emsp;本次评测任务为汉语框架语义解析（Chinese FSP，CFSP），该任务分为以下三个子任务：

1. 框架识别（Frame Identification）：识别句子中给定目标词或构式激活的框架。
2. 论元范围识别（Argument Identification）：识别句子中给定目标词或构式所支配论元的边界范围。
3. 论元角色识别（Role Identification）：预测论元范围识别任务中论元的语义角色标签。

&emsp;&emsp;本次评测设置了开放和封闭两个赛道，其中开放赛道的参赛队伍可以使用ChatGPT等大模型进行推理，但禁止对其进行微调，且需提交所使用的提示模板；封闭赛道中，参赛模型的参数量将会被限制。

# 评测数据

## 数据集规模

本次评测使用山西大学提供的CFN2.0数据集，包含标注例句及相关的框架信息。数据集基本信息如下：

| 数据集划分                        | Train       | Dev       | Test      | All         |
| --------------------------------- | ----------- | --------- | --------- | ----------- |
| 例句数                            | 11000(1000) | 2500(500) | 4500(500) | 18000(2000) |
| 框架数                            | 671(18)     | 354(9)    | 432(12)   | 695(20)     |
| 框架元素数                        | 947(25)     | 649(17)   | 711(21)   | 987(31)     |
| 词元数量                          | 2359        | 670       | 931       | 2596        |
| 注:括号内为面向构式的标注数据量。 |             |           |           |             |

## 数据样例及字段说明

1. 标注例句由json格式给出，数据包含以下字段：

> * sentence_id：例句id
> * cfn_spans：框架元素标注信息
>
>> * start：论元在句子中的起始位置
>> * end：论元在句中的结束位置
>> * fe_abbr：论元角色的框架元素缩写
>> * fe_name：论元角色的框架元素名称
>>
>
> * frame：例句所激活的框架名称
> * target：目标词的相关信息
>
>> * start：目标词在句中的起始位置
>> * end：目标词在句中的结束位置
>> * pos：目标词的词性
>>
>
> * text：标注例句
> * word：例句的分词结果及其词性信息

具体样例：

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
        {"start": 7, "end": 8, "pos": "v"},
        {"start": 9, "end": 9, "pos": "a"},
        {"start": 10, "end": 11, "pos": "n"},
        …
    ]
}]
```

2. 框架信息由json格式给出，数据包含以下字段：

> * frame_name：框架名称
> * frame_ename：框架英文名称
> * frame_def：框架定义
> * fes：框架元素信息
>
>> * fe_name：框架元素名称
>> * fe_abbr：框架元素缩写
>> * fe_ename：框架元素英文名称
>> * fe_def：框架元素定义
>>

具体样例：

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

# 评价标准

1. **框架识别**：以准确率（Accuracy）作为评价指标，具体定义如下：

$$
\rm{task1\_acc} = \rm{correct} / \rm{total}
$$

&emsp;&emsp;其中，correct为模型预测正确的数量，total为待识别框架总量。

2. **论元范围识别**：以F1作为评价指标，计算公式如下：

$$
\begin{array}{l}
{\rm{task2}}\_{\rm{f1}} = \frac{{{\rm{2*precision*recall}}}}{{{\rm{precision}} + {\rm{recall}}}}\\
{\rm{precision}} = \frac{{{\rm{InterSec(gold,pred)}}}}{{{\rm{Len(pred)}}}}\\
{\rm{recall}} = \frac{{{\rm{InterSec(gold,pred)}}}}{{{\rm{Len(gold)}}}}
\end{array}
$$

&emsp;&emsp;其中，$\rm{gold}$ 与 $\rm{pred}$ 分别表示真实span与预测span，$\rm{InterSec(gold,pred)}$ 表示二者共有的token数量， $\rm{Len(*)}$ 表示计算集合中的token数量。

3. **论元角色识别**: 以F1作为评价指标，计算公式如下:

$$
\begin{array}{l}
{\rm{task3}}\_{\rm{f1}} = \frac{{{\rm{2*precision*recall}}}}{{{\rm{precision}} + {\rm{recall}}}}\\
{\rm{precision}} = \frac{{{\rm{Count}}({\rm{gold}} \cap {\rm{pred}})}}{{{\rm{Count}}({\rm{pred}})}}\\
{\rm{recall}} = \frac{{{\rm{Count}}({\rm{gold}} \cap {\rm{pred}})}}{{{\rm{Count}}({\rm{gold}})}}
\end{array}
$$

&emsp;&emsp;其中，$\rm{gold}$ 与 $\rm{pred}$ 分别表示真实结果与预测结果的集合，$\rm{Count(*)}$ 表示集合中的元组数量。

4. **最终评测成绩**：以三个子任务分数的加权求和为最终结果，计算公式如下：

$$
{\rm{task\_score}}\;{\rm{ = }}\;{\rm{0}}.{\rm{4}}\;{\rm{*}}\;{\rm{task1\_acc}}\;{\rm{ + }}\;{\rm{0}}.{\rm{2}}\;{\rm{*task2\_f1 + 0}}.{\rm{4*task3\_f1}}
$$

# 评测赛程

具体赛程安排如下：

| 时间          | 事项                   |
| ------------- | ---------------------- |
| 3月1日-4月9日 | 开放报名               |
| 4月10日       | 训练集、验证集发布     |
| 5月13日       | 测试集发布             |
| 5月20日       | 测试集结果提交截止     |
| 5月31日       | 参赛队伍成绩及排名公布 |

# 组织者和联系人

评测组织者：李茹、谭红叶（山西大学）；常宝宝（北京大学）；戴新宇（南京大学）
任务负责人：闫智超（山西大学博士生，202312407023@email.sxu.edu.cn）
任务联系人：李俊材（山西大学博士生，202312407010@email.sxu.edu.cn）

# 任务奖项

本次评测为每个赛道评选出如下奖项：

1. 一等奖0-1名，奖励为笔记本电脑1台；
2. 二等奖0-1名，奖励为1200元；
3. 三等奖0-1名，奖励为800元。

# 赞助情况

笔记本电脑由百信信息技术有限公司董事长王宪朝提供赞助；
评测奖金由思腾合力（天津）科技有限公司高教负责人宋肖敏和太原市杰辉科技共同赞助。

# 任务网址

[https://github.com/SXUCFN/The-2nd-Chinese-Frame-Semantic-Parsing](https://github.com/SXUCFN/The-2nd-Chinese-Frame-Semantic-Parsing)

# 论文格式
会议投稿使用LaTeX模板。提交的论文最多包含 6 页正文，参考文献页数不限。由于本次会议采用双盲审稿，作者姓名和单位不能出现在投稿的论文中。因此，作者的自引不可采用“作者名字提出…”的方式，而是用“我们提出…”。不符合这些要求的论文将不经过完整的审稿流程而直接被拒稿。
