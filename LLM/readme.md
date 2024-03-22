# 环境依赖
 * 需要先安装依赖环境：
```bash
pip install "openai>=0.28.0"
```
 * 按照个人情况，修改`main.py`文件中的代理配置和api_key

# 数据准备
将数据放于`data`目录下

# 运行
 * 框架识别任务：
```bash
python main.py FI
```
 * 论元范围识别任务：
```bash
python main.py AI
```
 * 论元劫色识别任务：
```bash
python main.py RI
```