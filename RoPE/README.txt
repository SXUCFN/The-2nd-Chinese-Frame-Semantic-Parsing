需要从 https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main 下载文件 pytorch_model.bin 放置在文件夹 chinese_bert_wwm_ext中.


用python3先后依次运行训练文件train_task1.py, train_task2.py, train_task3.py得到训练参数，参数会保存在saves文件夹中.

得到参数后先后依次运行B轮的预测文件predict_task1, predict_task2, predict_task3即可得到结果A_task1_test, A_task2_test, A_task3_test于文件夹dataset中.
（注意必须先预测task2任务得到B_task2_test，再预测task3任务！）

提交的结果在文件夹SOTA中.