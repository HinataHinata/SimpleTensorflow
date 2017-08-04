# 没有完成的任务
01. slides_01 P43 分布到不同的CUP／GUP／Device 上
02. 一些pycharm的快捷键
    * command + B 跳到源码
    * command + option + L 格式化代码
    * command + shift + O 在代码中搜索
    * command + J 插入代码段
    * 前端网页选中之后按 Tab 可以缩进
    * 标签栏右键是支持分栏的
03. PyCharm 创建代码模板
    * perference -> Editor -> File and Code Template 选中 Python Script
    * 模板变量 '$ 符号+变量'
        * ${PROJECT_NAME} - 当前Project名称;
        * ${NAME} - 在创建文件的对话框中指定的文件名;
        * ${USER} - 当前用户名;
        * ${DATE} - 当前系统日期;
        * ${TIME} - 当前系统时间;
        * ${YEAR} - 年;
        * ${MONTH} - 月;
        * ${DAY} - 日;
        * ${HOUR} - 小时;
        * ${MINUTE} - 分钟；
        * ${PRODUCT_NAME} - 创建文件的IDE名称;
        * ${MONTH_NAME_SHORT} - 英文月份缩写, 如: Jan, Feb, etc;
        * ${MONTH_NAME_FULL} - 英文月份全称, 如: January, February, etc；
03. 单词预测 
    通过分析维基百科的一些词条作出分析，判断单词之间的距离。
    
    最后给每个单词一个位置，就可以计算任意两个单词之间的距离。
    
    可以考虑回归一片文章，将点连成图，分析图 或者分析文章中前后单词之间的距离组成的数组。
    
    复杂一点的考虑是通过前两个单词来计算下一个单词
    
    或者通过前几个单词计算下一个单词。这样就有点复杂了，据说会比较发散。
    
    