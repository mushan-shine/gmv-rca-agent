"""text-to-SQL 循环:生成 → 守卫 → 执行 → 评估 → 改进。

模块边界刻意与原简报 §5 一致 —— **LLM 只做它无可替代的那一步**:

* :mod:`rca.nl2sql.generate` —— LLM 把自然语言变成 SQL(唯一用到 LLM 的地方)
* :mod:`rca.nl2sql.guard`    —— 确定性静态校验:只读、表/列白名单、幻觉检测
* :mod:`rca.nl2sql.cases`    —— 程序化生成的标准答案(不手工标注)
* :mod:`rca.nl2sql.compare`  —— 执行式比对:比结果集,不比 SQL 文本
* :mod:`rca.nl2sql.loop`     —— L1 自修复内循环(把真实错误喂回去重生成)
* :mod:`rca.nl2sql.state`    —— Delta 状态表:下一轮读上一轮写下的东西
* :mod:`rca.nl2sql.evaluate` —— 跑评估集、算指标、写 eval_runs
"""
