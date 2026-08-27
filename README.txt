编程智能体（coding agent）

一、Git 仓库地址
https://github.com/lyb6666/coding-agent

二、如何运行
1. 安装依赖：pip install -r requirements.txt
2. 配置密钥：复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY
3. 运行：python main.py "你的编程任务"；不带参数则进入交互式会话（连续提问，exit 退出）

三、特色功能
· 手写 agent 主循环，不依赖任何 agent 框架/SDK，仅用 OpenAI 兼容客户端调用模型原生 tool calling 接口
· 本地执行四类工具：读文件、写文件、执行 shell 命令、列目录
· 安全边界：危险命令（rm -rf /、git reset --hard 等）自动拒绝，所有路径限定在工作目录内
· 上下文管理：token 超限时自动裁剪最旧的工具调用轮次，始终保留任务目标
· 错误兜底：工具执行异常与模型调用失败均返回可读信息、自动重试，并有最大轮数保护

四、说明
· 默认模型 deepseek-chat，可通过环境变量 DEEPSEEK_MODEL 切换
· API key 仅经环境变量或未入库的 .env 提供，不会出现在仓库中
· 源码位于 coding_agent/ 包，agent 生成的文件统一在 workspace/ 目录
