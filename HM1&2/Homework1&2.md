## Notes1

<img title="" src="image.png" alt="74026b7a-fcfd-43e8-94ee-867714e017d1" data-align="inline"> 

- 专用模型：针对特定任务的模型

- 通用大模型： 模型可以应对多种任务，多种模态

- 典型的数学评测数据集：Math 和GSM8K

### 从模型到应用需要进行哪些环节：

<img title="" src="img2.png" alt="807eb008-321b-4545-aa04-d816df436d63" style="zoom:67%;">

### 书生浦语全链条开发体系：

<img title="" src="img3.png" alt="953ea6e9-6b1f-4357-9878-90138e92869c" style="zoom:67%;">

- 预训练模型：InternLM

- 微调框架： Xtuner

- 部署框架：LMDeploy

- 评测：OpenCompass

- 应用：Lagent， agentLego

## Notes2 - 部署 `InternLM2-Chat-1.8B` 模型进行智能对话

#### 创建基本环境

```shell
# 安装pytorch等其他必要的包
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia


pip install huggingface-hub==0.17.3
pip install transformers==4.34 
pip install psutil==5.9.8
pip install accelerate==0.24.1
pip install streamlit==1.32.2 
pip install matplotlib==3.8.3 
pip install modelscope==1.9.5
pip install sentencepiece==0.1.99
```

![](img1.png)

#### 下载InternLM2-Chat-1.8B模型

```python
import os
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')

```

#### 创建书生浦语 智能会话

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_
8b"
# 从路径中加载预训练的模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

#提示语
system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    # 用户输入框
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)


```

## Homework 2

#### 效果如下

![loading-ag-86](300个字的小故事效果图.png)



## Notes2 - 部署实战营优秀作品 `八戒-Chat-1.8B` 模型

>  `八戒-Chat-1.8B` 是利用《西游记》剧本中所有关于猪八戒的台词和语句以及 LLM API 生成的相关数据结果，进行**全量微调**得到的猪八戒聊天模型。

### 配置基本环境

- 继续使用上面的demo环境

- 获取仓库中demo文件
  
  ```shell
  cd /root/
  git clone https://gitee.com/InternLM/Tutorial -b camp2
  # git clone https://github.com/InternLM/Tutorial -b camp2
  cd /root/Tutorial
  ```

### 运行bajie_download.py文件下载`八戒-Chat-1.8B` 模型

```shell
python /root/Tutorial/helloworld/bajie_download.py
```

### 输入运行命令

```shell
streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006
```

#### streamlit知识点

> Streamlit是一个开源的python库，可以轻松的**将python脚本文件转换成可交互的web应用程序**

   1. 使用streamli创建web应用程序的基本流程

- 编写一个python脚本，使用Streamlit的API添加想要的交互元素和逻辑。

- 使用 `streamlit run script.py`命令运行脚本

- 访问http://localhost:8501(默认端口)查看定义的web应用
2. 上述命令的参数含义
   
   - `streamlit run`：**这是启动 Streamlit 应用的命令**
   
   - `/root/Tutorial/helloworld/bajie_chat.py`：**这是要运行的 Python 脚本的路径**，**脚本应该包含 Streamlit 应用程序的代码**
   
   - `--server.address 127.0.0.1`：**这个参数设置 Streamlit 服务器监听的 IP 地址**。在这个例子中，它被设置为 `127.0.0.1`，也就是 localhost，**意味着服务器只能在运行它的同一台机器上访问**。如果想开放让外部网络访问，address可以设置0.0.0.0，但也会带来安全风险
   
   - `--server.port 80` **设置应用监听的端口为 80**，这是 HTTP 默认端口。你可以选择任何未被占用的端口。

3. 学习以下八戒chat的streamlit 代码示例

### 在本地服务器上远程访问八戒-chat的web服务

```shell
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374
```

> 这个命令通常用于：
> 
> - **从本地机器访问远程服务器上的服务**，如数据库，web服务等
> 
> - 目的在于创建一个安全通道，使得本地程序可以通过这个安全通道与远程的应用程序建立链接
> 
> - 本地访问远程的jupyterlab 或者 tensorboard都可以这样使用

命令解析：

- 建立一个SSH连接到远程服务器`ssh.intern-ai.org.cn`，使用端口`38374`，以root用户身份登录。

- 启用压缩，不执行远程命令，允许远程主机链接到本地端口的转发

- 设置本地端口转发规则。
  
  - `ssh`：这是启动SSH客户端的命令
  
  - -CNg：
    
    - -C： 启动压缩。加快数据传输
    
    - -N：不执行远程命令，仅建立隧道。如果想要执行命令，可以ssh后面直接跟命令
    
    - -g： 允许远程主机连接到本地端口的转发
  
  - -L `6006:127.0.0.1:6006`：这是本地端口转发的规则设置。
    
    - -L：指定本地端口转发。
      
      格式为**本地端口:远程主机:远程端口**。
  
  - `root@ssh.intern-ai.org.cn`：指定要连接的远程服务器的用户和主机名
  
  - `-p 38374`：指定**远程服务器上的端口号**，SSH默认端口是22，这里指定了一个不同的端口号38374

## homework2

![b8cb8554-acfc-4f1e-940a-a781255e6a85](file:///C:/Users/cathy/Pictures/Typedown/b8cb8554-acfc-4f1e-940a-a781255e6a85.png)




