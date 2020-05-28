---
layout: post
title:  "Linux 自定义 service"
date:   2020-05-27 23:57:01 +0800
categories: 环境配置
tag: 服务器配置
---

* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

**参考**

- [How To Setup Autorun a Python Script Using Systemd](https://tecadmin.net/setup-autorun-python-script-using-systemd/)
- [CentOS7 下手动配置服务，以指定用户启动进程](https://blog.csdn.net/hemowolf/article/details/77197085)

****

### 需求

需要将一段脚本（**Python** 或 **shell** 等）变成服务，并设置自启等等。

### Python 脚本

```python
#!/usr/bin/python3

from time import time, sleep

f = open("/home/pi/log.txt", 'w')
count = 0

while count < 20:
    stp = str(int(time()))
    print(stp)
    f.write(stp + '\n')
    sleep(1)
    count += 1

f.close()
```

### 创建服务

```shell
sudo vim /lib/systemd/system/dummy.service
```

内容如下：

```python
[Unit]
Description=Dummy Service
After=multi-user.target
Conflicts=getty@tty1.service

[Service]
User=pi
Group=pi

Type=idle
ExecStart=/usr/bin/python3 /home/pi/Desktop/myservice.py
WorkingDirectory=/home/pi/em/smart_cam
Restart=always
RestartSec=5
StartLimitInterval=0
StandardInput=tty-force

[Install]
WantedBy=multi-user.target
```

> - **Description** 为服务名，用于系统进程监听等等
> - **After** 表示在制定环境起来之后运行该服务
> - **User** 和 **Group** 表示以什么用户执行程序。默认为 **root**，此时可能会导致服务中创建的文件没有写权限。
> - **ExecStart** 指定要执行的程序或脚本，使用绝对路径
> - **WorkingDirectory**：设置工作路径，即程序运行的路径。例如，python 程序中，需要创建相对路径，那对应的根路径即通过这里设定。
> - **idle** 表示在其他东西加载完成之后运行，默认为 simple
> - **Restart=always** : 只要不是通过 **systemctl stop** 来停止服务，任何情况下都必须要重启服务，默认值为 `no`
> - **RestartSec=5** : 重启间隔，比如某次异常后，等待 `5(s)` 再进行启动，默认值 `0.1(s)`
> - **StartLimitInterval** : 无限次重启，默认是 `10` 秒内如果重启超过 `5` 次则不再重启，设置为 `0` 表示不限次数重启
>
> **需要使用绝对路径！！！**

修改文件权限：

```shell
sudo chmod 644 /lib/systemd/system/dummy.service
```

### 开启服务

```shell
sudo systemctl daemon-reload         # 重载守护进程
sudo systemctl enable dummy.service  # 添加服务
sudo systemctl start dummy.service   # 启动服务
```

### 服务管理

```shell
sudo systemctl daemon-reload         # 重载守护进程
sudo systemctl enable dummy.service  # 添加服务
sudo systemctl disable dummy.service # 移除服务
sudo systemctl start dummy.service   # 启动服务
sudo systemctl stop dummy.service    # 关闭服务
sudo systemctl restart dummy.service # 重启服务
sudo systemctl status dummy.service  # 查看状态
```
