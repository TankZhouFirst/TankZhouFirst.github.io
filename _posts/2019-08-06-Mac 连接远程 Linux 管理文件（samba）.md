---
layout: post
title:  "Mac 连接远程 Linux 管理文件"
date:   2019-08-06 08:56:01 +0800
categories: Linux
tag: Mac
---


* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

**参考**：

- [Mac连接远程Linux管理文件（samba）](<https://www.jianshu.com/p/fe7fd0286c4e>)

****

## Linux 配置

### 安装 samba

```shell
sudo apt-get install samba
```

### 创建并共享文件夹

先创建一个需要共享的文件夹，这里用 `shared_directory`。如果已经有，直接执行 `chmod` 改变它的权限。

```shell
mkdir shared_directory
sudo chmod 777 shared_directory
```

### 配置 samba.conf

可以直接修改 `/etc/samba/smb.conf`，在文件末尾添加：

```shell
[share]
      path = path/to/shared_directory
      available = yes
      browseable = yes
      public = yes
      writable = yes
```

### 添加 samba 账户

```shell
sudo touch /etc/samba/smbpasswd
sudo smbpasswd -a USER_NAME
```

`USER_NAME` 就是你需要添加的用户名。然后会提示输入两次密码

## 在 Mac 上连接

打开 **Finder**（或在桌面），`CMD + k`，在弹出窗口，填写相关服务器信息，验证连接身份即可。

由于Finder自带的 [.DS_Store](https://link.jianshu.com/?t=https://en.wikipedia.org/wiki/.DS_Store) 包含了太多信息，如果在服务器产生 `.DS_Store` 会造成[安全隐患](https://link.jianshu.com/?t=http://www.wooyun.org/bugs/wooyun-2015-091869)。如果没有特殊配置，你用 `Finder` 管理远程的文件夹会自动产生 `.DS_Store`。

在云端检查你的共享文件夹，**如果发现.DS_Store，立即删除！**

如何让 `Finder` 不在远程连接时产生 `.DS_Store`？

打开 `Mac` 的 `Terminal`，输入

```shell
defaults write com.apple.desktopservices DSDontWriteNetworkStores true
```

然后重启 `Mac`，**再试试**远程连接。