---
layout:     post
title:      "Linux 入门（一）"
subtitle:   "Linux 基本命令"
date:       2016-05-21 12:20:52
author:     "Wei Zhang"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Linux

---

一些常用的 linux 命令，供初学者学习

## 1. 文件&目录管理 

### ls
>list   列出文件

```bash	
 ls -al                  
 ls -a
 ls -l
```

### cd
>change dictionary 改变目录

```bash	
cd /home
cd ~                    #  ~ 表示用户目录
```

### rm
>remove 删除

```bash	
rm test.txt
rm -r test              #  -r  recursive 递归删除 删除test目录下所有文件
rm -rf test             #  -f  force 强制递归删除
```

### cp
>copy 复制

```bash
cp test.txt /home/xxx/
cp test.txt /home/xxx/a.txt     #  复制并重命名
```	

### mv

>move 移动 重命名

```bash 
mv test.c /home/xxx/project     #  移动
mv test.c main.c                #  重命名
```

### ln

>link 创建链接

```bash
ln test /home/xxx/              #创建硬链接（删除原文件仍然存在），必须是同一分区
ln -s test /media/usbxxx        #软链接，相当于快捷方式，可以跨硬盘，跨网络，原文件删除后失效
```

### tar

>tar 归档&压缩

```bash
tar -xvf test.tar               #解压
tar -cvf test.tar ./video ./music  #压缩video music 到test.tar
```

>更多参数请看man tar

### mkdir
>make dictionary 创建目录

```bash	
mkdir test      
```	

### rmdir
>remove dictionary 删除空目录

```bash	
rmdir test
```	
### pwd
>print name of current/working directory 输出当前工作目录

```bash
pwd
```	

### find
>find 搜索文件

```bash
find / -name test               #在/目录下按名字搜索 test 文件，更多参数请看man find
```

### whereis 
>where is 搜索命令位置

```bash
whereis vim
```

## 2. 用户&权限

### who
>show who is logged on 显示登录用户

```bash
who
who am i            #   输出自己的用户名 等于 whoami
```

### adduser or useradd
>add a new user 创建用户

```bash
adduser test
```

### deluser or userdel
>delete a user  删除用户及相关文件

```bash
deluser test
```	

### passwd

>password 修改密码

```bash
sudo passwd test       #  passwd 用户名
```	

### login & logout 

>log in 登录
>log out 登出

```bash
login username
logout
```	

### chmod
>change file mode bits 改变用户权限

```bash
chmod 777 test.sh 
chmod a+x test.sh       #chmod参数参见鸟哥档案权限部分
```

>P.S. 权限代码：
>用户
>>a all 所有用户
>>u owner 拥有者
>>g group 用户组
>>o other 其他用户
>权限
>>r read 读
>>w write 写
>>x execute 执行

### chgrp
>change group 改变文件用户组

```bash
chgrp xxx test.sh      #chgrp 用户组 文件名
```

### chown
>change owner 改变拥有者

```bash
chown username test.sh  #chown 用户名 文件名
```	

### sudo

>SuperUser DO 以root权限执行一条命令

```bash
sudo command            #不用多说了吧?
```

>P.S. 如果提示not a sudoer,在root用户下输入visudo,在

```bash
root    ALL=(ALL:ALL) ALL
```

下添加一行，除了root改为自己的用户名，其他不变

### su
>SuperUser 超级用户

```bash
su                      #这个也不需要解释吧？
```	

>P.S.安装完系统root默认没有密码，不能登录，需要用

```bash
sudo passwd root
```

>   修改密码


