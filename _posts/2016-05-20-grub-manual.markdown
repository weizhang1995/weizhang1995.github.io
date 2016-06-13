---
layout:     post
title:      "Grub 手动引导"
subtitle:   "如何在 Grub 引导失败的时候拯救你的系统"
date:       2016-05-20 12:20:52
author:     "Wei Zhang"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Linux
    - Grub

---

在 linux 的初学过程中，我们可能会遇到 grub 引导失败的情况，很多人遇到这种情况的时候只能无奈的选择重装，其实如果 boot 分区还在的情况下，可以手动引导恢复。

当分区表改变或/root分区被删除时开机会出现

```bash
grub-rescue >
```

## 1. 如果/boot及根目录所在分区完好

### step 1

>此时如果 /boot 分区还在，并且知道所在分区，如 ```(hd0,msdos5)``` ,则

```bash
grub-rescue > root=(hd0,msdos5)
grub-rescue > prefix=/boot/grub   #有的发行版是/boot/grub2 可以用ls /boot 查看
grub-rescue > insmod /boot/grub/normal.mod      #加载normal模块
grub-rescue > normal
```

	
>之后就会进入 normal 模式：

```bash
grub > 
```

### step 2    
>手动引导 linux

```bash
grub > root=(hd0,msdos5)
grub > linux /boot/vmlinuz*  ro root=/dev/sda5       #星号部分按tab补全
grub > initrd /boot/initrd*                          #星号部分按tab补全
grub > boot
```
	
>然后就可以进入系统了，进入终端：

```bash
sudo update-grub2
sudo grub-install /dev/sda
```

## 2. 如果/boot或根目录被删除

情况一：如果根目录被删除，```/boot```分区还在

>先按 step 1，进入 normal 模式

```bash
grub > rootnoverify (hd0,msdos1) 
grub > chainloader +1
grub > boot
```

情况二：如果```/boot```被删除了......重装吧～～

P.S.如果不知道在/boot在哪个分区，网上说用find，但我用过提示没有find命令，但可以一个一个分区试：

```bash
root=(hd0,msdosx）
ls /boot
```

看看有没有grub目录