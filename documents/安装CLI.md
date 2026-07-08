## 安装CLI

​	命令行界面

​	smallstep用于证书，密钥，身份，加密操作

```shell
winget install Smallstep.step       
```

把电脑连接到一个远程证书颁发机构CA，提供你的指纹fingerprint验证服务器真实可信，安装根证书，机器会信任这个机构的所有证书

```shell
step ca bootstrap --ca-url https://polo.cs.nott.ac.uk --fingerprint c6b629879d73e34575c17adb771f708ee3c515353e4153717aefdb378b7a8e9e
```

检查ssh-agent是否在运行

```
step ssh list
```

启动系统自带的ssh-agent

```shell
Get-Service ssh-agent
```

## 获取ssh证书

清除所有的证书

```shell
step ssh logout
```

使用ca进行身份验证并获取证书

```shell
step ssh login --provisioner=uon cert-2026-02-18
```

测试

```
step ssh list
```

这个证书的有效期是一天，如果过期需要重新获取

## Log in via the bastion

```shell
ssh alyzw89@strange.cs.nott.ac.uk
```

```shell
Taught Students:
ssh -J USERNAME@canal1.cs.nott.ac.uk:2222 USERNAME@students2024.cs.nott.ac.uk

Staff, associates and PGR:
ssh -J USERNAME@valley1.cs.nott.ac.uk:2222 USERNAME@trent.cs.nott.ac.uk
```

```shell
ssh -J alyzw89@canal1.cs.nott.ac.uk:2222 alyzw89@strange.cs.nott.ac.uk
```

注意堡垒机使用端口2222访问不是22













