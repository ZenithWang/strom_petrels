```shell
ssh-add -D
step ssh logout
Start-Service ssh-agent
step ssh login --provisioner=uon cert-2026-06-20
ssh-add -l
ssh -N -L 10022:strange.cs.nott.ac.uk:22 alyzw89@canal1.cs.nott.ac.uk -p 2222
//先运行这个创建隧道之后使用finalshell连接
//以下是使用powershell连接
step ssh list
ssh -J alyzw89@canal1.cs.nott.ac.uk:2222 alyzw89@strange.cs.nott.ac.uk

scp -r -o ProxyJump=alyzw89@canal1.cs.nott.ac.uk:2222 alyzw89@strange.cs.nott.ac.uk:/mnt/data2/skokholm/data/2021/recordings/011_210522_184504.flac D:\Documents\StromPetrels\download
```

finalshell中的配置

ip：127.0.0.1 端口：10022

认证: keyboard

用户名: alyzw89



