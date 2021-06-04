# Sunny-go-to-hell



## git 指令

< >:包起來的代表要填入你要的名稱

請大家盡量用 terminal 操作喔。


### 下載 & 更新

1. 下載一份到自己電腦裡面 

```bash
git clone <https://github.com/example/example.git>
```

2. 下載最新的進度 (從 github 上的 main 分支)

```bash
git pull origin main
```

### 日常 Workflow

**假設你改了你本機的 code, 要更新到 github 上**


1. 檢查一下目前的專案狀況

```bash
git status 
```

2. 如果有新增檔案的，要把他加入追蹤；把所有新東西加入追蹤，用 . 代替輸入所有檔案

```bash 
git add <example.py>    or    git add . 
```

3. 確認本次更改，並記錄目前改動了什麼東西

```bash
git commit -m <"log here">
```

4. 上傳更新 (到 github 的 main 分支)

```bash
git push origin main
```
