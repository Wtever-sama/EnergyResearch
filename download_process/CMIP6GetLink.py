import re

# 修改sh文件路径
f = open(r"E:\DataDownload\CMIP6\rsds-temp\wget_script_2025-2-18_13-18-52.sh", 'r')

content = f.read()
patten0 = "http://.*.nc"
patten1 = "https://.*.nc"

result0 = re.findall(patten0, content)
result1 = re.findall(patten1, content)

wf = open("./CMIP6LinkGetresult.txt", 'w')
wf.write('\n'.join(result0))
wf.write('\n'.join(result1))