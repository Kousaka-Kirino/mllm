# 数据：不进仓库，可重建

```bash
python -m stage1_data.build --profile pilot --seed 17
```

生成物：

| 路径 | 内容 | 大小 |
|---|---|---|
| `data/synth/{train,dev,val,test_locked}.jsonl` | 样本清单（含 family_id、meta.table、实测字形像素） | 每千条约 1 MB |
| `data/synth/images/` | 渲染出的 PNG | 每千张约 30 MB |
| `data/datacard.json` | 数据卡：计数、哈希、先验强度实测、像素分布 | 约 20 KB |
| `data/hashes.json` | 全量文件 sha256 清单（用于复现校验） | 约 100 KB |

这些文件被 `.gitignore` 排除，因为**它们由代码 + 种子确定性地生成**。
`python -m stage1_data.validate` 会重算哈希并与 `data/hashes.json` 比对，
所以"数据不在仓库里"不等于"数据不可复现"。
