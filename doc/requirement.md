> 本文件由 `大作业.docx` 自动抽取为 Markdown（图片已另存为 `大作业_image*.png`）。

大作业

总体目标：把MNIST训练集（mnist_train）划分为train_dataset和val_dataset，train_dataset用于训练CNN，val_dataset用于选择CNN的超参数，最后在MNIST测试集（mnist_test）上测试。

条件：对于train_dataset，只允许使用下列图片的标签，其他图片的标签不允许使用。允许使用标签的图片的索引（train_dataset里的图片索引）为：[1173, 3336, 12529, 12785, 12979, 17351, 27048, 40579, 43128, 46498]，即对于每个类别，允许使用的有标签的图片个数为1。

问题：如果只使用上述10个图片及其标签训练CNN，显然会导致过拟合（over-fitting），需要尽可能地使用更多的数据，或者使用图半监督学习。

参考思路：对于给定的每个类别的1个图片，在train_dataset里尽可能地找出与其相似的图片（即属于同一个簇（使用自编码机在嵌入空间中寻找簇）），给予它们标签（伪标签），或者使用多模态大模型对数据加伪标签，或者使用图半监督学习传播这10个标签，等等。

把mnist_train数据集划分为train_dataset和val_dataset时，不同的seed会导致不同的划分。为了统一划分方式，使得上述给定的图片的索引在所有机器上一致，提供如下代码：

统一划分基准代码（转写自原文档代码截图，另存为 `大作业_split.py`；本仓库 `data.py` 已逐字复现并通过锚点校验）：

```python
seed = 42

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止 hash 随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])

mnist_train = MNIST(root='./', download=True, transform=transform)
mnist_test = MNIST(root='./', train=False, download=True, transform=transform)

# split train/val subset
generator = torch.Generator().manual_seed(seed)
train_dataset, val_dataset = random_split(mnist_train, [50000, 10000], generator=generator)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

提交代码和报告至：yeweiysh@qq.com，代码附在报告末尾，格式便于人类阅读。截止日期为：2026/6/30 23:59:59.
