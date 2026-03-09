# BPE
## BPE traing
- 实现过程中参考了以下文章：[CS336 Lab 1 实验笔记](https://www.zhouxin.space/notes/notes-on-cs336-lab-1/)
- 注意点
  - 使用cProfile测量性能
  - 使用tqdm监控过程
  - 尽量注明类型
  - 先实现后优化

- 大致步骤：
	- 初始化 **Vocabulary**，`vocabulary: dict[int, bytes]`
	- Pre-tokenization：对每一个chunk并发运行pretokenization，结果：
		- `cur_token: dict[bytes,tuple[bytes]]`，代表此pretoken在当前词汇表下的分组
		- `pretoken_count: dict[bytes, int]`，代表对应pretoken的数目		
	- 计算每个pair的数量，大致算法如下
		```
		for pretoken, cnt in token_count:
			token = cur_token[pretoken]
			for i in range(len(token) - 1):
				pair = (token[i], token[i+1])
				pair_count[pair] += cnt
        ```
	- merge
		- 找到`best_pair`
		- 修改词汇表，增加`best_pair`
		- 修改`cur_token`
			- 这里可以进行优化，我们需要提前记录下对应`pair`影响的 `pretoken`，修改的时候可以不用遍历
			- 因此增加一个记录 `pair_to_token: dict[tuple[bytes], list[bytes]]`，这一步放在初始化进行
- 优化
  - pretokenizer原实现将所有chunk存到chunks:list[str] 里，然后传给process_chunk，这会导致内存占用过大，因此应该传入input_path, start, end, 让process_chunk自己读取chunk
  - cProfile显示性能瓶颈在寻找best_pair中，使用max的复杂度为 O(N), 比较耗时，可以利用堆进行优化
- 优化前
```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      148   43.647    0.295   43.647    0.295 {method 'acquire' of '_thread.lock' objects}
    51023   43.472    0.001   82.500    0.002 {built-in method builtins.max}
369218707   39.028    0.000   39.028    0.000 ...\src\bpe.py:105(<lambda>)
    20356    2.378    0.000    2.378    0.000 {method 'write' of '_io.TextIOWrapper' objects}
     9743    1.282    0.000    1.474    0.000 ...\src\bpe.py:55(_apply_merge)
```
- 优化后
```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      148   31.476    0.213   31.476    0.213 {method 'acquire' of '_thread.lock' objects}
     9743    2.599    0.000    2.974    0.000 ...\src\bpe.py:74(_apply_merge)
  8336370    1.174    0.000    1.537    0.000 ...\tqdm\utils.py:375(<genexpr>)
   556017    1.091    0.000    1.246    0.000 {built-in method _heapq.heappop}
    19593    0.977    0.000    0.977    0.000 {method 'write' of '_io.TextIOWrapper' objects}
    58761    0.434    0.000    1.971    0.000 {built-in method builtins.sum}
```

### 实验结果
- TinyStory
  - Training time: 0.0122 hours (44.03 seconds)
  - Peak memory usage: 0.1164 GB (119.24 MB)
  - Longest token: b' accomplishment'
  - 性能分析：即上面优化后的结果，最耗时的部分为进主程等待多进程预分词完成的同步过程，而核心逻辑最耗时的为合并操作（_apply_merge）
- owt
  - 直接训练OOM了，对 heap 进行优化，将 push 操作放到循环外进行，减少 heap 中的无效数据
  - Training time: 1057.392 seconds
  - Peak Memory: 5981.53MB
  - Longest Token: b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82' (64 bytes)
- TinyStories(after optimize)
  - 优化后的 TinyStories 速度如下
  - Peak Memory: 60.89MB
  - Training time: 35.902 seconds

## BPE Tokenizer
- encode_one_token：有两种算法，
  - 一种是遍历merges直接模拟，较慢
  - 另外一种首先记录一个字典 pair_to_id，然后遍历 pretoken 找到 id 最低的 pair（即最先merge的pair），进行merge，速度提升很多

### 实验结果
- ratio 对比：

| Dataset        | TinyStories vocab&merges | OpenWebText vocab&merges |
|----------------|-----------------------|-----------------------|
| TinyStories    | 4.13                  | 3.97                  |
| OpenWebText    | 2.63                  | 4.47                  |

- 为什么使用uint16：
  - uint16 可以表示从 0 到 65535 的整数范围，词汇表大小分别为 10,000 和 32,000，足够表示，而且可以必uint32节省空间

