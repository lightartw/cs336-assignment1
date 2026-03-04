# BPE traing
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