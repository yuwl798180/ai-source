
## Basic

### 函数参数要求

```python
# python3.8
def fun(a, b, /, c, d, *, e, f):
    print(a, b, c, d, e, f)

# 调用时，6个参数的要求：
# a, b 必须使用指定位置参数
# c, d 随意
# e, f 必须使用关键字参数
fun(1, 2, 3, d=4, e=5, f=6)
```

### yield 使用

```python
import sys

def fib(n):
    a, b, count = 0, 1, 0
    while True:
        if count >= n:
            return
        yield a
        a , b = b, a + b
        count += 1
# usage
fib_gen = fib(10)
while True:
    try:
        print(next(fib_gen), end=', ')
    except StopIteration:
        sys.exit()
```

