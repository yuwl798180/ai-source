<!-- TOC -->

- [基础运算](#基础运算)
- [字符串](#字符串)
- [list/tuple/set](#listtupleset)
- [读写数据](#读写数据)
- [流程控制](#流程控制)
- [类/继承](#类继承)
- [书写风格](#书写风格)

<!-- /TOC -->

## 基础运算

```python
5 / -3     # => -2
-5.0 / 3   # => -2.0
5.2 % 3    # => 2.2
5 ** 3     # => 125

True * 8   # => 8
False -5   # => -5
bool({})   # => False

'cool' if 3 > 2 else 'bad'
```

1. 除法 `/` 结果自动转换为小数，整除 `//` 为地板除

1. 布尔值 `True==1 False==0` 与布尔运算 `not and or`

1. `bool()` 只有 3 个结果为 `False` ： `None 0 空strings/lists/dicts/tuples`

1. `is` 是看两个变量是不是指向同一个对象， `==` 是看两个对象是不是有相同的值

1. 没有三目运算符 `?:`，等价的有 `true_ans if condition else wrong_ans`

## 字符串

```python
'hello ' + "world" + '!'    # => 'hello world!'
'hello world!'[6]           # => 'w'

'{} can be {}'.format('string', 'interpolated')
'{0} be nimble, {0} be quick, {1} be large'.format('Jack', 'Tim')
'{name} is {age}'.format(name='Bob', age=20)
'%s xxx %s' % ('str', 'str')

'%s can be %s the %s way' % ('Strings', 'interpolated', 'old')
```

1. `'{}'.format(str)` 格式化字符串输出

1. 之前的方式为 `'%s' % (str)`

## list/tuple/set

```python
# mutable

li.insert(i, x): 在 index i 处添加 x，只能添加一个

li.pop(i) : 在 i 处删除元素，只能删除一个

li.remove(x) : 删除第一个值为 x 的元素

del li[start:stop:step]: 随意删

clear() / copy() / reverse() / sort()

append(x) / extend(list) / lista + listb

count(x) / index(x)

li *= n

len(li) / x in li
```

```python
# tuple 元素不能修改，自带的括号可以省略。可以使用 + 和 del。常用做交换。

a, *b, c = (1, 2, 3, 4)   # => b == [3, 4]
a, b = b, a               # => tuple 快速交换
```

```python
# set 集合有交(&) 并(|) 补(-)
# add 方法添加新项
# a <= b 可以检查 a 是 b 的子集吗

one_set = set({1, 2, 3, 4, 5})
other_set = {4, 5, 6, 7}
print(one_set & other_set)    # => {4, 5}
print(one_set | other_set)    # => {1, 2, 3, 4, 5, 6, 7}
print(one_set - other_set)    # => {1, 2, 3}
```

```python
dict.keys() / dict.values() / dict.items()
dict.update(newdict)
dict.get(keyname)
dict.setdefault(keyname, value)
```

1. 元组（tuple）：小括号（可省略）、可以有相同元素、不可改只能读

1. 列表(list)：中括号、可以有相同元素、可以进行增删改查

1. 集合(set)：大括号、不存放相同元素，创建用 set()

## 读写数据

```python
import os
import json

datas = []
open with(fp) as f:
    for l in f:
        line = json.loads(l)
        datas.append(line)

open with(fn, 'w') as f:
    f.write(os.linesep.join([
        json.dumps(i, ensure_ascii=False) for i in datas
    ]))

```

## 流程控制

```python
some_var = 5

if some_var > 10:
    print("some_var比10大")
elif some_var < 10:    # elif句是可选的
    print("some_var比10小")
else:                  # else也是可选的
    print("some_var就是10")
```

```python
for animal in ["dog", "cat", "mouse"]:
    print("{} is a mammal".format(animal))
```

```python
x = 0
while x < 4:
    print(x)
    x += 1  # x = x + 1 的简写
```

```python
# 用try/except块处理异常状况

try:
    # 用raise抛出异常
    raise IndexError("This is an index error")
except IndexError as e:
    pass    # pass是无操作，但是应该在这里处理错误
except (TypeError, NameError):
    pass    # 可以同时处理不同类的错误
else:   # else语句是可选的，必须在所有的except之后
    print("All good!")   # 只有当try运行完没有错误的时候这句才会运行
```

```python
def connect_to_next_port(self, minimum):
    '''Connects to the next available port.

    Args:
        minimum: A port value greater or equal to 1024.
    Raises:
        ValueError: if the minimum port specified is less than 1024.
        ConnectionError: If no available port is found.
    Returns:
        The new minimum port.
    '''
    if minimum < 1024:
        raise VlueError('Minimum port must be at least 1024, not %d.' % (minimum,))
    port = self._find_next_open_port(minimum)
    if not port:
        raise ConnectionError('Could not connect to service on %d or higher.' % (minimum,))
    assert port >= minimum, 'Unexpected port %d when minimum was %d.' % (port, minimum)
    return port
```

```python
# iter(x) 构造迭代器，next(i) 不停的往后取值

somedict = {1:a, 2:b, 3:c}
i = iter(somedict.keys())
next(i)        # 1
i.__next__()   # 2
```

```python
def all_the_args(*args, **kwargs):
    print(args)     # (1, 'a', 1, 2, 3, 4)
    print(kwargs)   # {'a': 3, 'b': 4}

args = (1, 2, 3, 4)
kwargs = {"a": 3, "b": 4}
all_the_args(1, 'a', *args, **kwargs)
```

```python
x = 5

def setX(num):
    # 局部作用域的x和全局域的x是不同的
    x = num # => 43
    print ('local:', x) # => 43

def setGlobalX(num):
    global x
    print ('global:',x) # => 5
    x = num # 现在全局域的x被赋值
    print ('global:',x) # => 6

setX(43)
setGlobalX(6)

print('global:', x)   # 6
```

```python
# lambda 匿名函数

(lambda x: x > 2)(3)
```

```python
seasons = ['a', 'b', 'c', 'd']
list(enumerate(seasons))       # [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]
```

```python
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]

list(zip(x, y, z))   ## [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```

## 类/继承

```python
class Human:

    # class attribute
    species = "H.sapiens"

    # basic inititalizer, this is callled when this class is instantiated.
    def __init__(self, name):
        self.name = name
        self.age = 0

    # instance method
    def say(self, msg):
        print("{name}: {message}".format(name=self.name, message=msg))

    def sing(self):
        return 'yo... yo... yo...'

    #class method is shared among all instances
    @classmethod
    def get_species(cls):
        return cls.species

    # static method is called without a class or instance reference
    @staticmethod
    def grunt():
        return "*grunt*"

    # a property is just like a getter
    @property
    def age(self):
        return self._age

    # corresponding setter
    @age.setter
    def age(self, age):
        self._age = age

    # corresponding deleter
    @age.deleter
    def age(self):
        del self._age

# 示例
print('instantiate a class:')
i = Human(name="Ian")
i.say("hi")    # "Ian: hi"
j = Human(name="Joel")
j.say("hello")    # "Joel: hello"

print('\ncall class method:')
i.say(i.get_species())    # Ian: H.sapiens
Human.species = 'H.neanderthalensis'
i.say(i.get_species())    # Ian: H.neanderthalensis
j.say(j.get_species())    # Joel: H.neanderthalensis

print('\ncall static method:')
print(Human.grunt())     # => "*grunt*"

print('\nupdate property, use setter getter deleter:')
i.age = 42
i.say(i.age)   # => "Ian: 42"
j.say(j.age)   # => "Joel: 0"
del i.age
j.say(j.age)   # => "Joel: 0"
i.say(i.age)   # => raise an AttributeError: 'Human' object has no attribute '_age'
```

```python
# Inheritance
class Superhero(Human):

    # override parents' attributes
    species = 'Superhuman'

    # constructor
    def __init__(self, name, movie=False, superpowers=['super strengh', 'bulletproofing']):
        self.fictional = True
        self.movie = movie
        self.superpowers = superpowers

        # call parent class constructor
        super().__init__(name)

    # override sing method
    def sing(self):
        return 'Dun, dun, DUN!'

    def boast(self):
        for power in self.superpowers:
            print('I wield the power of {power}'.format(power=power))


# 示例
sup = Superhero(name='Tick')

if isinstance(sup, Human):
    print('I am human')
if type(sup) is Superhero:
    print('I am a superhero')

# Get the Method Resolution search Order
print(Superhero.__mro__)  # (<class '__main__.Superhero'>, <class '__main__.Human'>, <class 'object'>)

print('Am I Oscar eligible? ' + str(sup.movie))  # Am I Oscar eligible? False
```

```python
# another class definition for multiple inheritance
class Bat:
    species = 'Baty'

    def __init__(self, can_fly=True):
        self.fly = can_fly

    def say(self, msg):
        msg = '... ... ...'
        return msg

    def sonar(self):
        return '))) ... ((('

b = Bat()
print(b.say('hello'))  # ... ... ...
print(b.fly)  # True

# multiple inheritance
class Batman(Superhero, Bat):

    def __init__(self, *args, **kwargs):
        # 一般是这样继承: super(Batman, self).__init__(*args, **kwargs)

        Superhero.__init__(self, 'anonymous', movie=True, superpowers=['Wealthy'], *args, **kwargs)
        Bat.__init__(self, *args, can_fly=False, **kwargs)
        self.name = 'Sad Affleck'

    def sing(self):
        return 'nan nan nan nan nan batman!'


supman = Batman()
print(Batman.__mro__)  # (<class '__main__.Batman'>, <class '__main__.Superhero'>, <class '__main__.Human'>, <class '__main__.Bat'>, <class 'object'>)

print('\n' + supman.get_species())  # Superhuman
print(supman.sing())   # nan nan nan nan nan batman!
print(supman.sonar())  # ))) ... (((
supman.age = 100
print(supman.age)      # 100
print('Can I fly?\nans: ' + str(supman.fly))  # False


print(supman.__dict__)
{'fictional': True,
 'movie': True,
 'superpowers': ['Wealthy'],
 'name': 'Sad Affleck',
 '_age': 100,
 'fly': False}
```

```python
# Decorator

from functools import wraps
from time import time

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        print('%s(%r,%r) -> %r'%(func.__name__, args, kwargs, rv))
        after = time()
        print('time token:', after - before)
        return rv

    return wrapper

@timer
def sum(a, b=10):
    return a+b


sum(2)
# sum((2,),{}) -> 12
# time token: 0.00024509429931640625
```

## 书写风格

```python
result = [mapping_expr for value in iterable if filter_expr]

result = [{'key': value} for value in iterable
          if a_long_filter_expression(value)]

result = [complicated_transform(x)
          for x in iterable if predicate(x)]

descriptive_name = [
    transform({'key': key, 'value': value}, color='black')
    for key, value in generate_iterable(some_input)
    if complicated_condition_is_met(key, value)
]

result = []
for x in range(10):
    for y in range(5):
        if x * y > 10:
            result.append((x, y))

return {x: complicated_transform(x)
        for x in long_generator_function(parameter)
        in x is not None}

squares_generator = (x**2 for x in range(10))

unique_names = {user.name for user in users if user is not None}

eat(jelly_bean for jelly_bean in jelly_beans
    if jelly_bean.color == 'black')
```

```python
with open('hello.txt') as hello_file:
    for line in hello_file:
        print(line)
```
