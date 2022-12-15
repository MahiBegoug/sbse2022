# Search-Based Test Generation - Part 1


```python
import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import inspect
import ast
import astor

import sys

# For presenting as slides
#plt.rcParams['figure.figsize'] = [12, 8]
#plt.rcParams.update({'font.size': 22})
#plt.rcParams['lines.linewidth'] = 3
```

## The Test Data Generation Problem

The problem we will consider in this chapter is the following: Given an entry point to a program (a function), we want to find values for the parameters of this function such that the execution of the function reaches a particular point in the program. In other words, we aim to find a test input to the program that covers a target statement. We will then generalise this problem to finding test inputs to cover _all_ statements in the program.

Assume we are aiming to test the following function under test:


```python
def test_me(x, y):
    if x == 2 * (y + 1):
        return True
    else:
        return False
```

The `test_me` function has two input parameters, `x`and `y`, and it returns `True` or `False` depending on how the parameters relate:


```python
test_me(10, 10)
```




    False




```python
test_me(22, 10)
```




    True



In order to address the test generation problem as a search problem, we need to decide on an encoding, and derive appropriate search operators. It is possible to use bitvectors like we did on previous problems; however, a simpler interpretation of the parameters of a function is a list of the actual parameters. That is, we encode a test input as a list of parameters; we will start by assuming that all parameters are numeric.

The representation for inputs of this function is lists of length two, one element for `x` and one for `y`. As numeric values in Python are unbounded, we need to decide on some finite bounds for these values, e.g.:


```python
MAX = 1000
MIN = -MAX
```

For generating inputs we can now uniformly sample in the range (MIN, MAX). The length of the vector shall be the number of parameters of the function under test. Rather than hard coding such a parameter, we can also make our approach generalise better by using inspection to determine how many parameters the function under test has:


```python
from inspect import signature
sig = signature(test_me)
num_parameters = len(sig.parameters)
num_parameters
```




    2



As usual, we will define the representation implicitly using a function that produces random instances.


```python
def get_random_individual():
    return [random.randint(MIN, MAX) for _ in range(num_parameters)]
```


```python
get_random_individual()
```




    [-395, -549]



We need to define search operators matching this representation. To apply local search, we need to define the neighbourhood. For example, we could define one upper and one lower neighbour for each parameter:

- `x-1, y`
- `x+1, y`
- `x, y+1`
- `x, y-1`


```python
def get_neighbours(individual):
    neighbours = []
    for p in range(len(individual)): 
        if individual[p] > MIN:
            neighbour = individual[:]
            neighbour[p] = individual[p] - 1
            neighbours.append(neighbour)
        if individual[p] < MAX:
            neighbour = individual[:]
            neighbour[p] = individual[p] + 1
            neighbours.append(neighbour)
            
    return neighbours
```


```python
x = get_random_individual()
x
```




    [-960, 316]




```python
get_neighbours(x)
```




    [[-961, 316], [-959, 316], [-960, 315], [-960, 317]]



Before we can apply search, we also need to define a fitness function.  Suppose that we are interested in covering the `True` branch of the if-condition in the `test_me()` function, i.e. `x == 2 * (y + 1)`.


```python
def test_me(x, y):
    if x == 2 * (y + 1):
        return True
    else:
        return False
```

How close is a given input tuple for this function from reaching the target (true) branch of `x == 2 * (y + 1)`?

Let's consider an arbitrary point in the search space, e.g. `(274, 153)`. The if-condition compares the following values:


```python
x = 274
y = 153
x, 2 * (y + 1)
```




    (274, 308)



In order to make the branch true, both values need to be the same. Thus, the more they differ, the further we are away from making the comparison true, and the less they differ, the closer we are from making the comparison true. Thus, we can quantify "how false" the comparison is by calculating the difference between `x` and `2 * (y + 1)`. Thus, we can calculate this distance as `abs(x - 2 * (y + 1))`:


```python
def calculate_distance(x, y):
    return abs(x - 2 * (y + 1))
```


```python
calculate_distance(274, 153)
```




    34



We can use this distance value as our fitness function, since we can nicely measure how close we are to an optimal solution. Note, however, that "better" doesn't mean "bigger" in this case; the smaller the distance the better. This is not a problem, since any algorithm that can maximize a value can also be made to minimize it instead.

For each value in the search space of integer tuples, this distance value defines the elevation in our search landscape. Since our example search space is two-dimensional, the search landscape is three-dimensional and we can plot it to see what it looks like:


```python
x = np.outer(np.linspace(-10, 10, 30), np.ones(30))
y = x.copy().T
z = calculate_distance(x, y)

fig = plt.figure(figsize=(12, 12))
ax  = plt.axes(projection='3d')

ax.plot_surface(x, y, z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0);
```


    
![png](6/output_30_0.png)
    


The optimal values, i.e. those that make the if-condition true, have fitness value 0 and can be clearly seen at the bottom of the plot. The further away from the optimal values, the higher elevated the points in the search space.

This distance can serve as our fitness function if we aim to cover the true branch of the program in our example:


```python
def get_fitness(individual):
    x = individual[0]
    y = individual[1]
    return abs(x - 2 * (y + 1))
```

We can now use any local search algorithm we have defined previously, with only one modification: In the prior examples where we applied local search we were always maximising fitness values; now we are minimising, so a hillclimber, for example, should only move to neighbours with _smaller_ fitness values:


```python
max_steps = 10000
fitness_values = []
```

Let's use a steepest ascent hillclimber:


```python
def hillclimbing():
    current = get_random_individual()
    fitness = get_fitness(current)
    best, best_fitness = current[:], fitness
    print(f"Starting at fitness {best_fitness}: {best}")

    step = 0
    while step < max_steps and best_fitness > 0:
        neighbours = [(x, get_fitness(x)) for x in get_neighbours(current)]
        best_neighbour, neighbour_fitness = min(neighbours, key=lambda i: i[1])
        step += len(neighbours)        
        fitness_values.extend([best_fitness] * len(neighbours))
        if neighbour_fitness < fitness:
            current = best_neighbour
            fitness = neighbour_fitness
            if fitness < best_fitness:
                best = current[:]
                best_fitness = fitness
        else:
            # Random restart if no neighbour is better
            current = get_random_individual()
            fitness = get_fitness(current)
            step += 1
            if fitness < best_fitness:
                best = current[:]
                best_fitness = fitness
            fitness_values.append(best_fitness)


    print(f"Solution fitness after {step} fitness evaluations: {best_fitness}: {best}")
    return best
```


```python
max_steps = 10000
fitness_values = []
hillclimbing()
```

    Starting at fitness 1006: [910, -49]
    Solution fitness after 2012 fitness evaluations: 0: [910, 454]





    [910, 454]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10d156cb0>]




    
![png](6/output_39_1.png)
    


Since there are no local optima, the hillclimber will easily find the solution, even without restarts. However, this can take a while, in particular if we use a larger input space:


```python
MAX = 100000
MIN = -MAX
```


```python
fitness_values = []
hillclimbing()
plt.plot(fitness_values)
```

    Starting at fitness 179321: [74063, -52630]
    Solution fitness after 10000 fitness evaluations: 174321: [74063, -50130]





    [<matplotlib.lines.Line2D at 0x10d1d76a0>]




    
![png](6/output_42_2.png)
    


Unless the randomly chosen initial point is already close to an optimal solution, the hillclimber is going to be hopeless in moving through the search space within a reasonable number of iterations.

## Alternating Variable Method

The search problem represented by the `test_me` function has an easy fitness landscape with no local optima. However, it still takes quite long to reach the optimum, depending on where the random starting point lies in the search space. This is because the neighbourhood for real program inputs can be quite large, depending on the number of parameters, and even just the search space for each parameter individually can already be very large. In our example we restricted `MAX` and `MIN` to a very narrow range, but imagine doing this for 32 bit integers. Both these problems are addressed with an adapted version of our hillclimber known as the _Alternating Variable Method_, which differs from the hillclimber in two ways: 
1. Rather than considering the neighbourhood of all input parameters at once, we apply search to each parameter individually in turn
2. Rather than taking only small steps of size 1, we allow larger jumps in the search space.

Let's first consider the second aspect, larger jumps in the search space. The idea is to apply a _pattern_ search where we first decide on a direction in the search space to move, and then apply increasingly larger steps in that direction as long as the fitness improves. We only consider a single parameter, thus the "direction" simply refers to whether one increases or decreases this value. The function thus takes (1) the individual on which to perform the search, (2) a particular parameter we are considering, (3) the direction of the search, and (4) the starting fitness values. 


```python
def pattern_search(individual, parameter, direction, fitness):
    print(f"  {individual}, direction {direction}, fitness {fitness}")

    individual[parameter] = individual[parameter] + direction
    new_fitness = get_fitness(individual)
    if new_fitness < fitness:
        fitness_values.append(new_fitness)
        return pattern_search(individual, parameter, 2 * direction, new_fitness)
    else:
        # If fitness is not better we overshot. Undo last move, and return
        fitness_values.append(fitness)
        individual[parameter] = individual[parameter] - direction
        return fitness
```

For example, let's assume `y` is a large value (1000), and `x` is considerably smaller. For our example function, the optimal value for `x` would thus be at 2200. Applying the search to `x` we thus need to move in the positive direction (`1`), and the function will do this with increasing steps until it "overshoots".


```python
x = [0, 1000]
f = get_fitness(x)
pattern_search(x, 0, 1, get_fitness(x))
```

      [0, 1000], direction 1, fitness 2002
      [1, 1000], direction 2, fitness 2001
      [3, 1000], direction 4, fitness 1999
      [7, 1000], direction 8, fitness 1995
      [15, 1000], direction 16, fitness 1987
      [31, 1000], direction 32, fitness 1971
      [63, 1000], direction 64, fitness 1939
      [127, 1000], direction 128, fitness 1875
      [255, 1000], direction 256, fitness 1747
      [511, 1000], direction 512, fitness 1491
      [1023, 1000], direction 1024, fitness 979
      [2047, 1000], direction 2048, fitness 45





    45



If `x` is larger than `y` we would need to move in the other direction, and the search does this until it undershoots the target of 2200:


```python
x = [10000, 1000]
f = get_fitness(x)
pattern_search(x, 0, -1, get_fitness(x))
```

      [10000, 1000], direction -1, fitness 7998
      [9999, 1000], direction -2, fitness 7997
      [9997, 1000], direction -4, fitness 7995
      [9993, 1000], direction -8, fitness 7991
      [9985, 1000], direction -16, fitness 7983
      [9969, 1000], direction -32, fitness 7967
      [9937, 1000], direction -64, fitness 7935
      [9873, 1000], direction -128, fitness 7871
      [9745, 1000], direction -256, fitness 7743
      [9489, 1000], direction -512, fitness 7487
      [8977, 1000], direction -1024, fitness 6975
      [7953, 1000], direction -2048, fitness 5951
      [5905, 1000], direction -4096, fitness 3903
      [1809, 1000], direction -8192, fitness 193





    193



The AVM algorithm applies the pattern search as follows:
1. Start with the first parameter
2. Probe the neighbourhood of the parameter to find the direction of the search
3. Apply pattern search in that direction
4. Repeat probing + pattern search until no more improvement can be made
5. Move to the next parameter, and go to step 2

Like a regular hillclimber, the search may get stuck in local optima and needs to use random restarts. The algorithm is stuck if it probed all parameters in sequence and none of the parameters allowed a move that improved fitness.


```python
def probe_and_search(individual, parameter, fitness):
    new_parameters = individual[:]
    value = new_parameters[parameter]
    new_fitness = fitness
    # Try +1
    new_parameters[parameter] = individual[parameter] + 1
    print(f"Trying +1 at fitness {fitness}: {new_parameters}")
    new_fitness = get_fitness(new_parameters)
    if new_fitness < fitness:
        fitness_values.append(new_fitness)
        new_fitness = pattern_search(new_parameters, parameter, 2, new_fitness)
    else:
        # Try -1
        fitness_values.append(fitness)
        new_parameters[parameter] = individual[parameter] - 1
        print(f"Trying -1 at fitness {fitness}: {new_parameters}")
        new_fitness = get_fitness(new_parameters)
        if new_fitness < fitness:
            fitness_values.append(new_fitness)
            new_fitness = pattern_search(new_parameters, parameter, -2, new_fitness)
        else:
            fitness_values.append(fitness)
            new_parameters[parameter] = individual[parameter]
            new_fitness = fitness
            
    return new_parameters, new_fitness
```


```python
def avm():
    current = get_random_individual()
    fitness = get_fitness(current)
    best, best_fitness = current[:], fitness
    fitness_values.append(best_fitness)    
    print(f"Starting at fitness {best_fitness}: {current}")

    changed = True
    while len(fitness_values) < max_steps and best_fitness > 0:
        # Random restart
        if not changed: 
            current = get_random_individual()
            fitness = get_fitness(current)
            fitness_values.append(fitness)
        changed = False
            
        parameter = 0
        while parameter < len(current):
            print(f"Current parameter: {parameter}")
            new_parameters, new_fitness = probe_and_search(current, parameter, fitness)
            if current != new_parameters:
                # Keep on searching
                changed = True
                current = new_parameters
                fitness = new_fitness
                if fitness < best_fitness:
                    best_fitness = fitness
                    best = current[:]
            else:
                parameter += 1

    print(f"Solution fitness {best_fitness}: {best}")
    return best
```


```python
fitness_values = []
avm()
```

    Starting at fitness 12161: [34073, 10955]
    Current parameter: 0
    Trying +1 at fitness 12161: [34074, 10955]
    Trying -1 at fitness 12161: [34072, 10955]
      [34072, 10955], direction -2, fitness 12160
      [34070, 10955], direction -4, fitness 12158
      [34066, 10955], direction -8, fitness 12154
      [34058, 10955], direction -16, fitness 12146
      [34042, 10955], direction -32, fitness 12130
      [34010, 10955], direction -64, fitness 12098
      [33946, 10955], direction -128, fitness 12034
      [33818, 10955], direction -256, fitness 11906
      [33562, 10955], direction -512, fitness 11650
      [33050, 10955], direction -1024, fitness 11138
      [32026, 10955], direction -2048, fitness 10114
      [29978, 10955], direction -4096, fitness 8066
      [25882, 10955], direction -8192, fitness 3970
    Current parameter: 0
    Trying +1 at fitness 3970: [25883, 10955]
    Trying -1 at fitness 3970: [25881, 10955]
      [25881, 10955], direction -2, fitness 3969
      [25879, 10955], direction -4, fitness 3967
      [25875, 10955], direction -8, fitness 3963
      [25867, 10955], direction -16, fitness 3955
      [25851, 10955], direction -32, fitness 3939
      [25819, 10955], direction -64, fitness 3907
      [25755, 10955], direction -128, fitness 3843
      [25627, 10955], direction -256, fitness 3715
      [25371, 10955], direction -512, fitness 3459
      [24859, 10955], direction -1024, fitness 2947
      [23835, 10955], direction -2048, fitness 1923
      [21787, 10955], direction -4096, fitness 125
    Current parameter: 0
    Trying +1 at fitness 125: [21788, 10955]
      [21788, 10955], direction 2, fitness 124
      [21790, 10955], direction 4, fitness 122
      [21794, 10955], direction 8, fitness 118
      [21802, 10955], direction 16, fitness 110
      [21818, 10955], direction 32, fitness 94
      [21850, 10955], direction 64, fitness 62
      [21914, 10955], direction 128, fitness 2
    Current parameter: 0
    Trying +1 at fitness 2: [21915, 10955]
    Trying -1 at fitness 2: [21913, 10955]
      [21913, 10955], direction -2, fitness 1
    Current parameter: 0
    Trying +1 at fitness 1: [21914, 10955]
    Trying -1 at fitness 1: [21912, 10955]
      [21912, 10955], direction -2, fitness 0
    Current parameter: 0
    Trying +1 at fitness 0: [21913, 10955]
    Trying -1 at fitness 0: [21911, 10955]
    Current parameter: 1
    Trying +1 at fitness 0: [21912, 10956]
    Trying -1 at fitness 0: [21912, 10954]
    Solution fitness 0: [21912, 10955]





    [21912, 10955]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10d1c7760>]




    
![png](6/output_56_1.png)
    


The pattern search even works efficiently if we increase the size of the search space to 64-bit numbers:


```python
MAX = 2**32
MIN = -MAX
```


```python
fitness_values = []
avm()
```

    Starting at fitness 3861078628: [2383666104, -738706263]
    Current parameter: 0
    Trying +1 at fitness 3861078628: [2383666105, -738706263]
    Trying -1 at fitness 3861078628: [2383666103, -738706263]
      [2383666103, -738706263], direction -2, fitness 3861078627
      [2383666101, -738706263], direction -4, fitness 3861078625
      [2383666097, -738706263], direction -8, fitness 3861078621
      [2383666089, -738706263], direction -16, fitness 3861078613
      [2383666073, -738706263], direction -32, fitness 3861078597
      [2383666041, -738706263], direction -64, fitness 3861078565
      [2383665977, -738706263], direction -128, fitness 3861078501
      [2383665849, -738706263], direction -256, fitness 3861078373
      [2383665593, -738706263], direction -512, fitness 3861078117
      [2383665081, -738706263], direction -1024, fitness 3861077605
      [2383664057, -738706263], direction -2048, fitness 3861076581
      [2383662009, -738706263], direction -4096, fitness 3861074533
      [2383657913, -738706263], direction -8192, fitness 3861070437
      [2383649721, -738706263], direction -16384, fitness 3861062245
      [2383633337, -738706263], direction -32768, fitness 3861045861
      [2383600569, -738706263], direction -65536, fitness 3861013093
      [2383535033, -738706263], direction -131072, fitness 3860947557
      [2383403961, -738706263], direction -262144, fitness 3860816485
      [2383141817, -738706263], direction -524288, fitness 3860554341
      [2382617529, -738706263], direction -1048576, fitness 3860030053
      [2381568953, -738706263], direction -2097152, fitness 3858981477
      [2379471801, -738706263], direction -4194304, fitness 3856884325
      [2375277497, -738706263], direction -8388608, fitness 3852690021
      [2366888889, -738706263], direction -16777216, fitness 3844301413
      [2350111673, -738706263], direction -33554432, fitness 3827524197
      [2316557241, -738706263], direction -67108864, fitness 3793969765
      [2249448377, -738706263], direction -134217728, fitness 3726860901
      [2115230649, -738706263], direction -268435456, fitness 3592643173
      [1846795193, -738706263], direction -536870912, fitness 3324207717
      [1309924281, -738706263], direction -1073741824, fitness 2787336805
      [236182457, -738706263], direction -2147483648, fitness 1713594981
      [-1911301191, -738706263], direction -4294967296, fitness 433888667
    Current parameter: 0
    Trying +1 at fitness 433888667: [-1911301190, -738706263]
      [-1911301190, -738706263], direction 2, fitness 433888666
      [-1911301188, -738706263], direction 4, fitness 433888664
      [-1911301184, -738706263], direction 8, fitness 433888660
      [-1911301176, -738706263], direction 16, fitness 433888652
      [-1911301160, -738706263], direction 32, fitness 433888636
      [-1911301128, -738706263], direction 64, fitness 433888604
      [-1911301064, -738706263], direction 128, fitness 433888540
      [-1911300936, -738706263], direction 256, fitness 433888412
      [-1911300680, -738706263], direction 512, fitness 433888156
      [-1911300168, -738706263], direction 1024, fitness 433887644
      [-1911299144, -738706263], direction 2048, fitness 433886620
      [-1911297096, -738706263], direction 4096, fitness 433884572
      [-1911293000, -738706263], direction 8192, fitness 433880476
      [-1911284808, -738706263], direction 16384, fitness 433872284
      [-1911268424, -738706263], direction 32768, fitness 433855900
      [-1911235656, -738706263], direction 65536, fitness 433823132
      [-1911170120, -738706263], direction 131072, fitness 433757596
      [-1911039048, -738706263], direction 262144, fitness 433626524
      [-1910776904, -738706263], direction 524288, fitness 433364380
      [-1910252616, -738706263], direction 1048576, fitness 432840092
      [-1909204040, -738706263], direction 2097152, fitness 431791516
      [-1907106888, -738706263], direction 4194304, fitness 429694364
      [-1902912584, -738706263], direction 8388608, fitness 425500060
      [-1894523976, -738706263], direction 16777216, fitness 417111452
      [-1877746760, -738706263], direction 33554432, fitness 400334236
      [-1844192328, -738706263], direction 67108864, fitness 366779804
      [-1777083464, -738706263], direction 134217728, fitness 299670940
      [-1642865736, -738706263], direction 268435456, fitness 165453212
      [-1374430280, -738706263], direction 536870912, fitness 102982244
    Current parameter: 0
    Trying +1 at fitness 102982244: [-1374430279, -738706263]
    Trying -1 at fitness 102982244: [-1374430281, -738706263]
      [-1374430281, -738706263], direction -2, fitness 102982243
      [-1374430283, -738706263], direction -4, fitness 102982241
      [-1374430287, -738706263], direction -8, fitness 102982237
      [-1374430295, -738706263], direction -16, fitness 102982229
      [-1374430311, -738706263], direction -32, fitness 102982213
      [-1374430343, -738706263], direction -64, fitness 102982181
      [-1374430407, -738706263], direction -128, fitness 102982117
      [-1374430535, -738706263], direction -256, fitness 102981989
      [-1374430791, -738706263], direction -512, fitness 102981733
      [-1374431303, -738706263], direction -1024, fitness 102981221
      [-1374432327, -738706263], direction -2048, fitness 102980197
      [-1374434375, -738706263], direction -4096, fitness 102978149
      [-1374438471, -738706263], direction -8192, fitness 102974053
      [-1374446663, -738706263], direction -16384, fitness 102965861
      [-1374463047, -738706263], direction -32768, fitness 102949477
      [-1374495815, -738706263], direction -65536, fitness 102916709
      [-1374561351, -738706263], direction -131072, fitness 102851173
      [-1374692423, -738706263], direction -262144, fitness 102720101
      [-1374954567, -738706263], direction -524288, fitness 102457957
      [-1375478855, -738706263], direction -1048576, fitness 101933669
      [-1376527431, -738706263], direction -2097152, fitness 100885093
      [-1378624583, -738706263], direction -4194304, fitness 98787941
      [-1382818887, -738706263], direction -8388608, fitness 94593637
      [-1391207495, -738706263], direction -16777216, fitness 86205029
      [-1407984711, -738706263], direction -33554432, fitness 69427813
      [-1441539143, -738706263], direction -67108864, fitness 35873381
      [-1508648007, -738706263], direction -134217728, fitness 31235483
    Current parameter: 0
    Trying +1 at fitness 31235483: [-1508648006, -738706263]
      [-1508648006, -738706263], direction 2, fitness 31235482
      [-1508648004, -738706263], direction 4, fitness 31235480
      [-1508648000, -738706263], direction 8, fitness 31235476
      [-1508647992, -738706263], direction 16, fitness 31235468
      [-1508647976, -738706263], direction 32, fitness 31235452
      [-1508647944, -738706263], direction 64, fitness 31235420
      [-1508647880, -738706263], direction 128, fitness 31235356
      [-1508647752, -738706263], direction 256, fitness 31235228
      [-1508647496, -738706263], direction 512, fitness 31234972
      [-1508646984, -738706263], direction 1024, fitness 31234460
      [-1508645960, -738706263], direction 2048, fitness 31233436
      [-1508643912, -738706263], direction 4096, fitness 31231388
      [-1508639816, -738706263], direction 8192, fitness 31227292
      [-1508631624, -738706263], direction 16384, fitness 31219100
      [-1508615240, -738706263], direction 32768, fitness 31202716
      [-1508582472, -738706263], direction 65536, fitness 31169948
      [-1508516936, -738706263], direction 131072, fitness 31104412
      [-1508385864, -738706263], direction 262144, fitness 30973340
      [-1508123720, -738706263], direction 524288, fitness 30711196
      [-1507599432, -738706263], direction 1048576, fitness 30186908
      [-1506550856, -738706263], direction 2097152, fitness 29138332
      [-1504453704, -738706263], direction 4194304, fitness 27041180
      [-1500259400, -738706263], direction 8388608, fitness 22846876
      [-1491870792, -738706263], direction 16777216, fitness 14458268
      [-1475093576, -738706263], direction 33554432, fitness 2318948
    Current parameter: 0
    Trying +1 at fitness 2318948: [-1475093575, -738706263]
    Trying -1 at fitness 2318948: [-1475093577, -738706263]
      [-1475093577, -738706263], direction -2, fitness 2318947
      [-1475093579, -738706263], direction -4, fitness 2318945
      [-1475093583, -738706263], direction -8, fitness 2318941
      [-1475093591, -738706263], direction -16, fitness 2318933
      [-1475093607, -738706263], direction -32, fitness 2318917
      [-1475093639, -738706263], direction -64, fitness 2318885
      [-1475093703, -738706263], direction -128, fitness 2318821
      [-1475093831, -738706263], direction -256, fitness 2318693
      [-1475094087, -738706263], direction -512, fitness 2318437
      [-1475094599, -738706263], direction -1024, fitness 2317925
      [-1475095623, -738706263], direction -2048, fitness 2316901
      [-1475097671, -738706263], direction -4096, fitness 2314853
      [-1475101767, -738706263], direction -8192, fitness 2310757
      [-1475109959, -738706263], direction -16384, fitness 2302565
      [-1475126343, -738706263], direction -32768, fitness 2286181
      [-1475159111, -738706263], direction -65536, fitness 2253413
      [-1475224647, -738706263], direction -131072, fitness 2187877
      [-1475355719, -738706263], direction -262144, fitness 2056805
      [-1475617863, -738706263], direction -524288, fitness 1794661
      [-1476142151, -738706263], direction -1048576, fitness 1270373
      [-1477190727, -738706263], direction -2097152, fitness 221797
    Current parameter: 0
    Trying +1 at fitness 221797: [-1477190726, -738706263]
    Trying -1 at fitness 221797: [-1477190728, -738706263]
      [-1477190728, -738706263], direction -2, fitness 221796
      [-1477190730, -738706263], direction -4, fitness 221794
      [-1477190734, -738706263], direction -8, fitness 221790
      [-1477190742, -738706263], direction -16, fitness 221782
      [-1477190758, -738706263], direction -32, fitness 221766
      [-1477190790, -738706263], direction -64, fitness 221734
      [-1477190854, -738706263], direction -128, fitness 221670
      [-1477190982, -738706263], direction -256, fitness 221542
      [-1477191238, -738706263], direction -512, fitness 221286
      [-1477191750, -738706263], direction -1024, fitness 220774
      [-1477192774, -738706263], direction -2048, fitness 219750
      [-1477194822, -738706263], direction -4096, fitness 217702
      [-1477198918, -738706263], direction -8192, fitness 213606
      [-1477207110, -738706263], direction -16384, fitness 205414
      [-1477223494, -738706263], direction -32768, fitness 189030
      [-1477256262, -738706263], direction -65536, fitness 156262
      [-1477321798, -738706263], direction -131072, fitness 90726
      [-1477452870, -738706263], direction -262144, fitness 40346
    Current parameter: 0
    Trying +1 at fitness 40346: [-1477452869, -738706263]
      [-1477452869, -738706263], direction 2, fitness 40345
      [-1477452867, -738706263], direction 4, fitness 40343
      [-1477452863, -738706263], direction 8, fitness 40339
      [-1477452855, -738706263], direction 16, fitness 40331
      [-1477452839, -738706263], direction 32, fitness 40315
      [-1477452807, -738706263], direction 64, fitness 40283
      [-1477452743, -738706263], direction 128, fitness 40219
      [-1477452615, -738706263], direction 256, fitness 40091
      [-1477452359, -738706263], direction 512, fitness 39835
      [-1477451847, -738706263], direction 1024, fitness 39323
      [-1477450823, -738706263], direction 2048, fitness 38299
      [-1477448775, -738706263], direction 4096, fitness 36251
      [-1477444679, -738706263], direction 8192, fitness 32155
      [-1477436487, -738706263], direction 16384, fitness 23963
      [-1477420103, -738706263], direction 32768, fitness 7579
    Current parameter: 0
    Trying +1 at fitness 7579: [-1477420102, -738706263]
      [-1477420102, -738706263], direction 2, fitness 7578
      [-1477420100, -738706263], direction 4, fitness 7576
      [-1477420096, -738706263], direction 8, fitness 7572
      [-1477420088, -738706263], direction 16, fitness 7564
      [-1477420072, -738706263], direction 32, fitness 7548
      [-1477420040, -738706263], direction 64, fitness 7516
      [-1477419976, -738706263], direction 128, fitness 7452
      [-1477419848, -738706263], direction 256, fitness 7324
      [-1477419592, -738706263], direction 512, fitness 7068
      [-1477419080, -738706263], direction 1024, fitness 6556
      [-1477418056, -738706263], direction 2048, fitness 5532
      [-1477416008, -738706263], direction 4096, fitness 3484
      [-1477411912, -738706263], direction 8192, fitness 612
    Current parameter: 0
    Trying +1 at fitness 612: [-1477411911, -738706263]
    Trying -1 at fitness 612: [-1477411913, -738706263]
      [-1477411913, -738706263], direction -2, fitness 611
      [-1477411915, -738706263], direction -4, fitness 609
      [-1477411919, -738706263], direction -8, fitness 605
      [-1477411927, -738706263], direction -16, fitness 597
      [-1477411943, -738706263], direction -32, fitness 581
      [-1477411975, -738706263], direction -64, fitness 549
      [-1477412039, -738706263], direction -128, fitness 485
      [-1477412167, -738706263], direction -256, fitness 357
      [-1477412423, -738706263], direction -512, fitness 101
    Current parameter: 0
    Trying +1 at fitness 101: [-1477412422, -738706263]
    Trying -1 at fitness 101: [-1477412424, -738706263]
      [-1477412424, -738706263], direction -2, fitness 100
      [-1477412426, -738706263], direction -4, fitness 98
      [-1477412430, -738706263], direction -8, fitness 94
      [-1477412438, -738706263], direction -16, fitness 86
      [-1477412454, -738706263], direction -32, fitness 70
      [-1477412486, -738706263], direction -64, fitness 38
      [-1477412550, -738706263], direction -128, fitness 26
    Current parameter: 0
    Trying +1 at fitness 26: [-1477412549, -738706263]
      [-1477412549, -738706263], direction 2, fitness 25
      [-1477412547, -738706263], direction 4, fitness 23
      [-1477412543, -738706263], direction 8, fitness 19
      [-1477412535, -738706263], direction 16, fitness 11
      [-1477412519, -738706263], direction 32, fitness 5
    Current parameter: 0
    Trying +1 at fitness 5: [-1477412518, -738706263]
    Trying -1 at fitness 5: [-1477412520, -738706263]
      [-1477412520, -738706263], direction -2, fitness 4
      [-1477412522, -738706263], direction -4, fitness 2
    Current parameter: 0
    Trying +1 at fitness 2: [-1477412521, -738706263]
    Trying -1 at fitness 2: [-1477412523, -738706263]
      [-1477412523, -738706263], direction -2, fitness 1
    Current parameter: 0
    Trying +1 at fitness 1: [-1477412522, -738706263]
    Trying -1 at fitness 1: [-1477412524, -738706263]
      [-1477412524, -738706263], direction -2, fitness 0
    Current parameter: 0
    Trying +1 at fitness 0: [-1477412523, -738706263]
    Trying -1 at fitness 0: [-1477412525, -738706263]
    Current parameter: 1
    Trying +1 at fitness 0: [-1477412524, -738706262]
    Trying -1 at fitness 0: [-1477412524, -738706264]
    Solution fitness 0: [-1477412524, -738706263]





    [-1477412524, -738706263]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10c7d6650>]




    
![png](6/output_60_1.png)
    


## Program Instrumentation

Deriving fitness functions is not quite so easy. Of course we could come up with an equation that captures the relation between the sides of the triangle, but then essentially we would need to reproduce the entire program logic again in a function, which certainly does not help generalising to other programs. For example, consider how the fitness function would look like if the comparison was not made on the input parameters, but on values derived through computation within the function under test. Ideally, what we would want is to be able to pick a point in the program and come up with a fitness function automatically that describes how close we are to reaching this point.

There are two central ideas in order to achieve this:

- First, rather than trying to guess how close a program inputs gets to a target statement, we simply _run_ the program with the input and observe how close it actually gets.

- Second, during the execution we keep track of distance estimates like the one we calculated for the `test_me` function whenever we come across conditional statements.

In order to observe what an execution does, we need to *instrument* the program: We add new code immediately before or after the branching condition to keep track of the values observed and calculate the distance using these values.

Let's first consider what is done here conceptually. We first define a global variable in which we will store the distance, so that we can access it after the execution:


```python
distance = 0
```

Now the instrumented version just has to update the global variable immediately before executing the branching condition:


```python
def test_me_instrumented(x, y):
    global distance
    distance = abs(x - 2 * (y + 1))
    if x == 2 * (y + 1):
        return True
    else:
        return False
```

Let's try this out for a couple of example values:


```python
test_me_instrumented(0, 0)
distance
```




    2




```python
test_me_instrumented(22, 10)
distance
```




    0



Using this instrumented version of `test_me()`, we can define a fitness function which simply calculates the distance for the condition being true:


```python
def get_fitness(individual):
    global distance
    test_me_instrumented(*individual)
    fitness = distance
    return fitness
```

Let's try this on some example inputs:


```python
get_fitness([0, 0])
```




    2



When we have reached the target branch, the distance will be 0:


```python
get_fitness([22, 10])
```




    0



When implementing the instrumentation, we need to consider that the branching condition may have side-effects. For example, suppose that the branching condition were `x == 2 * foo(y)`, where `foo()` is a function that takes an integer as input. Naively instrumenting would lead to the following code:

```
    distance = abs(x - 2 * foo(y))
	if x == 2 * foo(y):
	...
```

Thus, the instrumentation would lead to `foo()` being executed *twice*. Suppose `foo()` changes the state of the system (e.g., by printing something, accessing the file system, changing some state variables, etc.), then clearly invoking `foo()` a second time is a bad idea. One way to overcome this problem is to _transform_ the conditions, rather than _adding_ tracing calls. For example, one can create temporary variables that hold the values necessary for the distance calculation and then use these in the branching condition:

```
	tmp1 = x
	tmp2 = 2 * foo(y)
	distance = compute_distance(tmp1, tmp2)
	if tmp1 == tmp2:
	...
```


```python
def evaluate_equals(op1, op2):
    global distance
    distance = abs(op1 - op2)
    if distance == 0:
        return True
    else:
        return False
```

Now the aim would be to transform the program automatically such that it looks like so:


```python
def test_me_instrumented(x, y):
    tmp1 = x
    tmp2 = 2 * (y + 1)    
    if evaluate_equals(tmp1, tmp2):
        return True
    else:
        return False
```

Replacing comparisons automatically is actually quite easy in Python, using the abstract syntax tree (AST) of the program. In the AST, a comparison will typically be a tree node with an operator attribute and two children for the left-hand and right-hand operators. To replace such comparisons with a call to `calculate_distance()` one simply needs to replace the comparison node in the AST with a function call node, and this is what the `BranchTransformer` class does using a NodeTransformer from Python's `ast` module:


```python
import ast
```


```python
class BranchTransformer(ast.NodeTransformer):

    def visit_FunctionDef(self, node):
        node.name = node.name + "_instrumented"
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if not isinstance(node.ops[0], ast.Eq):
            return node

        return ast.Call(func=ast.Name("evaluate_equals", ast.Load()),
                        args=[node.left,
                              node.comparators[0]],
                        keywords=[],
                        starargs=None,
                        kwargs=None)
```

The `BranchTransformer` parses a target Python program using the built-in parser `ast.parse()`, which returns the AST. Python provides an API to traverse and modify this AST. To replace the comparison with a function call we use an `ast.NodeTransformer`, which uses the visitor pattern where there is one `visit_*` function for each type of node in the AST. As we are interested in replacing comparisons, we override `visit_Compare`, where instead of the original comparison node we return a new node of type `ast.Func`, which is a function call node. The first parameter of this node is the name of the function `calculate_distance`, and the arguments are the two operands that our `calculate_distance` function expects.

You will notice that we also override `visit_FunctionDef`; this is just to change the name of the method by appending `_instrumented`, so that we can continue to use the original function together with the instrumented one.

The following code parses the source code of the `test_me()` function to an AST, then transforms it, and prints it out again (using the `to_source()` function from the `astor` library):


```python
import inspect
import ast
import astor
```


```python
source = inspect.getsource(test_me)
node = ast.parse(source)
BranchTransformer().visit(node)

# Make sure the line numbers are ok before printing
node = ast.fix_missing_locations(node)
print(astor.to_source(node))
```

    def test_me_instrumented(x, y):
        if evaluate_equals(x, 2 * (y + 1)):
            return True
        else:
            return False
    


To calculate a fitness value with the instrumented version, we need to compile the instrumented AST again, which is done using Python's `compile()` function. We then need to make the compiled function accessible, for which we first retrieve the current module from `sys.modules`, and then add the compiled code of the instrumented function to the list of functions of the current module using `exec`. After this, the `cgi_decode_instrumented()` function can be accessed.


```python
import sys
```


```python
def create_instrumented_function(f):
    source = inspect.getsource(f)
    node = ast.parse(source)
    node = BranchTransformer().visit(node)

    # Make sure the line numbers are ok so that it compiles
    node = ast.fix_missing_locations(node)

    # Compile and add the instrumented function to the current module
    current_module = sys.modules[__name__]
    code = compile(node, filename="<ast>", mode="exec")
    exec(code, current_module.__dict__)
```


```python
create_instrumented_function(test_me)
```


```python
test_me_instrumented(0, 0)
```




    False




```python
distance
```




    2




```python
test_me_instrumented(22, 10)
```




    True




```python
distance
```




    0



The estimate for any relational comparison of two values is defined in terms of the _branch distance_. Our `evaluate_equals` function indeed implements the branch distance function for an equality comparison. To generalise this we need similar estimates for other types of relational comparisons. Furthermore, we also have to consider the distance to such conditions evaluating to false, not just to true. Thus, each if-condition actually has two distance estimates, one to estimate how close it is to being true, and one how close it is to being false. If the condition is true, then the true distance is 0; if the condition is false, then the false distance is 0. That is, in a comparison `a == b`, if `a` is smaller than `b`, then the false distance is `0` by definition. 

The following table shows how to calculate the distance for different types of comparisons:

| Condition | Distance True | Distance False |
| ------------- |:-------------:| -----:|
| a == b      | abs(a - b) | 1 |
| a != b      | 1          | abs(a - b) |
| a < b       | b - a + 1  | a - b      |
| a <= b      | b - a      | a - b + 1  |
| a > b       | a - b + 1  | b - a      |


Note that several of the calculations add a constant `1`. The reason for this is quite simple: Suppose we want to have `a < b` evaluate to true, and let `a = 27` and `b = 27`. The condition is not true, but simply taking the difference would give us a result of `0`. To avoid this, we have to add a constant value. It is not important whether this value is `1` -- any positive constant works.

We generalise our `evaluate_equals` function to an `evaluate_condition` function that takes the operator as an additional parameter, and then implements the above table. In contrast to the previous `calculate_equals`, we will now calculate both, the true and the false distance:


```python
def evaluate_condition(op, lhs, rhs):
    distance_true = 0
    distance_false = 0
    if op == "Eq":
        if lhs == rhs:
            distance_false = 1
        else:
            distance_true = abs(lhs - rhs)

    # ... code for other types of conditions

    if distance_true == 0:
        return True
    else:
        return False
```

Let's consider a slightly larger function under test. We will use the well known triangle example, originating in Glenford Meyer's classical Art of Software Testing book 


```python
def triangle(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return 4 # invalid
    
    if a + b <= c or a + c <= b or b + c <= a:
        return 4 # invalid
    
    if a == b and b == c:
        return 1 # equilateral
    
    if a == b or b == c or a == c:
        return 2 # isosceles
    
    return 3 # scalene
```

The function takes as input the length of the three sides of a triangle, and returns a number representing the type of triangle:


```python
triangle(4,4,4)
```




    1



Adapting our representation is easy, we just need to correctly set the number of parameters:


```python
sig = signature(triangle)
num_parameters = len(sig.parameters)
num_parameters
```




    3



For the `triangle` function, however, we have multiple if-conditions; we have to add instrumentation to each of these using `evaluate_condition`. We also need to generalise from our global `distance` variable, since we now have two distance values per branch, and potentially multiple branches. Furthermore, a condition might be executed multiple times within a single execution (e.g., if it is in a loop), so rather than storing all values, we will only keep the _minimum_ value observed for each condition:


```python
distances_true = {}
distances_false = {}
```


```python
def update_maps(condition_num, d_true, d_false):
    global distances_true, distances_false

    if condition_num in distances_true.keys():
        distances_true[condition_num] = min(distances_true[condition_num], d_true)
    else:
        distances_true[condition_num] = d_true

    if condition_num in distances_false.keys():
        distances_false[condition_num] = min(distances_false[condition_num], d_false)
    else:
        distances_false[condition_num] = d_false
```

Now we need to finish implementing the `evaluate_condition` function. We add yet another parameter to denote the ID of the branch we are instrumenting:


```python
def evaluate_condition(num, op, lhs, rhs):
    distance_true = 0
    distance_false = 0

    # Make sure the distance can be calculated on number and character
    # comparisons (needed for cgi_decode later)
    if isinstance(lhs, str):
        lhs = ord(lhs)
    if isinstance(rhs, str):
        rhs = ord(rhs)

    if op == "Eq":
        if lhs == rhs:
            distance_false = 1
        else:
            distance_true = abs(lhs - rhs)

    elif op == "Gt":
        if lhs > rhs:
            distance_false = lhs - rhs
        else:
            distance_true = rhs - lhs + 1
    elif op == "Lt":
        if lhs < rhs:
            distance_false = rhs - lhs
        else:
            distance_true = lhs - rhs + 1
    elif op == "LtE":
        if lhs <= rhs:
            distance_false = rhs - lhs + 1
        else:
            distance_true = lhs - rhs
    # ...
    # handle other comparison operators
    # ...

    elif op == "In":
        minimum = sys.maxsize
        for elem in rhs.keys():
            distance = abs(lhs - ord(elem))
            if distance < minimum:
                minimum = distance

        distance_true = minimum
        if distance_true == 0:
            distance_false = 1
    else:
        assert False

    update_maps(num, normalise(distance_true), normalise(distance_false))

    if distance_true == 0:
        return True
    else:
        return False
```

We need to normalise branch distances since different comparisons will be on different scales, and this would bias the search. We will use the normalisaction function defined in the previous chapter:


```python
def normalise(x):
    return x / (1.0 + x)
```

We also need to extend our instrumentation function to take care of all comparisons, and not just equality comparisons:


```python
import ast
class BranchTransformer(ast.NodeTransformer):

    branch_num = 0

    def visit_FunctionDef(self, node):
        node.name = node.name + "_instrumented"
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if node.ops[0] in [ast.Is, ast.IsNot, ast.In, ast.NotIn]:
            return node

        self.branch_num += 1
        return ast.Call(func=ast.Name("evaluate_condition", ast.Load()),
                        args=[ast.Num(self.branch_num - 1),
                              ast.Str(node.ops[0].__class__.__name__),
                              node.left,
                              node.comparators[0]],
                        keywords=[],
                        starargs=None,
                        kwargs=None)
```

We can now take a look at the instrumented version of `triangle`:


```python
source = inspect.getsource(triangle)
node = ast.parse(source)
transformer = BranchTransformer()
transformer.visit(node)

# Make sure the line numbers are ok before printing
node = ast.fix_missing_locations(node)
num_branches = transformer.branch_num

print(astor.to_source(node))
```

    def triangle_instrumented(a, b, c):
        if evaluate_condition(0, 'LtE', a, 0) or evaluate_condition(1, 'LtE', b, 0
            ) or evaluate_condition(2, 'LtE', c, 0):
            return 4
        if evaluate_condition(3, 'LtE', a + b, c) or evaluate_condition(4,
            'LtE', a + c, b) or evaluate_condition(5, 'LtE', b + c, a):
            return 4
        if evaluate_condition(6, 'Eq', a, b) and evaluate_condition(7, 'Eq', b, c):
            return 1
        if evaluate_condition(8, 'Eq', a, b) or evaluate_condition(9, 'Eq', b, c
            ) or evaluate_condition(10, 'Eq', a, c):
            return 2
        return 3
    


To define an executable version of the instrumented triangle function, we can use our `create_instrumented_function` function again:


```python
create_instrumented_function(triangle)
```


```python
triangle_instrumented(4, 4, 4)
```




    1




```python
distances_true
```




    {0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8, 5: 0.8, 6: 0.0, 7: 0.0}




```python
distances_false
```




    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.5, 7: 0.5}



The branch distance functions above are defined only for atomic comparisons. However, in the `triangle` program all of the atomic comparisons are part of larger predicates, joined together by `and` and `or` connectors. 

For conjunctions the branch distance is defined such that the distance to make `A and B` true equals the sum of the branch distances for `A` and `B`, as both of the two conditions would need to be true. Similarly, the branch distance to make `A or B` true would be the _minimum_ of the two branch distances of `A` and `B`, as it suffices if one of the two conditions is true to make the entire expression true (and the false distance would be the sum of false distances of the conditions). For a negation `not A`, we can simply switch from the true distance to the false distance, or vice versa. Since predicates can consist of nested conditions, one would need to recursively calculate the branch distance.


Assume we want to find an input that covers the third if-condition, i.e., it produces a triangle where all sides have equal length. Considering the instrumented version of the triangle function we printed above, in order for this if-condition to evaluate to true we require conditions 0, 1, 2, 3, 4, and 5 to evaluate to false, and 6 and 7 to evaluate to true. Thus, the fitness function for this branch would be the sum of false distances for branches 0-5, and true distances for branches 6 and 7.


```python
def get_fitness(x):
    # Reset any distance values from previous executions
    global distances_true, distances_false
    distances_true  = {x: 1.0 for x in range(num_branches)}
    distances_false = {x: 1.0 for x in range(num_branches)}

    # Run the function under test
    triangle_instrumented(*x)

    # Sum up branch distances for our specific target branch
    fitness = 0.0
    for branch in [6, 7]:
        fitness += distances_true[branch]

    for branch in [0, 1, 2, 3, 4, 5]:
        fitness += distances_false[branch]

    return fitness
```


```python
get_fitness([5,5,5])
```




    0.0




```python
get_fitness(get_random_individual())
```




    1.9999999994312394




```python
MAX = 10000
MIN = -MAX
fitness_values = []
max_gen = 1000
hillclimbing()
```

    Starting at fitness 7.99009900990099: [-99, 1534, -8014]
    Solution fitness after 10002 fitness evaluations: 5.9998449371995655: [1, 1534, -6447]





    [1, 1534, -6447]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10d32cd00>]




    
![png](6/output_129_1.png)
    



```python
fitness_values = []
avm()
```

    Starting at fitness 6.999708114419148: [6710, -3424, -6516]
    Current parameter: 0
    Trying +1 at fitness 6.999708114419148: [6711, -3424, -6516]
    Trying -1 at fitness 6.999708114419148: [6709, -3424, -6516]
    Current parameter: 1
    Trying +1 at fitness 6.999708114419148: [6710, -3423, -6516]
      [6710, -3423, -6516], direction 2, fitness 6.99970802919708
      [6710, -3421, -6516], direction 4, fitness 6.999707858603564
      [6710, -3417, -6516], direction 8, fitness 6.999707516817783
      [6710, -3409, -6516], direction 16, fitness 6.999706830841395
      [6710, -3393, -6516], direction 32, fitness 6.999705449189985
      [6710, -3361, -6516], direction 64, fitness 6.999702646446625
      [6710, -3297, -6516], direction 128, fitness 6.99969687784177
      [6710, -3169, -6516], direction 256, fitness 6.999684642068749
      [6710, -2913, -6516], direction 512, fitness 6.999656946826758
      [6710, -2401, -6516], direction 1024, fitness 6.999583853516437
      [6710, -1377, -6516], direction 2048, fitness 6.999274836838289
      [6710, 671, -6516], direction 4096, fitness 5.9998465787051245
    Current parameter: 1
    Trying +1 at fitness 5.9998465787051245: [6710, 672, -6516]
    Trying -1 at fitness 5.9998465787051245: [6710, 670, -6516]
    Current parameter: 2
    Trying +1 at fitness 5.9998465787051245: [6710, 671, -6515]
      [6710, 671, -6515], direction 2, fitness 5.999846555163419
      [6710, 671, -6513], direction 4, fitness 5.999846508058327
      [6710, 671, -6509], direction 8, fitness 5.999846413761327
      [6710, 671, -6501], direction 16, fitness 5.999846224819314
      [6710, 671, -6485], direction 32, fitness 5.999845845537228
      [6710, 671, -6453], direction 64, fitness 5.9998450813323005
      [6710, 671, -6389], direction 128, fitness 5.9998435299640125
      [6710, 671, -6261], direction 256, fitness 5.999840332109213
      [6710, 671, -6005], direction 512, fitness 5.99983352755119
      [6710, 671, -5493], direction 1024, fitness 5.999818016378526
      [6710, 671, -4469], direction 2048, fitness 5.999776336390069
      [6710, 671, -2421], direction 4096, fitness 5.9995872884853485
      [6710, 671, 1675], direction 8192, fitness 2.999770957398076
    Current parameter: 2
    Trying +1 at fitness 2.999770957398076: [6710, 671, 1676]
      [6710, 671, 1676], direction 2, fitness 2.999770904925544
      [6710, 671, 1678], direction 4, fitness 2.99977079990832
      [6710, 671, 1682], direction 8, fitness 2.999770589584767
      [6710, 671, 1690], direction 16, fitness 2.9997701677775224
      [6710, 671, 1706], direction 32, fitness 2.999769319492503
      [6710, 671, 1738], direction 64, fitness 2.999767603997211
      [6710, 671, 1802], direction 128, fitness 2.9997640953054967
      [6710, 671, 1930], direction 256, fitness 2.9997567501824376
      [6710, 671, 2186], direction 512, fitness 2.999740596627756
      [6710, 671, 2698], direction 1024, fitness 2.9997008674842958
      [6710, 671, 3722], direction 2048, fitness 2.9995687796463995
      [6710, 671, 5770], direction 4096, fitness 2.9963099630996313
    Current parameter: 2
    Trying +1 at fitness 2.9963099630996313: [6710, 671, 5771]
      [6710, 671, 5771], direction 2, fitness 2.9962962962962965
      [6710, 671, 5773], direction 4, fitness 2.996268656716418
      [6710, 671, 5777], direction 8, fitness 2.996212121212121
      [6710, 671, 5785], direction 16, fitness 2.99609375
      [6710, 671, 5801], direction 32, fitness 2.9958333333333336
      [6710, 671, 5833], direction 64, fitness 2.9951923076923075
      [6710, 671, 5897], direction 128, fitness 2.9930555555555554
      [6710, 671, 6025], direction 256, fitness 2.9375
      [6710, 671, 6281], direction 512, fitness 1.9998344370860928
    Current parameter: 2
    Trying +1 at fitness 1.9998344370860928: [6710, 671, 6282]
    Trying -1 at fitness 1.9998344370860928: [6710, 671, 6280]
    Current parameter: 0
    Trying +1 at fitness 1.9998344370860928: [6711, 671, 6281]
    Trying -1 at fitness 1.9998344370860928: [6709, 671, 6281]
      [6709, 671, 6281], direction -2, fitness 1.9998344096704752
      [6707, 671, 6281], direction -4, fitness 1.9998343548119926
      [6703, 671, 6281], direction -8, fitness 1.9998342449859108
      [6695, 671, 6281], direction -16, fitness 1.9998340248962656
      [6679, 671, 6281], direction -32, fitness 1.9998335829588951
      [6647, 671, 6281], direction -64, fitness 1.9998326919859462
      [6583, 671, 6281], direction -128, fitness 1.9998308811094199
      [6455, 671, 6281], direction -256, fitness 1.9998271391529818
      [6199, 671, 6281], direction -512, fitness 1.999819135467535
      [5687, 671, 6281], direction -1024, fitness 1.9998006776958341
    Current parameter: 0
    Trying +1 at fitness 1.9998006776958341: [5688, 671, 6281]
    Trying -1 at fitness 1.9998006776958341: [5686, 671, 6281]
      [5686, 671, 6281], direction -2, fitness 1.9998006379585327
      [5684, 671, 6281], direction -4, fitness 1.999800558436378
      [5680, 671, 6281], direction -8, fitness 1.9998003992015967
      [5672, 671, 6281], direction -16, fitness 1.9998000799680127
      [5656, 671, 6281], direction -32, fitness 1.9997994384275972
      [5624, 671, 6281], direction -64, fitness 1.9997981429148162
    Current parameter: 0
    Trying +1 at fitness 1.9997981429148162: [5625, 671, 6281]
    Trying -1 at fitness 1.9997981429148162: [5623, 671, 6281]
      [5623, 671, 6281], direction -2, fitness 1.9997981021603068
      [5621, 671, 6281], direction -4, fitness 1.9997980206018986
      [5617, 671, 6281], direction -8, fitness 1.9997978572872448
    Current parameter: 0
    Trying +1 at fitness 1.9997978572872448: [5618, 671, 6281]
    Trying -1 at fitness 1.9997978572872448: [5616, 671, 6281]
      [5616, 671, 6281], direction -2, fitness 1.999797816417307
      [5614, 671, 6281], direction -4, fitness 1.9997977346278317
    Current parameter: 0
    Trying +1 at fitness 1.9997977346278317: [5615, 671, 6281]
    Trying -1 at fitness 1.9997977346278317: [5613, 671, 6281]
      [5613, 671, 6281], direction -2, fitness 1.9997976937082744
      [5611, 671, 6281], direction -4, fitness 1.9997976118194698
    Current parameter: 0
    Trying +1 at fitness 1.9997976118194698: [5612, 671, 6281]
    Trying -1 at fitness 1.9997976118194698: [5610, 671, 6281]
    Current parameter: 1
    Trying +1 at fitness 1.9997976118194698: [5611, 672, 6281]
      [5611, 672, 6281], direction 2, fitness 1.9997975708502023
      [5611, 674, 6281], direction 4, fitness 1.9997974888618875
      [5611, 678, 6281], direction 8, fitness 1.9997973246858534
      [5611, 686, 6281], direction 16, fitness 1.9997969955339019
      [5611, 702, 6281], direction 32, fitness 1.99979633401222
      [5611, 734, 6281], direction 64, fitness 1.9997949979499796
      [5611, 798, 6281], direction 128, fitness 1.9997922725384296
      [5611, 926, 6281], direction 256, fitness 1.9997865983781478
      [5611, 1182, 6281], direction 512, fitness 1.9997742663656886
      [5611, 1694, 6281], direction 1024, fitness 1.9997447677386422
      [5611, 2718, 6281], direction 2048, fitness 1.9996544574982722
      [5611, 4766, 6281], direction 4096, fitness 1.9988179669030735
    Current parameter: 1
    Trying +1 at fitness 1.9988179669030735: [5611, 4767, 6281]
      [5611, 4767, 6281], direction 2, fitness 1.9988165680473373
      [5611, 4769, 6281], direction 4, fitness 1.9988137603795968
      [5611, 4773, 6281], direction 8, fitness 1.9988081048867699
      [5611, 4781, 6281], direction 16, fitness 1.9987966305655838
      [5611, 4797, 6281], direction 32, fitness 1.9987730061349693
      [5611, 4829, 6281], direction 64, fitness 1.9987228607918262
      [5611, 4893, 6281], direction 128, fitness 1.9986091794158554
      [5611, 5021, 6281], direction 256, fitness 1.9983079526226735
      [5611, 5277, 6281], direction 512, fitness 1.9970149253731342
      [5611, 5789, 6281], direction 1024, fitness 1.994413407821229
    Current parameter: 1
    Trying +1 at fitness 1.994413407821229: [5611, 5790, 6281]
    Trying -1 at fitness 1.994413407821229: [5611, 5788, 6281]
      [5611, 5788, 6281], direction -2, fitness 1.99438202247191
      [5611, 5786, 6281], direction -4, fitness 1.9943181818181817
      [5611, 5782, 6281], direction -8, fitness 1.994186046511628
      [5611, 5774, 6281], direction -16, fitness 1.9939024390243902
      [5611, 5758, 6281], direction -32, fitness 1.9932432432432432
      [5611, 5726, 6281], direction -64, fitness 1.9913793103448276
      [5611, 5662, 6281], direction -128, fitness 1.9807692307692308
    Current parameter: 1
    Trying +1 at fitness 1.9807692307692308: [5611, 5663, 6281]
    Trying -1 at fitness 1.9807692307692308: [5611, 5661, 6281]
      [5611, 5661, 6281], direction -2, fitness 1.9803921568627452
      [5611, 5659, 6281], direction -4, fitness 1.9795918367346939
      [5611, 5655, 6281], direction -8, fitness 1.9777777777777779
      [5611, 5647, 6281], direction -16, fitness 1.972972972972973
      [5611, 5631, 6281], direction -32, fitness 1.9523809523809523
      [5611, 5599, 6281], direction -64, fitness 1.9230769230769231
    Current parameter: 1
    Trying +1 at fitness 1.9230769230769231: [5611, 5600, 6281]
      [5611, 5600, 6281], direction 2, fitness 1.9166666666666665
      [5611, 5602, 6281], direction 4, fitness 1.9
      [5611, 5606, 6281], direction 8, fitness 1.8333333333333335
      [5611, 5614, 6281], direction 16, fitness 1.75
    Current parameter: 1
    Trying +1 at fitness 1.75: [5611, 5615, 6281]
    Trying -1 at fitness 1.75: [5611, 5613, 6281]
      [5611, 5613, 6281], direction -2, fitness 1.6666666666666665
      [5611, 5611, 6281], direction -4, fitness 0.9985096870342772
    Current parameter: 1
    Trying +1 at fitness 0.9985096870342772: [5611, 5612, 6281]
    Trying -1 at fitness 0.9985096870342772: [5611, 5610, 6281]
    Current parameter: 2
    Trying +1 at fitness 0.9985096870342772: [5611, 5611, 6282]
    Trying -1 at fitness 0.9985096870342772: [5611, 5611, 6280]
      [5611, 5611, 6280], direction -2, fitness 0.9985074626865672
      [5611, 5611, 6278], direction -4, fitness 0.9985029940119761
      [5611, 5611, 6274], direction -8, fitness 0.9984939759036144
      [5611, 5611, 6266], direction -16, fitness 0.9984756097560976
      [5611, 5611, 6250], direction -32, fitness 0.9984375
      [5611, 5611, 6218], direction -64, fitness 0.9983552631578947
      [5611, 5611, 6154], direction -128, fitness 0.9981617647058824
      [5611, 5611, 6026], direction -256, fitness 0.9975961538461539
      [5611, 5611, 5770], direction -512, fitness 0.99375
    Current parameter: 2
    Trying +1 at fitness 0.99375: [5611, 5611, 5771]
    Trying -1 at fitness 0.99375: [5611, 5611, 5769]
      [5611, 5611, 5769], direction -2, fitness 0.9937106918238994
      [5611, 5611, 5767], direction -4, fitness 0.9936305732484076
      [5611, 5611, 5763], direction -8, fitness 0.9934640522875817
      [5611, 5611, 5755], direction -16, fitness 0.993103448275862
      [5611, 5611, 5739], direction -32, fitness 0.9922480620155039
      [5611, 5611, 5707], direction -64, fitness 0.9896907216494846
      [5611, 5611, 5643], direction -128, fitness 0.9696969696969697
    Current parameter: 2
    Trying +1 at fitness 0.9696969696969697: [5611, 5611, 5644]
    Trying -1 at fitness 0.9696969696969697: [5611, 5611, 5642]
      [5611, 5611, 5642], direction -2, fitness 0.96875
      [5611, 5611, 5640], direction -4, fitness 0.9666666666666667
      [5611, 5611, 5636], direction -8, fitness 0.9615384615384616
      [5611, 5611, 5628], direction -16, fitness 0.9444444444444444
      [5611, 5611, 5612], direction -32, fitness 0.5
    Current parameter: 2
    Trying +1 at fitness 0.5: [5611, 5611, 5613]
    Trying -1 at fitness 0.5: [5611, 5611, 5611]
      [5611, 5611, 5611], direction -2, fitness 0.0
    Current parameter: 2
    Trying +1 at fitness 0.0: [5611, 5611, 5612]
    Trying -1 at fitness 0.0: [5611, 5611, 5610]
    Solution fitness 0.0: [5611, 5611, 5611]





    [5611, 5611, 5611]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10d379390>]




    
![png](6/output_131_1.png)
    


Besides the local search algorithms, we can also use evolutionary search in order to find solutions to our test generation problem. We therefore need to define the usual search operators:


```python
tournament_size = 3
def tournament_selection(population):
    candidates = random.sample(population, tournament_size)        
    winner = min(candidates, key = lambda x: get_fitness(x))
    return winner
```


```python
elite_size = 2
def elitism_standard(population):
    population.sort(key = lambda k: get_fitness(k))
    return population[:elite_size]
```


```python
def mutate(solution):
    P_mutate = 1/len(solution)
    mutated = solution[:]
    for position in range(len(solution)):
        if random.random() < P_mutate:
            mutated[position] = int(random.gauss(mutated[position], 20))
    return mutated
```


```python
def singlepoint_crossover(parent1, parent2):
    pos = random.randint(0, len(parent1))
    offspring1 = parent1[:pos] + parent2[pos:]
    offspring2 = parent2[:pos] + parent1[pos:]
    return (offspring1, offspring2)
```


```python
population_size = 20
P_xover = 0.7
max_gen = 100
selection = tournament_selection
crossover = singlepoint_crossover
elitism = elitism_standard
MAX = 1000
MIN = -MAX
```


```python
def ga():
    population = [get_random_individual() for _ in range(population_size)]
    best_fitness = sys.maxsize
    for p in population:
        fitness = get_fitness(p)
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = p
    print(f"Iteration 0, best fitness: {best_fitness}")

    for iteration in range(max_gen):
        fitness_values.append(best_fitness)
        new_population = elitism(population)
        while len(new_population) < len(population):
            parent1 = selection(population)
            parent2 = selection(population)

            if random.random() < P_xover:
                offspring1, offspring2 = crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1, parent2

            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            
            new_population.append(offspring1)
            new_population.append(offspring2)

        population = new_population
        for p in population:
            fitness = get_fitness(p)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = p
        print(f"Iteration {iteration}, best fitness: {best_fitness}, size {len(best_solution)}")

    return best_solution
```


```python
fitness_values = []
ga()
```

    Iteration 0, best fitness: 1.9974424552429668
    Iteration 0, best fitness: 1.9850746268656716, size 3
    Iteration 1, best fitness: 1.981818181818182, size 3
    Iteration 2, best fitness: 1.973684210526316, size 3
    Iteration 3, best fitness: 1.9375, size 3
    Iteration 4, best fitness: 1.5, size 3
    Iteration 5, best fitness: 1.5, size 3
    Iteration 6, best fitness: 0.9807692307692307, size 3
    Iteration 7, best fitness: 0.9807692307692307, size 3
    Iteration 8, best fitness: 0.9807692307692307, size 3
    Iteration 9, best fitness: 0.9807692307692307, size 3
    Iteration 10, best fitness: 0.9807692307692307, size 3
    Iteration 11, best fitness: 0.9736842105263158, size 3
    Iteration 12, best fitness: 0.96, size 3
    Iteration 13, best fitness: 0.96, size 3
    Iteration 14, best fitness: 0.9545454545454546, size 3
    Iteration 15, best fitness: 0.6666666666666666, size 3
    Iteration 16, best fitness: 0.5, size 3
    Iteration 17, best fitness: 0.5, size 3
    Iteration 18, best fitness: 0.5, size 3
    Iteration 19, best fitness: 0.5, size 3
    Iteration 20, best fitness: 0.5, size 3
    Iteration 21, best fitness: 0.5, size 3
    Iteration 22, best fitness: 0.5, size 3
    Iteration 23, best fitness: 0.5, size 3
    Iteration 24, best fitness: 0.5, size 3
    Iteration 25, best fitness: 0.5, size 3
    Iteration 26, best fitness: 0.5, size 3
    Iteration 27, best fitness: 0.5, size 3
    Iteration 28, best fitness: 0.5, size 3
    Iteration 29, best fitness: 0.5, size 3
    Iteration 30, best fitness: 0.5, size 3
    Iteration 31, best fitness: 0.5, size 3
    Iteration 32, best fitness: 0.5, size 3
    Iteration 33, best fitness: 0.5, size 3
    Iteration 34, best fitness: 0.5, size 3
    Iteration 35, best fitness: 0.5, size 3
    Iteration 36, best fitness: 0.5, size 3
    Iteration 37, best fitness: 0.5, size 3
    Iteration 38, best fitness: 0.0, size 3
    Iteration 39, best fitness: 0.0, size 3
    Iteration 40, best fitness: 0.0, size 3
    Iteration 41, best fitness: 0.0, size 3
    Iteration 42, best fitness: 0.0, size 3
    Iteration 43, best fitness: 0.0, size 3
    Iteration 44, best fitness: 0.0, size 3
    Iteration 45, best fitness: 0.0, size 3
    Iteration 46, best fitness: 0.0, size 3
    Iteration 47, best fitness: 0.0, size 3
    Iteration 48, best fitness: 0.0, size 3
    Iteration 49, best fitness: 0.0, size 3
    Iteration 50, best fitness: 0.0, size 3
    Iteration 51, best fitness: 0.0, size 3
    Iteration 52, best fitness: 0.0, size 3
    Iteration 53, best fitness: 0.0, size 3
    Iteration 54, best fitness: 0.0, size 3
    Iteration 55, best fitness: 0.0, size 3
    Iteration 56, best fitness: 0.0, size 3
    Iteration 57, best fitness: 0.0, size 3
    Iteration 58, best fitness: 0.0, size 3
    Iteration 59, best fitness: 0.0, size 3
    Iteration 60, best fitness: 0.0, size 3
    Iteration 61, best fitness: 0.0, size 3
    Iteration 62, best fitness: 0.0, size 3
    Iteration 63, best fitness: 0.0, size 3
    Iteration 64, best fitness: 0.0, size 3
    Iteration 65, best fitness: 0.0, size 3
    Iteration 66, best fitness: 0.0, size 3
    Iteration 67, best fitness: 0.0, size 3
    Iteration 68, best fitness: 0.0, size 3
    Iteration 69, best fitness: 0.0, size 3
    Iteration 70, best fitness: 0.0, size 3
    Iteration 71, best fitness: 0.0, size 3
    Iteration 72, best fitness: 0.0, size 3
    Iteration 73, best fitness: 0.0, size 3
    Iteration 74, best fitness: 0.0, size 3
    Iteration 75, best fitness: 0.0, size 3
    Iteration 76, best fitness: 0.0, size 3
    Iteration 77, best fitness: 0.0, size 3
    Iteration 78, best fitness: 0.0, size 3
    Iteration 79, best fitness: 0.0, size 3
    Iteration 80, best fitness: 0.0, size 3
    Iteration 81, best fitness: 0.0, size 3
    Iteration 82, best fitness: 0.0, size 3
    Iteration 83, best fitness: 0.0, size 3
    Iteration 84, best fitness: 0.0, size 3
    Iteration 85, best fitness: 0.0, size 3
    Iteration 86, best fitness: 0.0, size 3
    Iteration 87, best fitness: 0.0, size 3
    Iteration 88, best fitness: 0.0, size 3
    Iteration 89, best fitness: 0.0, size 3
    Iteration 90, best fitness: 0.0, size 3
    Iteration 91, best fitness: 0.0, size 3
    Iteration 92, best fitness: 0.0, size 3
    Iteration 93, best fitness: 0.0, size 3
    Iteration 94, best fitness: 0.0, size 3
    Iteration 95, best fitness: 0.0, size 3
    Iteration 96, best fitness: 0.0, size 3
    Iteration 97, best fitness: 0.0, size 3
    Iteration 98, best fitness: 0.0, size 3
    Iteration 99, best fitness: 0.0, size 3





    [834, 834, 834]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10d3b61a0>]




    
![png](6/output_140_1.png)
    


We set `MAX` to a value as low as 1000, because the optimisation with our small mutational steps may take long to achieve that multiple values are equal, which some of the branches of the triangle program require (such as the one we are optimising for currently).


```python
MAX = 100000
MIN = -MAX
fitness_values = []
ga()
plt.plot(fitness_values)
```

    Iteration 0, best fitness: 1.9999718381255458
    Iteration 0, best fitness: 1.9999718301923997, size 3
    Iteration 1, best fitness: 1.9999428669370967, size 3
    Iteration 2, best fitness: 1.9999428669370967, size 3
    Iteration 3, best fitness: 1.9999427295114827, size 3
    Iteration 4, best fitness: 1.9999426769848094, size 3
    Iteration 5, best fitness: 1.9999425848309125, size 3
    Iteration 6, best fitness: 1.9999425485464783, size 3
    Iteration 7, best fitness: 1.9999424261615522, size 3
    Iteration 8, best fitness: 1.9999423398489304, size 3
    Iteration 9, best fitness: 1.9999422766104824, size 3
    Iteration 10, best fitness: 1.9999422766104824, size 3
    Iteration 11, best fitness: 1.9999422199110186, size 3
    Iteration 12, best fitness: 1.99994209612044, size 3
    Iteration 13, best fitness: 1.9999419886297716, size 3
    Iteration 14, best fitness: 1.9999419077495062, size 3
    Iteration 15, best fitness: 1.9999418841169292, size 3
    Iteration 16, best fitness: 1.9999418063314711, size 3
    Iteration 17, best fitness: 1.999941724941725, size 3
    Iteration 18, best fitness: 1.9999416058394162, size 3
    Iteration 19, best fitness: 1.9999414348462665, size 3
    Iteration 20, best fitness: 1.9999413799167596, size 3
    Iteration 21, best fitness: 1.9999413386519622, size 3
    Iteration 22, best fitness: 1.9999413386519622, size 3
    Iteration 23, best fitness: 1.9999412455934196, size 3
    Iteration 24, best fitness: 1.9999410585877637, size 3
    Iteration 25, best fitness: 1.9999410585877637, size 3
    Iteration 26, best fitness: 1.9999409611524384, size 3
    Iteration 27, best fitness: 1.999940807387238, size 3
    Iteration 28, best fitness: 1.999940807387238, size 3
    Iteration 29, best fitness: 1.9999407056033205, size 3
    Iteration 30, best fitness: 1.9999406633833738, size 3
    Iteration 31, best fitness: 1.9999406633833738, size 3
    Iteration 32, best fitness: 1.9999406528189911, size 3
    Iteration 33, best fitness: 1.9999405363620146, size 3
    Iteration 34, best fitness: 1.99994048681783, size 3
    Iteration 35, best fitness: 1.9999404513785506, size 3
    Iteration 36, best fitness: 1.9999403945878287, size 3
    Iteration 37, best fitness: 1.9999402806808002, size 3
    Iteration 38, best fitness: 1.9999402485659656, size 3
    Iteration 39, best fitness: 1.9999401734968592, size 3
    Iteration 40, best fitness: 1.9999401699174344, size 3
    Iteration 41, best fitness: 1.9999401663375815, size 3
    Iteration 42, best fitness: 1.9999399435469343, size 3
    Iteration 43, best fitness: 1.99993978805395, size 3
    Iteration 44, best fitness: 1.99993978805395, size 3
    Iteration 45, best fitness: 1.9999397517773225, size 3
    Iteration 46, best fitness: 1.9999396353978027, size 3
    Iteration 47, best fitness: 1.9999395916394829, size 3
    Iteration 48, best fitness: 1.9999394856278365, size 3
    Iteration 49, best fitness: 1.9999394159699504, size 3
    Iteration 50, best fitness: 1.9999393424724008, size 3
    Iteration 51, best fitness: 1.9999391542439915, size 3
    Iteration 52, best fitness: 1.9999390875312177, size 3
    Iteration 53, best fitness: 1.9999389312977098, size 3
    Iteration 54, best fitness: 1.9999388491408303, size 3
    Iteration 55, best fitness: 1.9999387592626614, size 3
    Iteration 56, best fitness: 1.999938612645795, size 3
    Iteration 57, best fitness: 1.9999384274367342, size 3
    Iteration 58, best fitness: 1.9999384274367342, size 3
    Iteration 59, best fitness: 1.9999383134908395, size 3
    Iteration 60, best fitness: 1.9999381188118812, size 3
    Iteration 61, best fitness: 1.9999381188118812, size 3
    Iteration 62, best fitness: 1.9999380344528443, size 3
    Iteration 63, best fitness: 1.9999379190464366, size 3
    Iteration 64, best fitness: 1.9999379190464366, size 3
    Iteration 65, best fitness: 1.9999377877317408, size 3
    Iteration 66, best fitness: 1.9999377179870454, size 3
    Iteration 67, best fitness: 1.9999375702334872, size 3
    Iteration 68, best fitness: 1.9999375546396903, size 3
    Iteration 69, best fitness: 1.9999374296083094, size 3
    Iteration 70, best fitness: 1.9999374296083094, size 3
    Iteration 71, best fitness: 1.9999373158653544, size 3
    Iteration 72, best fitness: 1.9999372883481752, size 3
    Iteration 73, best fitness: 1.9999371780374418, size 3
    Iteration 74, best fitness: 1.999937150399095, size 3
    Iteration 75, best fitness: 1.9999370633771791, size 3
    Iteration 76, best fitness: 1.9999369800857072, size 3
    Iteration 77, best fitness: 1.9999369800857072, size 3
    Iteration 78, best fitness: 1.999936852740591, size 3
    Iteration 79, best fitness: 1.9999367488931057, size 3
    Iteration 80, best fitness: 1.9999367488931057, size 3
    Iteration 81, best fitness: 1.9999365240573823, size 3
    Iteration 82, best fitness: 1.9999365240573823, size 3
    Iteration 83, best fitness: 1.9999364877738963, size 3
    Iteration 84, best fitness: 1.9999363827215473, size 3
    Iteration 85, best fitness: 1.9999362732602601, size 3
    Iteration 86, best fitness: 1.9999362732602601, size 3
    Iteration 87, best fitness: 1.9999362285568523, size 3
    Iteration 88, best fitness: 1.999936077729481, size 3
    Iteration 89, best fitness: 1.9999360163798068, size 3
    Iteration 90, best fitness: 1.9999359918069513, size 3
    Iteration 91, best fitness: 1.9999359138682389, size 3
    Iteration 92, best fitness: 1.9999359138682389, size 3
    Iteration 93, best fitness: 1.9999357986646122, size 3
    Iteration 94, best fitness: 1.9999357986646122, size 3
    Iteration 95, best fitness: 1.999935629224332, size 3
    Iteration 96, best fitness: 1.9999355545530708, size 3
    Iteration 97, best fitness: 1.9999355171524376, size 3
    Iteration 98, best fitness: 1.999935224770048, size 3
    Iteration 99, best fitness: 1.9999350565008442, size 3





    [<matplotlib.lines.Line2D at 0x10d42b130>]




    
![png](6/output_142_2.png)
    


Different mutation operators may yield different results: For example, rather than just adding random noise to the individual parameters, we can also probabilistically copy values from other parameters:


```python
def mutate(solution):
    P_mutate = 1/len(solution)
    mutated = solution[:]
    for position in range(len(solution)):
        if random.random() < P_mutate:
            if random.random() < 0.9:
                mutated[position] = int(random.gauss(mutated[position], 20))
            else:
                mutated[position] = random.choice(solution)
    return mutated
```

Let's see the performance of the resulting algorithm:


```python
fitness_values = []
MAX = 100000
MIN = -MAX
ga()
```

    Iteration 0, best fitness: 1.9997491219267436
    Iteration 0, best fitness: 1.9997491219267436, size 3
    Iteration 1, best fitness: 1.9997491219267436, size 3
    Iteration 2, best fitness: 1.9997481108312343, size 3
    Iteration 3, best fitness: 1.9997469635627532, size 3
    Iteration 4, best fitness: 1.999746771334515, size 3
    Iteration 5, best fitness: 0.9999531681730904, size 3
    Iteration 6, best fitness: 0.9999531506207543, size 3
    Iteration 7, best fitness: 0.999953119872486, size 3
    Iteration 8, best fitness: 0.999953119872486, size 3
    Iteration 9, best fitness: 0.999942624361696, size 3
    Iteration 10, best fitness: 0.999942624361696, size 3
    Iteration 11, best fitness: 0.9999426210695432, size 3
    Iteration 12, best fitness: 0.0, size 3
    Iteration 13, best fitness: 0.0, size 3
    Iteration 14, best fitness: 0.0, size 3
    Iteration 15, best fitness: 0.0, size 3
    Iteration 16, best fitness: 0.0, size 3
    Iteration 17, best fitness: 0.0, size 3
    Iteration 18, best fitness: 0.0, size 3
    Iteration 19, best fitness: 0.0, size 3
    Iteration 20, best fitness: 0.0, size 3
    Iteration 21, best fitness: 0.0, size 3
    Iteration 22, best fitness: 0.0, size 3
    Iteration 23, best fitness: 0.0, size 3
    Iteration 24, best fitness: 0.0, size 3
    Iteration 25, best fitness: 0.0, size 3
    Iteration 26, best fitness: 0.0, size 3
    Iteration 27, best fitness: 0.0, size 3
    Iteration 28, best fitness: 0.0, size 3
    Iteration 29, best fitness: 0.0, size 3
    Iteration 30, best fitness: 0.0, size 3
    Iteration 31, best fitness: 0.0, size 3
    Iteration 32, best fitness: 0.0, size 3
    Iteration 33, best fitness: 0.0, size 3
    Iteration 34, best fitness: 0.0, size 3
    Iteration 35, best fitness: 0.0, size 3
    Iteration 36, best fitness: 0.0, size 3
    Iteration 37, best fitness: 0.0, size 3
    Iteration 38, best fitness: 0.0, size 3
    Iteration 39, best fitness: 0.0, size 3
    Iteration 40, best fitness: 0.0, size 3
    Iteration 41, best fitness: 0.0, size 3
    Iteration 42, best fitness: 0.0, size 3
    Iteration 43, best fitness: 0.0, size 3
    Iteration 44, best fitness: 0.0, size 3
    Iteration 45, best fitness: 0.0, size 3
    Iteration 46, best fitness: 0.0, size 3
    Iteration 47, best fitness: 0.0, size 3
    Iteration 48, best fitness: 0.0, size 3
    Iteration 49, best fitness: 0.0, size 3
    Iteration 50, best fitness: 0.0, size 3
    Iteration 51, best fitness: 0.0, size 3
    Iteration 52, best fitness: 0.0, size 3
    Iteration 53, best fitness: 0.0, size 3
    Iteration 54, best fitness: 0.0, size 3
    Iteration 55, best fitness: 0.0, size 3
    Iteration 56, best fitness: 0.0, size 3
    Iteration 57, best fitness: 0.0, size 3
    Iteration 58, best fitness: 0.0, size 3
    Iteration 59, best fitness: 0.0, size 3
    Iteration 60, best fitness: 0.0, size 3
    Iteration 61, best fitness: 0.0, size 3
    Iteration 62, best fitness: 0.0, size 3
    Iteration 63, best fitness: 0.0, size 3
    Iteration 64, best fitness: 0.0, size 3
    Iteration 65, best fitness: 0.0, size 3
    Iteration 66, best fitness: 0.0, size 3
    Iteration 67, best fitness: 0.0, size 3
    Iteration 68, best fitness: 0.0, size 3
    Iteration 69, best fitness: 0.0, size 3
    Iteration 70, best fitness: 0.0, size 3
    Iteration 71, best fitness: 0.0, size 3
    Iteration 72, best fitness: 0.0, size 3
    Iteration 73, best fitness: 0.0, size 3
    Iteration 74, best fitness: 0.0, size 3
    Iteration 75, best fitness: 0.0, size 3
    Iteration 76, best fitness: 0.0, size 3
    Iteration 77, best fitness: 0.0, size 3
    Iteration 78, best fitness: 0.0, size 3
    Iteration 79, best fitness: 0.0, size 3
    Iteration 80, best fitness: 0.0, size 3
    Iteration 81, best fitness: 0.0, size 3
    Iteration 82, best fitness: 0.0, size 3
    Iteration 83, best fitness: 0.0, size 3
    Iteration 84, best fitness: 0.0, size 3
    Iteration 85, best fitness: 0.0, size 3
    Iteration 86, best fitness: 0.0, size 3
    Iteration 87, best fitness: 0.0, size 3
    Iteration 88, best fitness: 0.0, size 3
    Iteration 89, best fitness: 0.0, size 3
    Iteration 90, best fitness: 0.0, size 3
    Iteration 91, best fitness: 0.0, size 3
    Iteration 92, best fitness: 0.0, size 3
    Iteration 93, best fitness: 0.0, size 3
    Iteration 94, best fitness: 0.0, size 3
    Iteration 95, best fitness: 0.0, size 3
    Iteration 96, best fitness: 0.0, size 3
    Iteration 97, best fitness: 0.0, size 3
    Iteration 98, best fitness: 0.0, size 3
    Iteration 99, best fitness: 0.0, size 3





    [61199, 61199, 61199]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10d49e440>]




    
![png](6/output_147_1.png)
    


In our fitness function, we manually determined which branches need to evaluate which way, and how to sum up the fitness functions. In practice, this can be automated by combining the branch distance metric with the _approach level_, which was introduced (originally named approximation level) in this paper:

Wegener, J., Baresel, A., & Sthamer, H. (2001). Evolutionary test environment for automatic structural testing. Information and software technology, 43(14), 841-854.

The approach level calculates the distances of an execution from a target node in terms of graph distances on the control dependence graph. However, we will not cover the approach level in this chapter.

## Whole Test Suite Optimisation

Besides the question of how the best fitness function for a coverage goal looks like, there are some related questions: How much time should we spend on optimising for each coverage goal? It is possible that some coverage goals are infeasible (e.g., dead code, or or infeasible branches), so any time spent on these is wasted, while it may be missing for other goals that are feasible but would need more time. Test cases typically cover multiple goals at the same time; even if a test is optimised for one specific line or branch, it may coincidentally cover others along the execution. Thus, the order in which we select coverage goals for optimisation may influence the overall result, and the number of tests we require. In principle, one way to address these issues would be by casting test generation as a multi-objective optimisation problem, and aiming to produce tests for _all_ coverage goals at the same time. However, there is an issue with this: Multi-objective algorithms like the ones we considered in the previous chapter typically work well on 2-3 objectives, but code will generally have many more coverage objectives, rendering classical multi-objective algorithms infeasible (Pareto-dominance happens rarely with higher numbers of objectives). We will therefore now consider some alternatives.

The first alternative we consider is to switch our representation: Rather than optimising individual test cases for individual coverage objectives, we optimise entire test _suites_ to cover _all_ coverage objectives at the same time. Our encoding thus should describe multiple tests. But how many? This is very much problem specific. Thus, rather than hard coding the number of tests, we will only define an upper bound, and let the search decide what is the necessary number of tests.


```python
num_tests = 30
```


```python
def get_random_individual():
    num = random.randint(1, num_tests)
    return [[random.randint(MIN, MAX) for _ in range(num_parameters)] for _ in range(num)]
```

When applying mutation, we need to be able to modify individual tests as before. To keep things challenging, we will not use our optimised mutation that copies parameters, but aim to achieve the entire optimisation just using small steps:


```python
def mutate_test(solution):
    P_mutate = 1/len(solution)
    mutated = solution[:]
    for position in range(len(solution)):
        if random.random() < P_mutate:
            mutated[position] = min(MAX, max(MIN, int(random.gauss(mutated[position], MAX*0.01))))
            
    return mutated
```

However, modifying tests is only one of the things we can do when mutating our actual individuals, which consist of multiple tests. Besides modifying existing tests, we could also delete or add tests, for example like this.


```python
def mutate_set(solution):
    P_mutate = 1/len(solution)
    mutated = []
    for position in range(len(solution)):
        if random.random() >= P_mutate:
            mutated.append(solution[position][:])
            
    if not mutated:
        mutated = solution[:]
    for position in range(len(mutated)):
        if random.random() < P_mutate:
            mutated[position] = mutate_test(mutated[position])
 
    ALPHA = 1/3
    count = 1
    while random.random() < ALPHA ** count:
        count += 1
        mutated.append([random.randint(MIN, MAX) for _ in range(num_parameters)])
    
    return mutated
```

With a certain probability, each of the tests can be removed from a test suite; similarly, each remaining test may be mutated like we mutated tests previously. Finally, with a probability `ALPHA` we insert a new test; if we do so, we insert another one with probability `ALPHA`$^2$, and so on.

When crossing over two individuals, they might have different length, which makes choosing a crossover point difficult. For example, we might pick a crossover point that is longer than one of the parent chromosomes, and then what do we do? A simple solution would be to pick two different crossover points.


```python
def variable_crossover(parent1, parent2):
    pos1 = random.randint(1, len(parent1))
    pos2 = random.randint(1, len(parent2))
    offspring1 = parent1[:pos1] + parent2[pos2:]
    offspring2 = parent2[:pos2] + parent1[pos1:]
    return (offspring1, offspring2)
```

To see this works, we need to define the fitness function. Since we want to cover _everything_ we simply need to make sure that every single branch is covered at least once in a test suite. A branch is covered if its minimum branch distance is 0; thus, if everything is covered, then the sum of minimal branch distances should be 0.

There is one special case: If an if-statement is executed only once, then optimising the true/false distance may lead to a suboptimal, oscillising evolution. We therefore also count how often each if-condition was executed. If it was only executed once, then the fitness value for that branch needs to be higher than if it was executed twice. For this, we extend our `update_maps` function to also keep track of the execution count:


```python
condition_count = {}
def update_maps(condition_num, d_true, d_false):
    global distances_true, distances_false, condition_count

    if condition_num in condition_count.keys():
        condition_count[condition_num] = condition_count[condition_num] + 1
    else:
        condition_count[condition_num] = 1
        
    if condition_num in distances_true.keys():
        distances_true[condition_num] = min(
            distances_true[condition_num], d_true)
    else:
        distances_true[condition_num] = d_true

    if condition_num in distances_false.keys():
        distances_false[condition_num] = min(
            distances_false[condition_num], d_false)
    else:
        distances_false[condition_num] = d_false
```

The actual fitness function now is the sum of minimal distances after all tests have been executed. If an if-condition was not executed at all, then the true distance and the false distance will be 1, resulting in a sum of 2 for the if-condition. If the condition was covered only once, we set the fitness to exactly 1. If the condition was executed more than once, then at least either the true or false distance has to be 0, such that in sum, true and false distances will be less than 0.


```python
def get_fitness(x):
    # Reset any distance values from previous executions
    global distances_true, distances_false, condition_count
    distances_true =  {x: 1.0 for x in range(num_branches)}
    distances_false = {x: 1.0 for x in range(num_branches)}
    condition_count = {x:   0 for x in range(num_branches)}

    # Run the function under test
    for test in x:
        triangle_instrumented(*test)

    # Sum up branch distances
    fitness = 0.0
    for branch in range(num_branches):
        if condition_count[branch] == 1:
            fitness += 1
        else:
            fitness += distances_true[branch]
            fitness += distances_false[branch]

    return fitness
```

Before we run some experiments on this, let's make a small addition to our genetic algorithm: Since the size of individuals is variable it will be interesting to observe how this evolves. We'll captures the average population size in a separate list. Since the costs of evaluating fitness are no longer constant per individual but depend on the number of tests executed, we will also change our stopping criterion to the number of executed tests.


```python
from statistics import mean

length_values = []
max_executions = 10000

def ga():
    population = [get_random_individual() for _ in range(population_size)]
    best_fitness = sys.maxsize
    tests_executed = 0
    for p in population:
        fitness = get_fitness(p)
        tests_executed += len(p)
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = p
    while tests_executed < max_executions:
        fitness_values.append(best_fitness)
        length_values.append(mean([len(x) for x in population]))
        new_population = elitism(population)
        while len(new_population) < len(population):
            parent1 = selection(population)
            parent2 = selection(population)

            if random.random() < P_xover:
                offspring1, offspring2 = crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1, parent2

            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            
            new_population.append(offspring1)
            new_population.append(offspring2)
            tests_executed += len(offspring1) + len(offspring2)

        population = new_population
        for p in population:
            fitness = get_fitness(p)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = p
    print(f"Best fitness: {best_fitness}, size {len(best_solution)}")

    return best_solution
```

Since we now have all the operators we need in place, let's run a first experiment aiming to achieve 100% coverage on the triangle example.


```python
max_executions = 1000000
MAX = 1000
MIN = -MAX
crossover = variable_crossover
selection = tournament_selection
elitism   = elitism_standard 
mutate    = mutate_set
tournament_size = 4
population_size = 50
fitness_values = []
length_values = []
ga()
```

    Best fitness: 1.6639928698752229, size 599





    [[163, -456, 599],
     [-959, -679, -410],
     [-250, -937, -311],
     [-714, -221, 291],
     [-671, -292, 622],
     [755, 79, 171],
     [-957, 728, 850],
     [601, 675, 129],
     [850, 829, -838],
     [955, -584, 306],
     [349, -307, 662],
     [-627, -999, 506],
     [423, 738, 207],
     [697, -759, 433],
     [-316, -826, 245],
     [38, -736, 465],
     [-358, 699, 200],
     [248, 602, 104],
     [606, -133, 946],
     [255, 330, 335],
     [-376, 687, 192],
     [200, -899, 422],
     [442, 441, 802],
     [-849, 683, 601],
     [731, -183, 721],
     [931, -847, 953],
     [682, -346, -440],
     [-117, -105, -443],
     [218, 49, 531],
     [-733, 183, -990],
     [939, 3, 325],
     [64, 737, 522],
     [237, 530, -62],
     [-559, -615, -926],
     [-839, 129, 274],
     [211, 671, -877],
     [120, -431, -259],
     [261, 333, 335],
     [-358, 699, 186],
     [200, -899, 422],
     [442, 441, 802],
     [931, -849, 960],
     [682, -346, -440],
     [-117, -105, -443],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [64, 737, 522],
     [237, 530, -62],
     [-839, 129, 274],
     [120, -431, -259],
     [323, -379, -684],
     [-884, -628, 599],
     [-198, 986, -563],
     [688, 576, 687],
     [611, 324, -664],
     [-198, 984, -575],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [673, -90, -501],
     [916, 3, 331],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [923, -845, 938],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [930, -82, 273],
     [-849, 683, 601],
     [731, -183, 721],
     [931, -847, 953],
     [682, -346, -440],
     [-117, -105, -443],
     [218, 49, 531],
     [-733, 183, -990],
     [916, 3, 331],
     [64, 737, 522],
     [237, 530, -62],
     [-561, -615, -932],
     [-839, 129, 274],
     [211, 671, -877],
     [120, -431, -259],
     [323, -379, -675],
     [-884, -628, 608],
     [-198, 986, -576],
     [692, 576, 685],
     [611, 324, -664],
     [-192, 984, -575],
     [255, 330, 335],
     [-358, 699, 186],
     [200, -899, 422],
     [682, -346, -440],
     [-117, -105, -443],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [64, 737, 522],
     [-839, 129, 274],
     [120, -431, -259],
     [323, -379, -684],
     [-884, -628, 608],
     [-198, 986, -563],
     [688, 576, 687],
     [611, 324, -664],
     [-198, 984, -575],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [73, 176, -940],
     [672, -437, 292],
     [442, 439, 786],
     [-849, 683, 601],
     [923, -845, 938],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [-198, 984, -576],
     [73, 176, -918],
     [692, -92, -503],
     [64, 737, 522],
     [-839, 129, 274],
     [211, 671, -877],
     [323, -379, -684],
     [-884, -628, 608],
     [915, -88, 268],
     [-195, 984, -576],
     [725, 564, 751],
     [606, -146, 946],
     [-358, 699, 186],
     [200, -899, 422],
     [442, 441, 802],
     [-884, -628, 608],
     [-198, 986, -576],
     [692, 576, 685],
     [611, 324, -664],
     [-198, 980, -565],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [64, 737, 522],
     [237, 530, -62],
     [-561, -615, -932],
     [211, 671, -877],
     [120, -431, -259],
     [-884, -628, 608],
     [930, -82, 273],
     [-198, 984, -576],
     [73, 176, -940],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [64, 737, 522],
     [237, 530, -62],
     [-561, -615, -932],
     [211, 671, -877],
     [-884, -628, 608],
     [930, -82, 273],
     [-198, 984, -576],
     [692, -92, -503],
     [64, 737, 522],
     [-839, 129, 274],
     [120, -431, -259],
     [323, -379, -684],
     [-884, -628, 608],
     [-198, 986, -563],
     [688, 576, 687],
     [611, 324, -664],
     [-198, 984, -575],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [-561, -615, -932],
     [-839, 129, 274],
     [211, 671, -877],
     [120, -431, -259],
     [323, -379, -675],
     [-884, -628, 608],
     [-198, 986, -576],
     [692, 576, 685],
     [611, 324, -664],
     [-192, 984, -575],
     [255, 330, 335],
     [-358, 699, 186],
     [200, -899, 422],
     [442, 442, 815],
     [676, -346, -440],
     [-117, -105, -443],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [64, 737, 522],
     [-839, 129, 274],
     [120, -431, -259],
     [323, -379, -684],
     [-884, -628, 608],
     [-198, 986, -563],
     [688, 576, 687],
     [611, 324, -664],
     [-198, 984, -575],
     [73, 176, -940],
     [672, -437, 292],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [64, 737, 522],
     [237, 530, -62],
     [-561, -615, -932],
     [211, 671, -877],
     [120, -431, -259],
     [-884, -628, 608],
     [930, -82, 273],
     [-198, 984, -576],
     [73, 176, -940],
     [925, 3, 331],
     [697, -759, 433],
     [-316, -826, 245],
     [38, -736, 465],
     [-358, 699, 186],
     [216, -728, -51],
     [248, 599, 104],
     [606, -133, 946],
     [931, -849, 960],
     [682, -346, -440],
     [-117, -105, -443],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [64, 737, 522],
     [237, 530, -62],
     [-839, 129, 274],
     [120, -431, -259],
     [323, -379, -684],
     [-884, -628, 599],
     [-198, 986, -563],
     [688, 576, 687],
     [611, 324, -664],
     [-198, 984, -575],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [673, -90, -501],
     [916, 3, 331],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [923, -845, 938],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [930, -82, 273],
     [-849, 683, 601],
     [731, -183, 721],
     [931, -847, 953],
     [682, -346, -440],
     [-117, -105, -443],
     [218, 49, 531],
     [-733, 183, -990],
     [916, 3, 331],
     [64, 737, 522],
     [237, 530, -62],
     [-561, -615, -932],
     [-839, 129, 274],
     [211, 671, -877],
     [120, -431, -259],
     [323, -379, -675],
     [-884, -628, 608],
     [-198, 986, -576],
     [692, 576, 685],
     [611, 324, -664],
     [-192, 984, -575],
     [255, 330, 335],
     [-358, 699, 186],
     [200, -899, 422],
     [442, 441, 802],
     [682, -346, -440],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [64, 737, 522],
     [-839, 129, 274],
     [120, -431, -259],
     [323, -379, -684],
     [-884, -628, 608],
     [-198, 986, -563],
     [687, 586, 687],
     [611, 318, -634],
     [-198, 984, -575],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [73, 176, -940],
     [672, -437, 292],
     [442, 439, 786],
     [-849, 683, 601],
     [923, -845, 938],
     [218, 49, 531],
     [-733, 183, -990],
     [916, 3, 331],
     [-198, 984, -576],
     [73, 176, -918],
     [692, -92, -503],
     [64, 737, 522],
     [-839, 129, 274],
     [211, 671, -877],
     [309, -379, -684],
     [-884, -628, 608],
     [915, -88, 268],
     [-195, 984, -576],
     [725, 564, 751],
     [606, -146, 946],
     [-358, 699, 186],
     [200, -899, 422],
     [442, 441, 802],
     [-884, -628, 608],
     [-198, 986, -576],
     [692, 576, 685],
     [611, 324, -664],
     [-198, 980, -565],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [64, 737, 522],
     [237, 530, -62],
     [-561, -615, -932],
     [211, 671, -877],
     [120, -431, -259],
     [-884, -628, 608],
     [930, -82, 273],
     [-198, 984, -576],
     [73, 176, -940],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [64, 737, 522],
     [237, 530, -62],
     [-561, -615, -932],
     [211, 671, -877],
     [-884, -628, 608],
     [930, -82, 273],
     [-198, 984, -576],
     [692, -92, -503],
     [64, 737, 522],
     [-839, 129, 274],
     [120, -431, -259],
     [323, -379, -684],
     [-884, -628, 608],
     [-198, 986, -563],
     [688, 576, 687],
     [611, 324, -664],
     [-198, 984, -575],
     [73, 176, -940],
     [672, -437, 292],
     [-733, 183, -990],
     [-561, -615, -932],
     [-839, 129, 274],
     [211, 671, -877],
     [120, -431, -259],
     [323, -379, -675],
     [-884, -628, 608],
     [-198, 986, -576],
     [692, 576, 685],
     [611, 324, -664],
     [-192, 984, -575],
     [255, 330, 335],
     [-358, 699, 186],
     [200, -899, 422],
     [442, 442, 815],
     [682, -346, -440],
     [-117, -105, -443],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [64, 737, 522],
     [-839, 129, 274],
     [120, -431, -259],
     [323, -379, -684],
     [-884, -628, 608],
     [-198, 986, -563],
     [688, 576, 687],
     [611, 324, -664],
     [-198, 984, -575],
     [73, 176, -940],
     [672, -437, 292],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [64, 737, 522],
     [237, 530, -62],
     [-561, -615, -932],
     [211, 671, -877],
     [120, -431, -259],
     [930, -82, 273],
     [-198, 984, -576],
     [73, 176, -940],
     [916, 3, 331],
     [697, -759, 433],
     [-316, -826, 245],
     [38, -736, 465],
     [-358, 699, 186],
     [216, -728, -51],
     [248, 599, 104],
     [606, -133, 946],
     [255, 330, 335],
     [-198, 984, -576],
     [73, 176, -940],
     [916, 3, 331],
     [64, 737, 522],
     [-849, 683, 601],
     [731, -183, 721],
     [218, 49, 531],
     [-733, 183, -990],
     [939, 3, 325],
     [64, 737, 522],
     [237, 530, -62],
     [-559, -615, -926],
     [-839, 129, 274],
     [211, 671, -877],
     [120, -431, -259],
     [323, -379, -684],
     [-884, -628, 608],
     [-198, 986, -576],
     [692, 576, 685],
     [611, 324, -664],
     [-192, 984, -575],
     [261, 333, 335],
     [-358, 699, 186],
     [200, -899, 422],
     [442, 441, 802],
     [931, -849, 960],
     [682, -346, -440],
     [-117, -105, -443],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [64, 737, 522],
     [237, 530, -62],
     [-839, 129, 274],
     [120, -431, -259],
     [323, -379, -684],
     [-884, -628, 599],
     [-198, 986, -563],
     [688, 576, 687],
     [611, 324, -664],
     [-198, 984, -575],
     [73, 176, -940],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -105, -512],
     [916, 3, 331],
     [73, 156, -923],
     [672, -437, 292],
     [218, 49, 531],
     [923, -845, 938],
     [218, 49, 531],
     [-733, 183, -990],
     [697, -92, -503],
     [916, 3, 331],
     [-849, 683, 601],
     [731, -183, 721],
     [931, -847, 953],
     [682, -346, -440],
     [-117, -105, -443],
     [218, 49, 531],
     [-733, 183, -990],
     [916, 3, 331],
     [64, 737, 522],
     [237, 530, -62],
     [-561, -615, -932],
     [-839, 129, 274],
     [211, 671, -877],
     [120, -431, -259],
     [323, -379, -675],
     [-198, 986, -576],
     [692, 576, 685],
     [611, 324, -664],
     [-192, 984, -575],
     [255, 330, 335],
     [-358, 699, 186],
     [200, -899, 422],
     [442, 442, 815],
     [682, -346, -440],
     [-117, -105, -443],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [64, 737, 522],
     [-839, 129, 274],
     [120, -431, -259],
     [323, -379, -684],
     [-884, -628, 608],
     [-198, 986, -563],
     [688, 576, 687],
     [611, 324, -664],
     [-198, 984, -575],
     [73, 176, -940],
     [672, -437, 292],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [73, 176, -940],
     [672, -437, 292],
     [218, 49, 531],
     [-733, 183, -990],
     [692, -92, -503],
     [916, 3, 331],
     [64, 737, 522],
     [237, 530, -62],
     [-561, -615, -932],
     [211, 671, -877],
     [120, -431, -259],
     [-884, -628, 608],
     [930, -82, 273],
     [-198, 984, -576],
     [73, 176, -940],
     [916, 3, 331],
     [697, -759, 433],
     [-316, -826, 245],
     [38, -736, 465],
     [-358, 699, 186],
     [216, -728, -51],
     [248, 599, 104],
     [606, -133, 946],
     [255, 330, 335],
     [73, 176, -940],
     [916, 3, 331],
     [64, 737, 522],
     [-561, -615, -932],
     [-839, 129, 274],
     [211, 671, -877],
     [323, -379, -684],
     [-884, -628, 608],
     [930, -82, 265],
     [-198, 984, -576],
     [-790, 703, -764],
     [-144, -339, -694],
     [-297, 628, -826],
     [692, -92, -503],
     [916, 3, 331],
     [323, -379, -684],
     [-884, -628, 608],
     [930, -82, 265],
     [-198, 984, -576],
     [-790, 703, -764],
     [-144, -339, -694],
     [-297, 628, -826],
     [-853, 558, 386],
     [692, 576, 684],
     [424, -594, 750],
     [496, -432, -487],
     [-102, -184, 477],
     [-824, 604, 358],
     [368, 664, -530],
     [731, -318, -683],
     [634, 942, -444],
     [827, -299, 163],
     [626, 411, 414],
     [-813, 860, -343],
     [-788, -914, -831],
     [-227, 645, 760],
     [-423, -980, 321],
     [534, 95, -367]]




```python
plt.plot(fitness_values)
```




    [<matplotlib.lines.Line2D at 0x10d5beda0>]




    
![png](6/output_170_1.png)
    


The plot shows iterations of the genetic algorithm on the x-axis. Very likely, the result likely isn't great. But why? Let's look at the average population length.


```python
plt.plot(length_values)
```




    [<matplotlib.lines.Line2D at 0x10d5c97e0>]




    
![png](6/output_172_1.png)
    


What you can see here is a phenomenon called _bloat_: Individuals grow in size through mutation and crossover, and the search has no incentive to reduce their size (adding a test can never decrease coverage; removing a test can). As a result the individuals just keep growing, and quickly eat up all the available resources for the search. How to deal with this problem will be covered in the next chapter.
