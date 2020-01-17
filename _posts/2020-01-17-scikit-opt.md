
## 6. immune algorithm (IA)
-> Demo code: [examples/demo_ia.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ia.py#L6)
```python

from sko.IA import IA_TSP

ia_tsp = IA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=500, max_iter=800, prob_mut=0.2,
                T=0.7, alpha=0.95)
best_points, best_distance = ia_tsp.run()
print('best routine:', best_points, 'best_distance:', best_distance)

```

**![sa_tsp1](https://github.com/zhang5389/zhang5389.github.io/blob/master/images/sa_tsp1.gif?raw=true)**



![sa_tsp1](../images/sa_tsp1.gif)



![pso](../images/pso.gif)

![ia2](../images/ia2.png)

## 7. artificial fish swarm algorithm (AFSA)
-> Demo code: [examples/demo_asfs.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_asfs.py#L1)
```python
def func(x):
    x1, x2 = x
    return 1 / x1 ** 2 + x1 ** 2 + 1 / x2 ** 2 + x2 ** 2


from sko.ASFA import ASFA

asfa = ASFA(func, n_dim=2, size_pop=50, max_iter=300,
            max_try_num=100, step=0.5, visual=0.3,
            q=0.98, delta=0.5)
best_x, best_y = asfa.run()
print(best_x, best_y)
```
