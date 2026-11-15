# Data Structures & Algorithms with Python for Data Science

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Focus](https://img.shields.io/badge/Focus-DSA%20%7C%20Data%20Science%20%7C%20ML-purple)](#)
[![Level](https://img.shields.io/badge/Level-Beginner%20to%20Interview--Ready-success)](#)

A practical, visual, industry-oriented guide to **Data Structures and
Algorithms (DSA) in Python**, taught through problems relevant to Data
Science, Machine Learning, analytics, and production software.

The goal is to connect computer science fundamentals with real data
workflows: preprocessing large datasets, engineering features, ranking
recommendations, traversing graphs, selecting top-K results, and
understanding the computational cost of ML algorithms.

> **Learning philosophy:** Understand the problem, choose a suitable
> data structure, implement a correct solution, analyse its complexity,
> and validate performance using realistic data.

------------------------------------------------------------------------

## Table of contents

1.  [Why do we need DSA?](#1-why-do-we-need-dsa)
2.  [Why DSA matters in Data
    Science](#2-why-dsa-matters-in-data-science)
3.  [Complexity analysis](#3-complexity-analysis)
4.  [Repository structure](#4-repository-structure)
5.  [Learning path](#5-learning-path)
6.  [Data structures: concepts and
    implementations](#6-data-structures-concepts-and-implementations)
7.  [DSA in the Data Science and ML
    lifecycle](#7-dsa-in-the-data-science-and-ml-lifecycle)
8.  [Visual learning lab](#8-visual-learning-lab)
9.  [How to run the code](#9-how-to-run-the-code)
10. [Engineering standards](#10-engineering-standards)
11. [Interview preparation](#11-interview-preparation)
12. [Suggested projects](#12-suggested-projects)
13. [References](#13-references)

------------------------------------------------------------------------

# 1. Why do we need DSA?

## What are data structures and algorithms?

-   A **data structure** is a way to organise and store data so that
    operations such as access, insertion, deletion, and search can be
    performed effectively.
-   An **algorithm** is a finite, well-defined sequence of steps used to
    solve a problem.

They work together. A suitable data structure makes an algorithm
practical; a suitable algorithm makes the use of that structure
efficient.

### A practical example

Suppose we need to check whether each incoming customer ID exists in a
list of registered customers.

``` python
registered_ids = [1001, 1002, 1003, 1004]
incoming_ids = [1003, 1005, 1001]

# List membership scans the registered IDs.
valid_list = [
    customer_id for customer_id in incoming_ids
    if customer_id in registered_ids
]

# Set membership uses a hash table internally.
registered_lookup = set(registered_ids)
valid_set = [
    customer_id for customer_id in incoming_ids
    if customer_id in registered_lookup
]

print(valid_set)  # [1003, 1001]
```

For `n` registered IDs and `m` incoming IDs:

-   Repeated list membership can take **O(n × m)** time in the worst
    case.
-   Building a set and checking membership takes **O(n + m)** expected
    time overall.

The set-based approach is generally preferable for repeated membership
checks. Actual runtime also depends on data size, hashing, memory, and
implementation details.

## Why programmers need DSA

1.  **Scalability:** solutions continue to work as data volume and user
    traffic grow.
2.  **Speed:** avoid unnecessary repeated operations.
3.  **Memory efficiency:** select representations that do not waste
    memory.
4.  **Reliability:** reason about edge cases and correctness.
5.  **Maintainability:** explain why a solution is appropriate.
6.  **Interview readiness:** demonstrate structured problem-solving, not
    just syntax knowledge.

## How DSA improves problem-solving

A repeatable approach:

1.  Clarify inputs, outputs, constraints, and edge cases.
2.  Start with a simple, correct solution.
3.  Identify repeated work or expensive operations.
4.  Select a data structure that supports the required operations.
5.  Improve the algorithm and state its complexity.
6.  Test normal, boundary, and adversarial cases.
7.  Benchmark with representative data before claiming a speed-up.

## Real-world applications

  -----------------------------------------------------------------------
  Application             DSA commonly involved   Why it matters
  ----------------------- ----------------------- -----------------------
  Search engines          Inverted indexes, hash  Retrieve and rank
                          maps, heaps, graphs     relevant documents

  Databases               B-trees, hash indexes,  Query data efficiently
                          sorting, join           
                          algorithms              

  Recommendation systems  Graphs, heaps, hashing, Find related
                          similarity search       users/items and top
                                                  recommendations

  Navigation              Weighted graphs,        Find efficient routes
                          shortest-path           
                          algorithms              

  Streaming analytics     Queues, heaps, hash     Maintain recent events
                          maps                    and running summaries

  Compilers               Trees, stacks, hash     Parse and analyse
                          tables                  source code

  ML systems              Arrays, trees, graphs,  Store features, split
                          heaps                   data, rank outputs
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 2. Why DSA matters in Data Science

Data Science is not only model training. A real workflow repeatedly
loads, validates, transforms, joins, searches, aggregates, ranks, and
serves data. DSA helps make each step efficient and understandable.

  ------------------------------------------------------------------------
  Data Science task       DSA connection          Industry use case
  ----------------------- ----------------------- ------------------------
  Data preprocessing      Arrays, hash maps,      Deduplicate and validate
                          sorting                 millions of records

  Large datasets          Streaming, queues,      Process files larger
                          chunking, external      than RAM
                          sorting                 

  Searching and sorting   Binary search, hashing, Find time windows or
                          sorting algorithms      order events

  Memory optimisation     In-place operations,    Reduce ETL memory
                          compact arrays,         pressure
                          generators              

  Feature engineering     Hash maps, sets, prefix Count categories and
                          sums, sorting           build rolling features

  Pandas and NumPy        Vectorised arrays,      Fast column operations
                          hash-based indexes,     and table joins
                          joins                   

  ML optimisation         Complexity analysis,    Understand training and
                          trees, heaps, matrix    inference cost
                          algorithms              

  Recommendation systems  Graphs, heaps, hash     Generate and rank
                          maps                    candidate items

  Decision Trees / Random Tree structures         Partition feature space
  Forests                                         and make predictions

  Data processing         Hash tables             Group, count, and join
                                                  records

  Top-K predictions       Heaps / partial         Return the
                          selection               highest-scoring products

  Optimisation            Dynamic programming     Solve
                                                  overlapping-subproblem
                                                  problems

  Social network analysis Graphs                  Explore connections and
                                                  communities
  ------------------------------------------------------------------------

## Practical implementations

### A. Efficient preprocessing with a set

``` python
def remove_duplicate_ids(ids):
    seen = set()
    unique = []

    for item in ids:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return unique

print(remove_duplicate_ids([7, 2, 7, 3, 2, 9]))
# [7, 2, 3, 9]
```

**Performance:** expected time O(n), additional space O(n). Original
order is preserved.

**Industry use:** deduplicate customer IDs, validate entities, and
detect overlap between training and validation datasets.

### B. Top-K predictions with a heap

``` python
import heapq

scores = [
    ("item_A", 0.91),
    ("item_B", 0.72),
    ("item_C", 0.98),
    ("item_D", 0.86),
]

top_2 = heapq.nlargest(2, scores, key=lambda pair: pair[1])
print(top_2)
# [('item_C', 0.98), ('item_A', 0.91)]
```

For `n` candidates and `k` requested results, heap-based selection is
typically around **O(n log k)**, compared with **O(n log n)** for
sorting all candidates. If `k` is close to `n`, a full sort may be
competitive.

**Industry use:** return the highest-scoring products, search results,
or model predictions without fully sorting every candidate.

### C. Graphs for recommendations

Represent users and items as nodes, and interactions as edges. Traversal
can discover related items or users.

**Industry use:** collaborative filtering, social recommendations,
fraud-ring detection, and knowledge graphs.

Production recommenders often combine graph methods with embeddings,
ranking models, and business rules; a graph alone is not the complete
recommendation system.

### D. Trees in Machine Learning

A decision tree repeatedly applies feature-based conditions to divide
observations into smaller groups. Random Forest combines predictions
from many trees.

**Industry use:** customer segmentation, credit-risk classification,
churn prediction, and interpretable rule-based predictions.

The underlying tree structure makes inference interpretable as a path
from root to leaf. Training cost depends on sample count, feature count,
depth, split strategy, and implementation.

### E. Dynamic programming

Dynamic programming is useful when a problem has overlapping subproblems
and optimal substructure.

**Industry use:** sequence alignment, resource allocation, scheduling,
and certain optimisation problems. It is not automatically the best
approach for every optimisation task.

## Complexity of common ML operations

  -----------------------------------------------------------------------
  Operation / model family            Typical complexity intuition
  ----------------------------------- -----------------------------------
  Vector arithmetic on `n` values     O(n)

  Dense matrix multiplication         O(mkn)
  `(m×k) · (k×n)`                     

  Brute-force pairwise distances for  O(n²d)
  `n` rows, `d` features              

  Sorting `n` observations            O(n log n)

  Balanced tree lookup                O(log n)

  Linear model inference with `d`     O(d) per row
  features                            

  Decision-tree inference             O(depth) per row

  Brute-force k-NN query against `n`  O(nd), plus selection/ranking
  rows with `d` features              

  Heap top-K from `n` candidates      O(n log k)
  -----------------------------------------------------------------------

These are simplified models, not universal guarantees. Sparse data,
batching, hardware acceleration, approximate algorithms, parallelism,
and library implementation can change practical costs.

------------------------------------------------------------------------

# 3. Complexity analysis

Complexity analysis estimates how an algorithm's resource use grows with
input size.

-   **Time complexity:** how the number of operations grows.
-   **Space complexity:** how additional memory use grows. State whether
    input storage is included.

## Common growth rates

  Complexity   Name           Typical example
  ------------ -------------- ----------------------------------
  O(1)         Constant       Access an array element by index
  O(log n)     Logarithmic    Binary search in sorted data
  O(n)         Linear         Scan a list once
  O(n log n)   Linearithmic   Efficient comparison sorting
  O(n²)        Quadratic      Compare every pair
  O(2ⁿ)        Exponential    Naive subset enumeration
  O(n!)        Factorial      Enumerate all permutations

### Visual intuition

``` text
Growth as input size increases →

O(1)       ───────────────  almost flat
O(log n)   ──────────────── slowly rises
O(n)       /                proportional growth
O(n log n) /                 rises a little faster
O(n²)      /                 rises steeply
O(2ⁿ)      /                 becomes impractical quickly
```

## Big O, Omega, and Theta

-   **Big O --- O(g(n)):** asymptotic upper bound. Often used to
    describe worst-case growth.
-   **Big Omega --- Ω(g(n)):** asymptotic lower bound.
-   **Big Theta --- Θ(g(n)):** tight asymptotic bound; both upper and
    lower bounds hold.

These notations describe bounds, not best/worst/average cases by
themselves. State the case being analysed separately.

### Example: linear search

``` python
def linear_search(values, target):
    for index, value in enumerate(values):
        if value == target:
            return index
    return -1
```

-   Best case: Θ(1), when the first item matches.
-   Worst case: Θ(n), when the item is last or absent.
-   Auxiliary space: O(1).

### Example: binary search

``` python
def binary_search(values, target):
    """Return target index in a sorted sequence, otherwise -1."""
    left, right = 0, len(values) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if values[mid] == target:
            return mid
        if values[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

-   Requirement: input must be sorted.
-   Time: O(log n).
-   Auxiliary space: O(1).

------------------------------------------------------------------------

# 4. Repository structure

The repository is designed to grow module by module. Keep explanations,
runnable code, tests, and visualisations close to the concept they
teach.

``` text
dsa-python-for-data-science/
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── 00_python_fundamentals/
│   ├── 01_variables_and_data_types.py
│   ├── 02_conditionals.py
│   ├── 03_loops.py
│   ├── 04_functions.py
│   ├── 05_recursion.py
│   ├── 06_list_comprehensions.py
│   ├── 07_lambda_functions.py
│   └── 08_exception_handling.py
│
├── 01_complexity_analysis/
│   ├── big_o_examples.py
│   ├── time_benchmarking.py
│   └── space_complexity.py
│
├── 02_linear_data_structures/
│   ├── arrays_and_lists.py
│   ├── strings.py
│   ├── tuples.py
│   ├── stacks.py
│   ├── queues.py
│   └── linked_lists.py
│
├── 03_hash_based_structures/
│   ├── dictionaries.py
│   ├── sets.py
│   └── hash_tables.py
│
├── 04_nonlinear_data_structures/
│   ├── trees.py
│   ├── binary_search_trees.py
│   ├── heaps.py
│   ├── graphs.py
│   └── tries.py
│
├── 05_searching_and_sorting/
│   ├── linear_search.py
│   ├── binary_search.py
│   ├── bubble_sort.py
│   ├── merge_sort.py
│   ├── quick_sort.py
│   └── heap_sort.py
│
├── 06_algorithmic_techniques/
│   ├── two_pointers/
│   ├── sliding_window/
│   ├── recursion_backtracking/
│   ├── greedy/
│   └── dynamic_programming/
│
├── 07_dsa_for_data_science/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── top_k_recommendations/
│   ├── graph_analytics/
│   └── ml_complexity/
│
├── 08_visualizations/
│   ├── list_operations.py
│   ├── stack_queue_animation.py
│   ├── tree_traversal.py
│   ├── heap_operations.py
│   ├── graph_traversal.py
│   └── trie_visualization.py
│
├── 09_interview_patterns/
│   ├── arrays_hashing/
│   ├── strings/
│   ├── trees_graphs/
│   └── dynamic_programming/
│
├── tests/
│   └── ...
│
└── notebooks/
    ├── complexity_experiments.ipynb
    └── dsa_for_data_science.ipynb
```

------------------------------------------------------------------------

# 5. Learning path

## Module A --- Python fundamentals

  -----------------------------------------------------------------------
  Topic                   What to learn           Data Science connection
  ----------------------- ----------------------- -----------------------
  Variables and data      `int`, `float`, `str`,  Represent measurements,
  types                   `bool`, `None`          labels, missing values

  Conditional statements  `if`, `elif`, `else`    Rule-based data
                                                  validation

  Loops                   `for`, `while`,         Process records and
                          iteration               streams

  Functions               Parameters, return      Reusable preprocessing
                          values, scope           functions

  Recursion               Base case and recursive Tree traversal and
                          case                    divide-and-conquer

  List comprehensions     Concise transformations Build filtered
                                                  collections

  Lambda functions        Small anonymous         Sorting keys and simple
                          functions               transformations

  Exception handling      `try`, `except`,        Robust file and data
                          `else`, `finally`       pipeline handling
  -----------------------------------------------------------------------

Example:

``` python
# Keep only valid positive measurements.
measurements = [12.5, -1, 8.0, None, 0, 16.2]
valid = [x for x in measurements if x is not None and x > 0]
print(valid)  # [12.5, 8.0, 16.2]
```

## Module B --- Data structures

Every module should include: definition, operation walkthrough, Python
implementation, complexity table, edge cases, tests, a visual, and a
Data Science use case.

  ---------------------------------------------------------------------------
  Structure         Core operations   Typical complexity    Data Science
                                                            example
  ----------------- ----------------- --------------------- -----------------
  Array / Python    Index, append,    Index O(1); append    Feature values
  list              insert, delete,   amortized O(1);       and batches
                    search            search O(n)           

  String            Index, slice,     Index O(1);           Text cleaning and
                    search, transform transformations often tokenisation
                                      O(n) or more          

  Tuple             Index, unpack,    Index O(1); immutable Fixed records and
                    iterate                                 coordinate pairs

  Dictionary        Get, set, delete, Expected O(1); worst  Category counts
                    membership        case can be O(n)      and lookup tables

  Set               Add, remove,      Expected O(1)         Deduplication and
                    membership                              set operations

  Stack             Push, pop, peek   O(1)                  DFS, parsing,
                                                            undo operations

  Queue             Enqueue, dequeue  O(1) with             Event pipelines
                                      `collections.deque`   and BFS

  Linked list       Insert/delete     Access/search O(n);   Pointer-based
                    with node         local insert/delete   structures
                    reference,        O(1)                  
                    traverse                                

  Hash table        Insert, lookup,   Expected O(1)         Indexing and
                    delete                                  grouping

  Tree              Traverse, search, Traversal O(n)        Hierarchical data
                    insert (depends                         and model
                    on type)                                structure

  Binary Search     Search, insert,   Average O(log n);     Ordered lookup
  Tree              delete            worst O(n) if skewed  concepts

  Heap              Push, pop, peek   Push/pop O(log n);    Top-K predictions
                                      peek O(1)             and scheduling

  Graph             BFS, DFS, edge    Traversal O(V + E)    User-item
                    operations        with adjacency lists  networks

  Trie              Insert, search,   O(L), where L is key  Autocomplete and
                    prefix search     length                vocabulary lookup
  ---------------------------------------------------------------------------

### Important Python distinctions

-   Python `list` is a dynamic array, not a linked list.
-   Use `collections.deque` for efficient queue operations at both ends.
-   Python `dict` and `set` are hash-based structures.
-   `heapq` implements a min-heap.
-   A normal binary search tree is not automatically balanced.

------------------------------------------------------------------------

# 6. Data structures: concepts and implementations

## Arrays and lists

A Python list stores references in a dynamic array. It offers fast
indexing and efficient append-at-end in amortised terms, but inserting
or deleting near the beginning shifts later elements.

``` python
values = [10, 20, 30]
values.append(40)
values[1] = 25
print(values)  # [10, 25, 30, 40]
```

**Data Science use:** hold collections, batches, and intermediate
results. For large numerical workloads, prefer NumPy arrays when
vectorised numerical operations are appropriate.

## Strings

Strings are immutable Unicode sequences.

``` python
text = "  Customer Churn  "
clean = text.strip().lower().replace(" ", "_")
print(clean)  # customer_churn
```

**Data Science use:** normalise text fields, prepare tokens, and clean
categorical values. Repeated string concatenation in a loop can be
inefficient; collect pieces and use `"".join(parts)`.

## Tuples

Tuples are immutable ordered collections.

``` python
record_key = ("customer_104", "2026-11")
customer_id, month = record_key
```

**Data Science use:** represent fixed keys, coordinates, and immutable
records.

## Dictionaries and hash tables

``` python
counts = {}

for label in ["A", "B", "A", "C", "B", "A"]:
    counts[label] = counts.get(label, 0) + 1

print(counts)  # {'A': 3, 'B': 2, 'C': 1}
```

Expected time: O(n) for counting n values; additional space O(u), where
u is the number of unique labels.

**Data Science use:** category frequencies, ID lookups, caching, and
joining small reference datasets.

## Sets

``` python
train_ids = {101, 102, 103}
validation_ids = {103, 104}
overlap = train_ids & validation_ids
print(overlap)  # {103}
```

**Data Science use:** identify train/test leakage through overlapping
entity IDs, remove duplicates, and compare cohorts.

## Stacks

A stack follows **Last In, First Out (LIFO)**.

``` python
stack = []
stack.append("load")
stack.append("clean")
stack.append("train")
last_step = stack.pop()
print(last_step)  # train
```

**Data Science use:** depth-first search, parsing nested expressions,
and tracking reversible operations.

## Queues

A queue follows **First In, First Out (FIFO)**.

``` python
from collections import deque

jobs = deque(["job_1", "job_2"])
jobs.append("job_3")
next_job = jobs.popleft()
print(next_job)  # job_1
```

**Data Science use:** process incoming events in order, manage work
queues, and implement breadth-first search.

## Linked lists

A linked list stores nodes connected by references. It is useful for
learning pointer-based structures, although Python lists or deques are
usually preferable in everyday application code.

``` python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        node = Node(value)

        if self.head is None:
            self.head = node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = node

    def to_list(self):
        result = []
        current = self.head

        while current:
            result.append(current.value)
            current = current.next

        return result
```

Appending with this implementation is O(n), because it traverses to the
tail. Maintaining a tail pointer makes append O(1).

## Trees and binary search trees

A tree is a hierarchical structure. A binary search tree (BST) orders
values so that smaller keys go left and larger keys go right.

``` text
          8
        /   \
       3     10
      / \      \
     1   6      14
        / \     /
       4   7   13
```

In a balanced BST, search/insert/delete are O(log n); in a skewed BST
they can degrade to O(n). Balanced structures such as AVL or red-black
trees maintain stronger guarantees.

**Data Science use:** decision-tree models use a related hierarchical
split structure. A model's tree is not necessarily a conventional BST
ordered by one key.

## Heaps

A heap maintains a partial ordering, not a fully sorted array.

``` python
import heapq

scores = [0.72, 0.98, 0.86, 0.91]
top_scores = heapq.nlargest(2, scores)
print(top_scores)  # [0.98, 0.91]
```

A heap is particularly useful when the full ranking is not required.

**Data Science use:** top-K recommendations, anomaly ranking, and task
scheduling.

## Graphs

A graph consists of vertices (nodes) and edges (connections). It may be
directed/undirected and weighted/unweighted.

``` python
from collections import deque

graph = {
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A"],
    "D": ["B"],
}


def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return order


print(bfs(graph, "A"))  # ['A', 'B', 'C', 'D']
```

BFS and DFS take O(V + E) time with an adjacency-list representation.

**Data Science use:** social networks, customer-item interactions, fraud
rings, knowledge graphs, and graph-based recommendations.

## Tries

A trie (prefix tree) stores strings by shared prefixes. Each edge
represents a character; a terminal marker identifies a complete word.

``` text
(root)
  |
  c
  |
  a
 / \
t   r
*   *
```

This example stores `cat` and `car`; `*` marks the end of a word.

``` python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root

        for char in word:
            node = node.children.setdefault(char, TrieNode())

        node.is_end = True

    def search(self, word):
        node = self.root

        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]

        return node.is_end

    def starts_with(self, prefix):
        node = self.root

        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]

        return True


trie = Trie()

for word in ["cat", "car", "care"]:
    trie.insert(word)

print(trie.search("car"))       # True
print(trie.search("ca"))        # False
print(trie.starts_with("ca"))   # True
```

For a word of length L, insertion/search/prefix lookup take O(L)
expected time with hash-map children. Space depends on the number of
nodes and alphabet representation.

**Data Science use:** autocomplete for search interfaces, prefix-based
vocabulary lookup, and dictionary support in NLP preprocessing.

------------------------------------------------------------------------

# 7. DSA in the Data Science and ML lifecycle

``` mermaid
flowchart TD
    A[Raw data] --> B[Ingestion and validation]
    B --> C[Preprocessing]
    C --> D[Feature engineering]
    D --> E[Model training]
    E --> F[Evaluation]
    F --> G[Serving and monitoring]

    B -. Queues, hashing .-> B
    C -. Arrays, sorting, sets .-> C
    D -. Hash maps, prefix sums .-> D
    E -. Trees, graphs, matrix operations .-> E
    F -. Heaps, ranking .-> F
    G -. Caches, indexes, queues .-> G
```

  -----------------------------------------------------------------------
  Lifecycle stage         DSA contribution        Practical question
  ----------------------- ----------------------- -----------------------
  Ingestion               Queues, streaming       Can records be
                          buffers                 processed without
                                                  loading everything into
                                                  memory?

  Validation              Sets, hash maps         Are IDs duplicated or
                                                  invalid?

  Preprocessing           Arrays, sorting,        Can repeated scans be
                          hashing                 reduced?

  Feature engineering     Prefix sums, sliding    Can rolling features be
                          windows, maps           computed in one pass?

  Training                Trees, graphs, matrix   What drives training
                          algorithms              time and memory?

  Evaluation              Heaps, sorting          Can top errors or top
                                                  predictions be selected
                                                  efficiently?

  Serving                 Hash maps, caches,      Can a prediction
                          indexes                 request retrieve
                                                  features quickly?
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 8. Visual learning lab

The repository should make abstract structures visible. Use diagrams for
static explanations and small Python visualisers for operations and
traversal order.

## Stack: LIFO

``` text
        pop()
         ↑
      ┌───────┐
      │  C    │  ← top
      ├───────┤
      │  B    │
      ├───────┤
      │  A    │
      └───────┘
        push()
```

## Queue: FIFO

``` text
dequeue()                         enqueue()
   ←                                  ←
 FRONT ┌─────┬─────┬─────┐ REAR
       │  A  │  B  │  C  │
       └─────┴─────┴─────┘
```

## Binary tree traversal

``` text
             A
           /   \
          B     C
         / \     \
        D   E     F

Preorder:   A, B, D, E, C, F
Inorder:    D, B, E, A, C, F
Postorder:  D, E, B, F, C, A
Level order: A, B, C, D, E, F
```

## Graph traversal

``` text
    A ----- B
    |       |
    |       |
    C ----- D

One possible BFS from A: A, B, C, D
```

Traversal order can differ depending on the order in which neighbours
are stored.

## Trie visualisation

``` text
                 (root)
                /      \
               c        d
               |        |
               a        o
             /   \      |
            t     r     g
            *     |     *
                  e
                  *
```

Words represented: `cat`, `car`, `care`, `dog`.

### Visualisation implementation suggestions

-   Use **NetworkX + Plotly** for interactive graph layouts.
-   Use **Plotly** for array operation traces and algorithm runtime
    charts.
-   Use **Graphviz** or Matplotlib for tree/trie node diagrams.
-   Keep visualisation code separate from core data-structure
    implementations.
-   Label nodes, edges, traversal order, and the operation being
    demonstrated.

------------------------------------------------------------------------

# 9. How to run the code

## Requirements

-   Python 3.10 or later
-   Git
-   Optional: Jupyter Notebook for experiments
-   Optional visualisation packages: Plotly, NetworkX, Matplotlib

## Clone the repository

``` bash
git clone https://github.com/<your-username>/dsa-python-for-data-science.git
cd dsa-python-for-data-science
```

## Create and activate a virtual environment

``` bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

## Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Core DSA modules should use the Python standard library wherever
possible. Optional visualisation dependencies can be listed separately
or installed with an extras group.

## Run an example

``` bash
python 04_nonlinear_data_structures/graphs.py
```

## Run tests

``` bash
python -m pytest
```

------------------------------------------------------------------------

# 10. Engineering standards

Every topic should follow this template:

1.  **Problem statement** --- what are we solving?
2.  **Intuition** --- explain it in plain language.
3.  **Visual walkthrough** --- show the structure or algorithm step by
    step.
4.  **Implementation** --- readable, typed Python where appropriate.
5.  **Complexity** --- time and auxiliary space for each operation.
6.  **Edge cases** --- empty input, duplicates, one element, invalid
    input, and large input.
7.  **Tests** --- deterministic tests, including boundary cases.
8.  **Data Science application** --- a realistic use case.
9.  **Benchmark** --- compare alternatives on increasing input sizes.
10. **Further reading** --- trusted documentation or papers.

## Code quality checklist

-   Use descriptive names and small functions.
-   Add docstrings for public classes and functions.
-   Avoid clever code when a clear implementation is better.
-   Validate assumptions (for example, binary search requires sorted
    input).
-   Prefer `collections.deque` over `list.pop(0)` for queues.
-   Use `timeit` for timing and `tracemalloc` for Python allocation
    experiments.
-   Benchmark multiple input sizes and repeat measurements.
-   Do not claim a benchmark proves asymptotic complexity.
-   Separate algorithmic complexity from Python/library overhead.
-   Test correctness before benchmarking.

------------------------------------------------------------------------

# 11. Interview preparation

For every problem, practise this communication sequence:

1.  Restate the problem and clarify constraints.
2.  Walk through a small example.
3.  Explain a brute-force solution.
4.  Identify the bottleneck.
5.  Propose an improved data structure or algorithm.
6.  State time and space complexity.
7.  Code the solution and test edge cases.
8.  Discuss trade-offs and production considerations.

## Core patterns to master

-   Arrays and hashing
-   Two pointers
-   Sliding window
-   Binary search
-   Stack and monotonic stack
-   Queue and BFS
-   DFS and recursion
-   Trees and binary search trees
-   Heaps and top-K
-   Graph traversal and shortest paths
-   Greedy algorithms
-   Backtracking
-   Dynamic programming
-   Sorting and intervals

Do not memorise complexity tables in isolation. Derive the cost from the
operations performed.

------------------------------------------------------------------------

# 12. Suggested projects

  -----------------------------------------------------------------------
  Project                 DSA concepts            Data Science outcome
  ----------------------- ----------------------- -----------------------
  Customer data           Hashing, sets, sorting  Clean entity records
  deduplication engine                            efficiently

  Streaming transaction   Queues, hash maps,      Maintain rolling counts
  monitor                 heaps                   and rank high-risk
                                                  events

  Product recommendation  Graphs, heaps, hashing  Generate and rank
  prototype                                       candidate products

  Search autocomplete     Trie, heaps             Prefix lookup and
                                                  suggestion ranking

  Social network explorer Graphs, BFS, DFS        Analyse reachability
                                                  and connections

  ML prediction ranking   Heaps, sorting, arrays  Return top-K model
  service                                         predictions

  Large CSV processing    Chunking, external      Process data under
  pipeline                sorting, hashing        memory constraints
  -----------------------------------------------------------------------

For each project, include a dataset description, problem statement,
baseline, optimised implementation, complexity analysis, benchmark,
tests, and a short results report.

------------------------------------------------------------------------

# 13. References

-   [Python Tutorial](https://docs.python.org/3/tutorial/)
-   [Python data
    structures](https://docs.python.org/3/tutorial/datastructures.html)
-   [Python
    `collections`](https://docs.python.org/3/library/collections.html)
-   [Python `heapq`](https://docs.python.org/3/library/heapq.html)
-   [Python `timeit`](https://docs.python.org/3/library/timeit.html)
-   [NumPy documentation](https://numpy.org/doc/stable/)
-   [pandas documentation](https://pandas.pydata.org/docs/)
-   [scikit-learn user
    guide](https://scikit-learn.org/stable/user_guide.html)
-   [NetworkX documentation](https://networkx.org/documentation/stable/)
-   [Big-O Cheat Sheet](https://www.bigocheatsheet.com/) --- a quick
    reference; verify details against implementation-specific
    documentation.

------------------------------------------------------------------------

## Contribution roadmap

-   [ ] Complete Python fundamentals with runnable examples and tests.
-   [ ] Add complexity-analysis experiments and benchmark notebooks.
-   [ ] Implement all linear and hash-based structures.
-   [ ] Implement trees, BSTs, heaps, graphs, and tries.
-   [ ] Add searching and sorting algorithms.
-   [ ] Add algorithmic patterns and dynamic programming.
-   [ ] Build Data Science case studies using realistic datasets.
-   [ ] Add interactive visualisations and interview problem sets.
-   [ ] Add CI checks, type checking, and contribution guidelines.

## License

This project is intended for educational use. Add a `LICENSE` file (MIT
is a common choice for open-source educational repositories) and confirm
the chosen license before publishing.

------------------------------------------------------------------------

**If this repository helps you connect computer science fundamentals
with practical Data Science, give it a star and share improvements
through issues or pull requests.**
