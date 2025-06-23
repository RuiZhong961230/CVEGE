# CVEGE
Chaotic vegetation evolution: leveraging multiple seeding strategies and a mutation module for global optimization problems

## Abstract 
This paper focuses on improving the overall performance of the vegetation evolution (VEGE) algorithm and proposes a chaotic VEGE with multiple seeding strategies and a mutation module (CVEGE). While the original VEGE exhibits robust exploitation capabilities, it falls short in terms of exploration and overcoming local optima. Thus, we introduce the chaotic local search operators, multiple seed dispersion strategies, and a unique mutation module to address these mentioned limitations. Furthermore, we incorporate a simplified sigmoid transfer function into CVEGE and propose a binary variant known as binary chaotic vegetation evolution (BCVEGE). In numerical experiments, we evaluate CVEGE on 10-D, 30-D, 50-D, and 100-D CEC2020 benchmark functions, as well as four engineering optimization problems. Additionally, BCVEGE is subjected to testing on two combinatorial optimization problems: wrapper-based feature selection tasks and classic 0/1 knapsack problems. Here, we employ two classic algorithms (i.e. differential evolution and particle swarm optimization) and seven state-of-the-art competitor algorithms including the original VEGE as the competitor algorithms. The sufficient numerical experiments and statistical analysis practically show that our proposal: CVEGE and BCVEGE, are competitive with compared algorithms. Furthermore, the demonstrated performance and scalability of CVEGE and BCVEGE suggest their potential utility across a wide range of optimization tasks.

## Citation
@Article{Zhong:23,  
AUTHOR = {Rui Zhong and Chao Zhang and Jun Yu},  
TITLE = {Chaotic Vegetation Evolution: Leveraging Multiple Seeding Strategies and a Mutation Module for Global Optimization Problems},  
JOURNAL = {Evolutionary Intelligence},  
VOLUME = {17},  
YEAR = {2024},  
ISSN = {},  
PAGES = {2387–2411},  
DOI = {https://doi.org/10.1007/s12065-023-00892-6 },  
}


## Datasets and Libraries
CEC benchmarks and Engineering problems are provided by opfunu==1.0.0 and enoppy==0.1.1 libraries, respectively.

## Contact
If you have any questions, please don't hesitate to contact zhongrui[at]iic.hokudai.ac.jp
