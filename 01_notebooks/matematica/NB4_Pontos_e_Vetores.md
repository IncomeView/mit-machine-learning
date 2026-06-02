<a id="indice"></a>

# 📘 NB4 — Pontos e Vetores

Este notebook desenvolve, com rigor matemático, os fundamentos de pontos,
vetores, normas, produtos internos, projeções e distâncias — a base da geometria
analítica e da álgebra linear moderna.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Pontos em $\mathbb{R}^n$](#pontos)
3. [Vetores](#vetores)  
   3.1 [Definição](#def-vetor)  
   3.2 [Operações Básicas](#operacoes)  
   3.3 [Combinações Lineares](#combinacoes)
4. [Normas](#normas)  
   4.1 [Norma Euclidiana](#norma-euclidiana)  
   4.2 [Propriedades](#propriedades-norma)  
   4.3 [Outras Normas](#outras-normas)
5. [Produto Interno](#produto-interno)  
   5.1 [Definição](#def-produto)  
   5.2 [Interpretação Geométrica](#interpretacao-produto)  
   5.3 [Ortogonalidade](#ortogonalidade)
6. [Projeções](#projecoes)  
   6.1 [Projeção de um Vetor em Outro](#proj-vetor)  
   6.2 [Projeção em Subespaços](#proj-subespaco)
7. [Distâncias](#distancias)
8. [Exemplos Resolvidos](#exemplos)
9. [Exercícios](#exercicios)
10. [Conexões com Outros Capítulos](#conexoes)
11. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

Pontos e vetores são a linguagem fundamental da geometria analítica e da álgebra
linear.  
Eles permitem descrever:

- posições,  
- direções,  
- deslocamentos,  
- distâncias,  
- projeções,  
- ângulos,  
- hiperplanos,  
- transformações lineares.

Este notebook estabelece a base geométrica para todos os capítulos seguintes.

---

<a id="pontos"></a>
# 2. 📍 Pontos em $\mathbb{R}^n$

Um ponto em $\mathbb{R}^n$ é um objeto puramente posicional:

$$
x = (x_1, x_2, \dots, x_n)
$$

Ele **não** tem direção nem magnitude.

---

<a id="vetores"></a>
# 3. 🧭 Vetores

<a id="def-vetor"></a>
## 3.1 Definição

Um vetor é um objeto com:

- direção  
- sentido  
- magnitude  

Representado por:

$$
v = (v_1, v_2, \dots, v_n)
$$

---

<a id="operacoes"></a>
## 3.2 Operações Básicas

### **Soma**

$$
u + v = (u_1 + v_1, \dots, u_n + v_n)
$$

### **Multiplicação por escalar**

$$
\alpha v = (\alpha v_1, \dots, \alpha v_n)
$$

### **Vetor diferença**

Entre pontos $a$ e $b$:

$$
b - a
$$

---

<a id="combinacoes"></a>
## 3.3 Combinações Lineares

Uma combinação linear de vetores $v_1,\dots,v_k$ é:

$$
c_1 v_1 + \cdots + c_k v_k
$$

É a base de:

- subespaços  
- independência linear  
- posto  
- projeções  
- decomposições  

---

<a id="normas"></a>
# 4. 📏 Normas

<a id="norma-euclidiana"></a>
## 4.1 Norma Euclidiana

A norma mais comum é:

$$
\|v\| = \sqrt{v_1^2 + \dots + v_n^2}
$$

---

<a id="propriedades-norma"></a>
## 4.2 Propriedades

1. **Não-negatividade**  
   $$
   \|v\| \ge 0
   $$

2. **Homogeneidade**  
   $$
   \|\alpha v\| = |\alpha| \|v\|
   $$

3. **Desigualdade triangular**  
   $$
   \|u + v\| \le \|u\| + \|v\|
   $$

---

<a id="outras-normas"></a>
## 4.3 Outras Normas

### Norma $L^1$

$$
\|v\|_1 = \sum |v_i|
$$

### Norma $L^\infty$

$$
\|v\|_\infty = \max_i |v_i|
$$

---

<a id="produto-interno"></a>
# 5. 🎯 Produto Interno

<a id="def-produto"></a>
## 5.1 Definição

O produto interno é:

$$
u \cdot v = \sum_{i=1}^n u_i v_i
$$

---

<a id="interpretacao-produto"></a>
## 5.2 Interpretação Geométrica

$$
u \cdot v = \|u\| \|v\| \cos(\theta)
$$

onde $\theta$ é o ângulo entre os vetores.

---

<a id="ortogonalidade"></a>
## 5.3 Ortogonalidade

Vetores são ortogonais quando:

$$
u \cdot v = 0
$$

---

<a id="projecoes"></a>
# 6. 📐 Projeções

<a id="proj-vetor"></a>
## 6.1 Projeção de um Vetor em Outro

$$
\text{proj}_u(v)
=
\frac{u \cdot v}{\|u\|^2} u
$$

---

<a id="proj-subespaco"></a>
## 6.2 Projeção em Subespaços

Se $U$ é um subespaço com base ortonormal $\{u_1,\dots,u_k\}$:

$$
\text{proj}_U(v)
=
\sum_{i=1}^k (v \cdot u_i) u_i
$$

---

<a id="distancias"></a>
# 7. 📏 Distâncias

A distância entre pontos $a$ e $b$ é:

$$
\|b - a\|
$$

---

<a id="exemplos"></a>
# 8. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Norma**

$$
\|(3,4)\| = 5
$$

---

### **Exemplo 2 — Produto interno**

$$
(1,2,3)\cdot(4,0,-1) = 1\cdot4 + 2\cdot0 + 3(-1) = 1
$$

---

### **Exemplo 3 — Projeção**

Projete $v=(2,1)$ em $u=(1,1)$:

$$
\text{proj}_u(v)
=
\frac{3}{2}(1,1)
=
\left(\frac{3}{2},\frac{3}{2}\right)
$$

---

<a id="exercicios"></a>
# 9. 📝 Exercícios

1. Calcule $\|(1,-2,2)\|$.  
2. Determine se $(1,2)$ e $(-4,2)$ são ortogonais.  
3. Projete $(3,0)$ em $(1,1)$.  
4. Mostre que $\|u+v\|^2 = \|u\|^2 + \|v\|^2 + 2u\cdot v$.  
5. Mostre que $\|u-v\|^2 = \|u\|^2 + \|v\|^2 - 2u\cdot v$.  

---

<a id="conexoes"></a>
# 10. 🔗 Conexões com Outros Capítulos

- NB5: planos e hiperplanos usam vetores normais.  
- NB10–NB11: matrizes atuam sobre vetores.  
- NB12–NB15: vetores são base para independência, posto e autovalores.  

---

<a id="resumo"></a>
# 11. 🧾 Resumo Final

Este notebook desenvolveu:

- pontos e vetores  
- normas  
- produtos internos  
- projeções  
- distâncias  
- exemplos e exercícios  

Esses conceitos são a base da geometria analítica e da álgebra linear.

---

[↑ Voltar ao Índice](#indice)
