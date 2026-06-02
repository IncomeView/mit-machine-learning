<a id="indice"></a>

# 📘 NB2 — Somas, Produtos e Séries Geométricas

Este notebook desenvolve, com rigor matemático, os fundamentos de somatórios,
produtos e séries geométricas — conceitos essenciais em análise, probabilidade,
otimização e algoritmos.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Somatórios](#somatorios)  
   2.1 [Definição Formal](#definicao-soma)  
   2.2 [Propriedades Fundamentais](#propriedades-soma)  
   2.3 [Teoremas Importantes](#teoremas-soma)  
   2.4 [Exemplos Resolvidos](#exemplos-soma)  
   2.5 [Exercícios](#exercicios-soma)
3. [Produtos](#produtos)  
   3.1 [Definição Formal](#definicao-produto)  
   3.2 [Propriedades](#propriedades-produto)  
   3.3 [Exemplos Resolvidos](#exemplos-produto)  
   3.4 [Exercícios](#exercicios-produto)
4. [Séries Geométricas](#series)  
   4.1 [Definição](#definicao-series)  
   4.2 [Soma Finita](#soma-finita)  
   4.3 [Soma Infinita](#soma-infinita)  
   4.4 [Convergência](#convergencia)  
   4.5 [Exemplos Resolvidos](#exemplos-series)  
   4.6 [Exercícios](#exercicios-series)
5. [Conexões com Outros Capítulos](#conexoes)
6. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

Somatórios, produtos e séries geométricas são ferramentas fundamentais em:

- análise matemática  
- probabilidade  
- estatística  
- otimização  
- algoritmos iterativos  
- modelos probabilísticos  
- análise de convergência  

Eles aparecem em:

- definição de médias e variâncias  
- gradientes e funções de custo  
- likelihoods  
- métodos iterativos  
- análise de estabilidade  
- processos estocásticos  

Este notebook estabelece a base para todos esses temas.

---

<a id="somatorios"></a>
# 2. 🧮 Somatórios

<a id="definicao-soma"></a>
## 2.1 Definição Formal

O somatório é uma notação compacta para somas repetidas:

$$
\sum_{i=1}^n a_i = a_1 + a_2 + \dots + a_n
$$

---

<a id="propriedades-soma"></a>
## 2.2 Propriedades Fundamentais

### **Linearidade**

$$
\sum_{i=1}^n (a_i + b_i)
=
\sum_{i=1}^n a_i
+
\sum_{i=1}^n b_i
$$

### **Constante para fora**

$$
\sum_{i=1}^n c\,a_i
=
c \sum_{i=1}^n a_i
$$

### **Separação de índices**

Se $m < k$:

$$
\sum_{i=m}^k a_i
=
\sum_{i=m}^j a_i
+
\sum_{i=j+1}^k a_i
$$

---

<a id="teoremas-soma"></a>
## 2.3 Teoremas Importantes

### **Teorema 1 — Soma dos primeiros $n$ inteiros**

$$
\sum_{i=1}^n i = \frac{n(n+1)}{2}
$$

### **Teorema 2 — Soma dos quadrados**

$$
\sum_{i=1}^n i^2 = \frac{n(n+1)(2n+1)}{6}
$$

### **Teorema 3 — Soma dos cubos**

$$
\sum_{i=1}^n i^3 = \left( \frac{n(n+1)}{2} \right)^2
$$

---

<a id="exemplos-soma"></a>
## 2.4 Exemplos Resolvidos

### **Exemplo 1**

Calcule:

$$
\sum_{i=1}^5 i
$$

**Solução:**

$$
\frac{5(6)}{2} = 15
$$

---

### **Exemplo 2**

Calcule:

$$
\sum_{i=1}^4 (2i + 1)
$$

**Solução:**

$$
\sum 2i + \sum 1 = 2\sum i + 4 = 2(10) + 4 = 24
$$

---

<a id="exercicios-soma"></a>
## 2.5 Exercícios

1. Calcule $\sum_{i=1}^{20} i$.  
2. Prove que $\sum_{i=1}^n (2i-1) = n^2$.  
3. Mostre que $\sum_{i=1}^n (a+bi) = na + b\frac{n(n+1)}{2}$.

---

<a id="produtos"></a>
# 3. 🧮 Produtos

<a id="definicao-produto"></a>
## 3.1 Definição Formal

O produto representa multiplicações repetidas:

$$
\prod_{i=1}^n a_i = a_1 a_2 \dots a_n
$$

---

<a id="propriedades-produto"></a>
## 3.2 Propriedades

### **Constante para dentro**

$$
\prod_{i=1}^n c a_i = c^n \prod_{i=1}^n a_i
$$

### **Produto de potências**

$$
\prod_{i=1}^n r^i = r^{\frac{n(n+1)}{2}}
$$

---

<a id="exemplos-produto"></a>
## 3.3 Exemplos Resolvidos

### **Exemplo 1**

$$
\prod_{i=1}^4 2 = 2^4 = 16
$$

### **Exemplo 2**

$$
\prod_{i=1}^3 i = 1 \cdot 2 \cdot 3 = 6
$$

---

<a id="exercicios-produto"></a>
## 3.4 Exercícios

1. Calcule $\prod_{i=1}^5 3$.  
2. Mostre que $\prod_{i=1}^n (1+r) = (1+r)^n$.  

---

<a id="series"></a>
# 4. 🧮 Séries Geométricas

<a id="definicao-series"></a>
## 4.1 Definição

Uma série geométrica tem a forma:

$$
a + ar + ar^2 + \dots + ar^{n-1}
$$

---

<a id="soma-finita"></a>
## 4.2 Soma Finita

Se $r \neq 1$:

$$
\sum_{k=0}^{n-1} ar^k
=
a \frac{1 - r^n}{1 - r}
$$

---

<a id="soma-infinita"></a>
## 4.3 Soma Infinita

Se $|r| < 1$:

$$
\sum_{k=0}^{\infty} ar^k = \frac{a}{1 - r}
$$

---

<a id="convergencia"></a>
## 4.4 Convergência

A série geométrica converge **somente** quando:

$$
|r| < 1
$$

---

<a id="exemplos-series"></a>
## 4.5 Exemplos Resolvidos

### **Exemplo 1**

$$
\sum_{k=0}^{3} 2(0.5)^k = 3.75
$$

### **Exemplo 2**

$$
\sum_{k=0}^{\infty} 0.9^k = 10
$$

---

<a id="exercicios-series"></a>
## 4.6 Exercícios

1. Calcule $\sum_{k=0}^{5} 3(2)^k$.  
2. Determine se $\sum_{k=0}^{\infty} (-0.8)^k$ converge.  
3. Prove a fórmula da soma finita.  

---

<a id="conexoes"></a>
# 5. 🔗 Conexões com Outros Capítulos

- NB3: funções e convexidade usam somatórios.  
- NB6–NB7: séries geométricas aparecem em probabilidade.  
- NB8–NB9: análise de convergência em otimização.  
- NB12–NB15: produtos e somatórios aparecem em álgebra linear.  

---

<a id="resumo"></a>
# 6. 🧾 Resumo Final

Este notebook desenvolveu:

- somatórios  
- produtos  
- séries geométricas  
- teoremas fundamentais  
- exemplos e exercícios  

Esses conceitos são essenciais para análise, probabilidade e otimização.

---

[↑ Voltar ao Índice](#indice)
