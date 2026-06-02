<a id="indice"></a>

# 📘 NB14 — Polinômios e Geometria

Este notebook desenvolve, com rigor matemático, os fundamentos de polinômios,
raízes, fatoração, curvas polinomiais, geometria analítica e conexões profundas
com álgebra linear e autovalores.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Polinômios](#polinomios)  
   2.1 [Definição](#def-polinomio)  
   2.2 [Grau](#grau)  
   2.3 [Operações](#operacoes)
3. [Raízes e Fatoração](#raizes)  
   3.1 [Teorema Fundamental da Álgebra](#tfa)  
   3.2 [Fatoração em Termos Lineares](#fatoracao)
4. [Derivadas de Polinômios](#derivadas)
5. [Curvas Polinomiais e Geometria](#curvas)  
   5.1 [Retas](#retas)  
   5.2 [Parábolas](#parabolas)  
   5.3 [Cônicas](#conicas)
6. [Polinômios Característicos](#caracteristico)
7. [Autovalores e Polinômios](#autovalores)
8. [Exemplos Resolvidos](#exemplos)
9. [Exercícios](#exercicios)
10. [Conexões com Outros Capítulos](#conexoes)
11. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

Polinômios são objetos centrais na matemática:

- descrevem curvas  
- aproximam funções  
- definem autovalores  
- aparecem em equações diferenciais  
- modelam fenômenos físicos  
- são base para interpolação e regressão  

Este notebook conecta polinômios com geometria e álgebra linear.

---

<a id="polinomios"></a>
# 2. 🧮 Polinômios

<a id="def-polinomio"></a>
## 2.1 Definição

Um polinômio em $x$ é:

$$
p(x) = a_0 + a_1 x + a_2 x^2 + \cdots + a_n x^n
$$

com $a_n \ne 0$.

---

<a id="grau"></a>
## 2.2 Grau

O **grau** é o maior expoente com coeficiente não nulo.

---

<a id="operacoes"></a>
## 2.3 Operações

### Soma

$$
(p+q)(x) = p(x) + q(x)
$$

### Produto

$$
(pq)(x) = \sum_{i+j=k} a_i b_j x^k
$$

### Composição

$$
(p \circ q)(x) = p(q(x))
$$

---

<a id="raizes"></a>
# 3. 🔍 Raízes e Fatoração

<a id="tfa"></a>
## 3.1 Teorema Fundamental da Álgebra

Todo polinômio de grau $n \ge 1$ tem **exatamente $n$ raízes complexas**,
contadas com multiplicidade.

---

<a id="fatoracao"></a>
## 3.2 Fatoração em Termos Lineares

Se $p$ tem raízes $r_1,\dots,r_n$:

$$
p(x) = a_n (x - r_1)(x - r_2)\cdots(x - r_n)
$$

---

<a id="derivadas"></a>
# 4. 📐 Derivadas de Polinômios

A derivada é:

$$
p'(x) = a_1 + 2a_2 x + \cdots + n a_n x^{n-1}
$$

Propriedades:

- $p'$ tem grau $n-1$  
- raízes de $p'$ estão entre raízes de $p$ (Teorema de Rolle)  

---

<a id="curvas"></a>
# 5. 🧭 Curvas Polinomiais e Geometria

<a id="retas"></a>
## 5.1 Retas

Polinômios de grau 1:

$$
ax + b
$$

representam retas.

---

<a id="parabolas"></a>
## 5.2 Parábolas

Polinômios de grau 2:

$$
ax^2 + bx + c
$$

representam parábolas.

Vértice:

$$
x_v = -\frac{b}{2a}
$$

---

<a id="conicas"></a>
## 5.3 Cônicas

Equações quadráticas em duas variáveis:

$$
Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
$$

representam:

- elipses  
- parábolas  
- hipérboles  

A classificação depende do discriminante:

$$
\Delta = B^2 - 4AC
$$

---

<a id="caracteristico"></a>
# 6. 🧩 Polinômios Característicos

Para uma matriz $A \in \mathbb{R}^{n \times n}$:

$$
p_A(\lambda) = \det(\lambda I - A)
$$

É um polinômio de grau $n$.

---

<a id="autovalores"></a>
# 7. 🎯 Autovalores e Polinômios

Os autovalores de $A$ são as raízes do polinômio característico:

$$
p_A(\lambda) = 0
$$

Conexão profunda:

👉 **autovalores são raízes de um polinômio**.

---

<a id="exemplos"></a>
# 8. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Fatoração**

$$
p(x) = x^2 - 5x + 6
$$

Raízes: 2 e 3.

$$
p(x) = (x-2)(x-3)
$$

---

### **Exemplo 2 — Polinômio característico**

Para:

$$
A =
\begin{pmatrix}
2 & 1 \\
0 & 3
\end{pmatrix}
$$

$$
p_A(\lambda) = (\lambda-2)(\lambda-3)
$$

Autovalores: 2 e 3.

---

### **Exemplo 3 — Parábola**

$$
f(x) = x^2 - 4x + 1
$$

Vértice:

$$
x_v = 2
$$

---

<a id="exercicios"></a>
# 9. 📝 Exercícios

1. Fatore $x^2 + x - 6$.  
2. Encontre o vértice de $f(x) = 3x^2 - 6x + 2$.  
3. Determine o polinômio característico de  
   $$
   A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
   $$
4. Classifique a cônica  
   $$
   4x^2 + 9y^2 - 36 = 0
   $$
5. Mostre que todo polinômio de grau ímpar tem pelo menos uma raiz real.  

---

<a id="conexoes"></a>
# 10. 🔗 Conexões com Outros Capítulos

- [Determinante](ca://s?q=Criar_Notebook_13_Determinante) — usado no polinômio característico.  
- [Autovalores e Autovetores](ca://s?q=Criar_Notebook_15_Autovalores) — raízes do polinômio característico.  
- [Independência Linear e Posto](ca://s?q=Criar_Notebook_12_Posto) — fatoração e multiplicidade.  
- [Funções e Convexidade](ca://s?q=Criar_Notebook_3_Propriedades_de_Funcoes) — análise de curvas polinomiais.  

---

<a id="resumo"></a>
# 11. 🧾 Resumo Final

Este notebook desenvolveu:

- definição e operações com polinômios  
- raízes e fatoração  
- curvas polinomiais  
- cônicas  
- polinômio característico  
- relação com autovalores  
- exemplos e exercícios  

Polinômios conectam geometria, álgebra linear e análise de forma profunda.

---

[↑ Voltar ao Índice](#indice)
