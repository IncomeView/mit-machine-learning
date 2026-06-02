<a id="indice"></a>

# 📘 NB10 — Matrizes e Vetores

Este notebook desenvolve, com rigor matemático, os fundamentos de vetores,
matrizes, transformações lineares, operações matriciais e propriedades
estruturais essenciais para álgebra linear moderna.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Vetores em $\mathbb{R}^n$](#vetores)
3. [Matrizes](#matrizes)  
   3.1 [Definição](#def-matriz)  
   3.2 [Notação e Índices](#indices)  
   3.3 [Matriz Identidade](#identidade)
4. [Operações com Matrizes](#operacoes)  
   4.1 [Soma](#soma)  
   4.2 [Multiplicação por Escalar](#escalar)  
   4.3 [Multiplicação Matriz–Vetor](#matriz-vetor)
5. [Transformações Lineares](#transformacoes)
6. [Matriz Inversa](#inversa)
7. [Espaços de Linhas e Colunas](#espacos)
8. [Exemplos Resolvidos](#exemplos)
9. [Exercícios](#exercicios)
10. [Conexões com Outros Capítulos](#conexoes)
11. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

Matrizes são a linguagem fundamental da álgebra linear.

Elas representam:

- transformações lineares  
- sistemas lineares  
- projeções  
- rotações  
- mudanças de base  
- decomposições espectrais  

Este notebook estabelece a base para:

- independência linear  
- posto  
- determinante  
- autovalores e autovetores  
- decomposições matriciais  

---

<a id="vetores"></a>
# 2. 🧭 Vetores em $\mathbb{R}^n$

Um vetor coluna é:

$$
v =
\begin{pmatrix}
v_1 \\
\vdots \\
v_n
\end{pmatrix}
$$

Interpretado como:

- direção  
- magnitude  
- ponto em $\mathbb{R}^n$  
- coordenadas em uma base  

---

<a id="matrizes"></a>
# 3. 🧱 Matrizes

<a id="def-matriz"></a>
## 3.1 Definição

Uma matriz $A \in \mathbb{R}^{m \times n}$ é uma tabela de números reais:

$$
A =
\begin{pmatrix}
a_{11} & \dots & a_{1n} \\
\vdots & \ddots & \vdots \\
a_{m1} & \dots & a_{mn}
\end{pmatrix}
$$

---

<a id="indices"></a>
## 3.2 Notação e Índices

- $a_{ij}$ é o elemento da **linha $i$** e **coluna $j$**.  
- $A_{i,:}$ é a linha $i$.  
- $A_{:,j}$ é a coluna $j$.  

---

<a id="identidade"></a>
## 3.3 Matriz Identidade

A matriz identidade $I_n$ é:

$$
(I_n)_{ij} =
\begin{cases}
1 & i = j \\
0 & i \ne j
\end{cases}
$$

Ela é o “1” da álgebra linear:

$$
I_n v = v
$$

---

<a id="operacoes"></a>
# 4. 🔧 Operações com Matrizes

<a id="soma"></a>
## 4.1 Soma

$$
A + B = (a_{ij} + b_{ij})
$$

Definida apenas quando $A$ e $B$ têm o mesmo tamanho.

---

<a id="escalar"></a>
## 4.2 Multiplicação por Escalar

$$
\alpha A = (\alpha a_{ij})
$$

---

<a id="matriz-vetor"></a>
## 4.3 Multiplicação Matriz–Vetor

Se $A$ é $m \times n$ e $v$ é $n \times 1$:

$$
Av = \text{combinação linear das colunas de } A
$$

Explicitamente:

$$
Av =
v_1 A_{:,1} + v_2 A_{:,2} + \cdots + v_n A_{:,n}
$$

---

<a id="transformacoes"></a>
# 5. 🔄 Transformações Lineares

Toda matriz $A$ define uma transformação linear:

$$
T_A(x) = Ax
$$

Propriedades:

- preserva somas  
- preserva multiplicação por escalar  

Transformações lineares incluem:

- rotações  
- reflexões  
- projeções  
- cisalhamentos  

---

<a id="inversa"></a>
# 6. 🔁 Matriz Inversa

Uma matriz quadrada $A$ é invertível se existe $A^{-1}$ tal que:

$$
AA^{-1} = A^{-1}A = I
$$

Condições equivalentes:

- $\det(A) \ne 0$  
- $\text{rank}(A) = n$  
- colunas são linearmente independentes  

---

<a id="espacos"></a>
# 7. 🧩 Espaços de Linhas e Colunas

### **Espaço coluna**

$$
\text{Col}(A) = \{Ax : x \in \mathbb{R}^n\}
$$

### **Espaço linha**

$$
\text{Row}(A) = \{x^T A : x \in \mathbb{R}^m\}
$$

Ambos têm dimensão igual ao **posto** de $A$.

---

<a id="exemplos"></a>
# 8. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Matriz–vetor**

$$
A =
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix},
\quad
v =
\begin{pmatrix}
1 \\
-1
\end{pmatrix}
$$

$$
Av =
\begin{pmatrix}
1\cdot1 + 2(-1) \\
3\cdot1 + 4(-1)
\end{pmatrix}
=
\begin{pmatrix}
-1 \\
-1
\end{pmatrix}
$$

---

### **Exemplo 2 — Inversa**

Para:

$$
A =
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$

Se $ad - bc \ne 0$:

$$
A^{-1} = \frac{1}{ad - bc}
\begin{pmatrix}
d & -b \\
-c & a
\end{pmatrix}
$$

---

<a id="exercicios"></a>
# 9. 📝 Exercícios

1. Calcule $Av$ para  
   $$
   A = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix},
   \quad
   v = \begin{pmatrix} 1 \\ 4 \end{pmatrix}
   $$
2. Determine se  
   $$
   A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}
   $$
   é invertível.  
3. Encontre o espaço coluna de  
   $$
   A = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{pmatrix}
   $$
4. Mostre que $T(x) = Ax$ é linear.  
5. Mostre que se $A$ é invertível, então $A^{-1}$ é única.  

---

<a id="conexoes"></a>
# 10. 🔗 Conexões com Outros Capítulos

- [Multiplicação de Matrizes](ca://s?q=Criar_Notebook_11_Multiplicacao) — generaliza operações.  
- [Independência Linear e Posto](ca://s?q=Criar_Notebook_12_Posto) — estrutura dos espaços coluna e linha.  
- [Determinante](ca://s?q=Criar_Notebook_13_Determinante) — invertibilidade.  
- [Autovalores e Autovetores](ca://s?q=Criar_Notebook_15_Autovalores) — análise espectral.  

---

<a id="resumo"></a>
# 11. 🧾 Resumo Final

Este notebook desenvolveu:

- vetores  
- matrizes  
- operações matriciais  
- transformações lineares  
- matriz inversa  
- espaços coluna e linha  
- exemplos e exercícios  

Este capítulo é a base da álgebra linear moderna.

---

[↑ Voltar ao Índice](#indice)
