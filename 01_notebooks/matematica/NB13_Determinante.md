<a id="indice"></a>

# 📘 NB13 — Determinante

Este notebook desenvolve, com rigor matemático, o conceito de determinante:
definição, propriedades fundamentais, interpretação geométrica, cálculo em
matrizes pequenas e implicações estruturais em álgebra linear.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Determinante de Matrizes $2 \times 2$](#2x2)
3. [Determinante de Matrizes $3 \times 3$](#3x3)
4. [Definição Geral via Expansão de Laplace](#laplace)
5. [Propriedades Fundamentais](#propriedades)
6. [Interpretação Geométrica](#interpretacao)
7. [Determinante e Invertibilidade](#invertibilidade)
8. [Determinante e Autovalores](#autovalores)
9. [Exemplos Resolvidos](#exemplos)
10. [Exercícios](#exercicios)
11. [Conexões com Outros Capítulos](#conexoes)
12. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

O determinante é uma das quantidades mais importantes da álgebra linear.

Ele indica:

- se uma matriz é invertível  
- o fator de escala de volume de uma transformação linear  
- se vetores são linearmente independentes  
- se um sistema linear tem solução única  
- o produto dos autovalores  

Este notebook formaliza o determinante com rigor matemático.

---

<a id="2x2"></a>
# 2. 🧮 Determinante de Matrizes $2 \times 2$

Para:

$$
A =
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$

o determinante é:

$$
\det(A) = ad - bc
$$

---

<a id="3x3"></a>
# 3. 🧮 Determinante de Matrizes $3 \times 3$

Para:

$$
A =
\begin{pmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{pmatrix}
$$

o determinante é:

$$
\det(A)
=
a(ei - fh)
- b(di - fg)
+ c(dh - eg)
$$

---

<a id="laplace"></a>
# 4. 📐 Definição Geral via Expansão de Laplace

Para uma matriz $A \in \mathbb{R}^{n \times n}$, o determinante pode ser definido
recursivamente:

$$
\det(A) = \sum_{j=1}^n (-1)^{1+j} a_{1j} \det(A_{1j})
$$

onde $A_{1j}$ é a matriz obtida removendo a linha 1 e a coluna $j$.

---

<a id="propriedades"></a>
# 5. 📏 Propriedades Fundamentais

### **1. Determinante do produto**

$$
\det(AB) = \det(A)\det(B)
$$

### **2. Determinante da inversa**

Se $A$ é invertível:

$$
\det(A^{-1}) = \frac{1}{\det(A)}
$$

### **3. Troca de linhas**

Trocar duas linhas muda o sinal:

$$
\det(A_{\text{com linhas trocadas}}) = -\det(A)
$$

### **4. Linha multiplicada por escalar**

Se multiplicamos uma linha por $c$:

$$
\det(A') = c \det(A)
$$

### **5. Linha igual a combinação linear de outras**

Se uma linha é combinação linear das outras:

$$
\det(A) = 0
$$

### **6. Matriz triangular**

Se $A$ é triangular:

$$
\det(A) = \prod_{i=1}^n a_{ii}
$$

---

<a id="interpretacao"></a>
# 6. 🧭 Interpretação Geométrica

O determinante mede o **fator de escala de volume** da transformação linear
$x \mapsto Ax$.

- Se $\det(A) = 2$: o volume dobra  
- Se $\det(A) = 0$: o volume colapsa → transformação não é invertível  
- Se $\det(A) < 0$: há inversão de orientação  

Em $\mathbb{R}^2$, o determinante mede a área do paralelogramo formado pelas colunas.

Em $\mathbb{R}^3$, mede o volume do paralelepípedo.

---

<a id="invertibilidade"></a>
# 7. 🔁 Determinante e Invertibilidade

Uma matriz quadrada $A$ é invertível se e somente se:

$$
\det(A) \ne 0
$$

Equivalentemente:

- colunas são independentes  
- linhas são independentes  
- posto = $n$  
- transformação não colapsa volume  

---

<a id="autovalores"></a>
# 8. 🧩 Determinante e Autovalores

Se $A$ tem autovalores $\lambda_1,\dots,\lambda_n$:

$$
\det(A) = \lambda_1 \lambda_2 \cdots \lambda_n
$$

E o traço é:

$$
\text{tr}(A) = \lambda_1 + \cdots + \lambda_n
$$

---

<a id="exemplos"></a>
# 9. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Determinante $2 \times 2$**

$$
A =
\begin{pmatrix}
3 & 1 \\
2 & 4
\end{pmatrix}
$$

$$
\det(A) = 3\cdot4 - 1\cdot2 = 10
$$

---

### **Exemplo 2 — Determinante triangular**

$$
A =
\begin{pmatrix}
2 & * & * \\
0 & 3 & * \\
0 & 0 & 5
\end{pmatrix}
$$

$$
\det(A) = 2\cdot3\cdot5 = 30
$$

---

### **Exemplo 3 — Determinante e volume**

As colunas de:

$$
A =
\begin{pmatrix}
1 & 0 \\
1 & 2
\end{pmatrix}
$$

formam um paralelogramo de área:

$$
\det(A) = 2
$$

---

<a id="exercicios"></a>
# 10. 📝 Exercícios

1. Calcule o determinante de  
   $$
   \begin{pmatrix}
   1 & 2 \\
   3 & 4
   \end{pmatrix}
   $$
2. Mostre que se duas linhas são iguais, $\det(A)=0$.  
3. Calcule o determinante de  
   $$
   \begin{pmatrix}
   2 & 1 & 0 \\
   0 & 3 & 1 \\
   0 & 0 & 4
   \end{pmatrix}
   $$
4. Mostre que $\det(A^T) = \det(A)$.  
5. Mostre que se $A$ é invertível, então $\det(A^{-1}) = 1/\det(A)$.  

---

<a id="conexoes"></a>
# 11. 🔗 Conexões com Outros Capítulos

- [Independência Linear e Posto](ca://s?q=Criar_Notebook_12_Posto) — determinante detecta dependência.  
- [Matrizes e Vetores](ca://s?q=Criar_Notebook_10_Matrizes) — estrutura básica.  
- [Multiplicação de Matrizes](ca://s?q=Criar_Notebook_11_Multiplicacao) — determinante do produto.  
- [Autovalores e Autovetores](ca://s?q=Criar_Notebook_15_Autovalores) — determinante é o produto dos autovalores.  

---

<a id="resumo"></a>
# 12. 🧾 Resumo Final

Este notebook desenvolveu:

- determinante $2 \times 2$ e $3 \times 3$  
- definição geral via Laplace  
- propriedades fundamentais  
- interpretação geométrica  
- relação com invertibilidade  
- relação com autovalores  
- exemplos e exercícios  

O determinante é uma das quantidades centrais da álgebra linear.

---

[↑ Voltar ao Índice](#indice)
