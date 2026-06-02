<a id="indice"></a>

# 📘 NB15 — Autovalores e Autovetores

Este notebook desenvolve, com rigor matemático, os conceitos de autovalores,
autovetores, diagonalização, decomposições espectrais e suas interpretações
geométricas — o ápice da álgebra linear.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Autovalores e Autovetores](#definicao)  
   2.1 [Definição](#def)  
   2.2 [Equação Característica](#caracteristica)
3. [Autovetores e Subespaços Invariantes](#subespacos)
4. [Diagonalização](#diagonalizacao)  
   4.1 [Condição Necessária e Suficiente](#condicao-diagonalizacao)  
   4.2 [Interpretação Geométrica](#interpretacao-diagonalizacao)
5. [Matrizes Simétricas](#simetricas)
6. [Decomposição Espectral](#espectral)
7. [Relação com PCA](#pca)
8. [Exemplos Resolvidos](#exemplos)
9. [Exercícios](#exercicios)
10. [Conexões com Outros Capítulos](#conexoes)
11. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

Autovalores e autovetores são o coração da álgebra linear moderna.

Eles aparecem em:

- PCA  
- SVD  
- análise espectral  
- sistemas dinâmicos  
- equações diferenciais  
- otimização quadrática  
- decomposições matriciais  
- teoria da informação  

Este notebook fecha a coleção com o conceito mais profundo da disciplina.

---

<a id="definicao"></a>
# 2. 🧩 Autovalores e Autovetores

<a id="def"></a>
## 2.1 Definição

Um escalar $\lambda$ é um **autovalor** de $A$ se existe um vetor não nulo $v$
tal que:

$$
A v = \lambda v
$$

O vetor $v$ é um **autovetor** associado a $\lambda$.

---

<a id="caracteristica"></a>
## 2.2 Equação Característica

A equação:

$$
\det(A - \lambda I) = 0
$$

define os autovalores de $A$.

O polinômio:

$$
p_A(\lambda) = \det(\lambda I - A)
$$

é o **polinômio característico**.

---

<a id="subespacos"></a>
# 3. 🧭 Autovetores e Subespaços Invariantes

O conjunto de todos os autovetores associados a $\lambda$ (mais o vetor zero)
forma o **autoespaço**:

$$
E_\lambda = \{v : Av = \lambda v\}
$$

É um subespaço de $\mathbb{R}^n$.

---

<a id="diagonalizacao"></a>
# 4. 🧱 Diagonalização

Uma matriz $A$ é diagonalizável se existe uma matriz invertível $P$ tal que:

$$
A = P D P^{-1}
$$

onde $D$ é diagonal.

---

<a id="condicao-diagonalizacao"></a>
## 4.1 Condição Necessária e Suficiente

$A$ é diagonalizável se e somente se:

- possui **$n$ autovetores linearmente independentes**

Equivalentemente:

- a soma das dimensões dos autoespaços é $n$

---

<a id="interpretacao-diagonalizacao"></a>
## 4.2 Interpretação Geométrica

Diagonalizar significa:

- encontrar uma base onde a transformação linear age como **estiramentos**  
- cada autovetor é uma direção preservada  
- cada autovalor é o fator de escala nessa direção  

---

<a id="simetricas"></a>
# 5. 📐 Matrizes Simétricas

Se $A = A^T$, então:

- todos os autovalores são reais  
- autoespaços associados a autovalores distintos são ortogonais  
- $A$ é diagonalizável por matriz ortogonal  

Ou seja:

$$
A = Q D Q^T
$$

Essa é a **decomposição espectral**.

---

<a id="espectral"></a>
# 6. 🎼 Decomposição Espectral

Para matrizes simétricas:

$$
A = \sum_{i=1}^n \lambda_i u_i u_i^T
$$

onde:

- $\lambda_i$ são autovalores  
- $u_i$ são autovetores ortonormais  

Interpretação:

👉 **$A$ é a soma de projeções escaladas**.

---

<a id="pca"></a>
# 7. 📊 Relação com PCA

O PCA usa:

- autovalores da matriz de covariância  
- autovetores como direções principais  

Autovalores grandes → direções de maior variância.

---

<a id="exemplos"></a>
# 8. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Autovalores de matriz $2 \times 2$**

$$
A =
\begin{pmatrix}
2 & 1 \\
1 & 2
\end{pmatrix}
$$

Polinômio característico:

$$
\det(\lambda I - A)
=
(\lambda-2)^2 - 1
=
\lambda^2 - 4\lambda + 3
$$

Raízes:

$$
\lambda = 1, 3
$$

---

### **Exemplo 2 — Autovetores**

Para $\lambda = 3$:

$$
(A - 3I)v = 0
$$

$$
\begin{pmatrix}
-1 & 1 \\
1 & -1
\end{pmatrix}
v = 0
$$

Autovetores: múltiplos de $(1,1)$.

---

### **Exemplo 3 — Decomposição espectral**

$$
A =
\begin{pmatrix}
3 & 0 \\
0 & 1
\end{pmatrix}
=
3 e_1 e_1^T + 1 e_2 e_2^T
$$

---

<a id="exercicios"></a>
# 9. 📝 Exercícios

1. Encontre os autovalores de  
   $$
   A = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}
   $$
2. Determine um autovetor para cada autovalor.  
3. Verifique se $A$ é diagonalizável.  
4. Mostre que matrizes simétricas têm autovalores reais.  
5. Calcule a decomposição espectral de  
   $$
   A = \begin{pmatrix} 2 & 0 \\ 0 & 5 \end{pmatrix}
   $$

---

<a id="conexoes"></a>
# 10. 🔗 Conexões com Outros Capítulos

- [Polinômios e Geometria](ca://s?q=Criar_Notebook_14_Polinomios) — polinômio característico.  
- [Determinante](ca://s?q=Criar_Notebook_13_Determinante) — produto dos autovalores.  
- [Independência Linear e Posto](ca://s?q=Criar_Notebook_12_Posto) — diagonalização exige independência.  
- [Matrizes e Vetores](ca://s?q=Criar_Notebook_10_Matrizes) — estrutura fundamental.  

---

<a id="resumo"></a>
# 11. 🧾 Resumo Final

Este notebook desenvolveu:

- autovalores e autovetores  
- equação característica  
- diagonalização  
- matrizes simétricas  
- decomposição espectral  
- relação com PCA  
- exemplos e exercícios  

Este capítulo encerra a coleção com o conceito mais profundo da álgebra linear.

---

[↑ Voltar ao Índice](#indice)
