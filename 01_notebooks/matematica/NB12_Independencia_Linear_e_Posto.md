<a id="indice"></a>

# 📘 NB12 — Independência Linear e Posto

Este notebook desenvolve, com rigor matemático, os conceitos fundamentais de
independência linear, subespaços, dimensão, posto de uma matriz e suas
implicações estruturais na álgebra linear.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Combinações Lineares](#combinacoes)
3. [Independência Linear](#independencia)  
   3.1 [Definição](#def-independencia)  
   3.2 [Critério Algébrico](#criterio)  
   3.3 [Interpretação Geométrica](#interpretacao)
4. [Subespaço Gerado e Dimensão](#subespaco)
5. [Posto de uma Matriz](#posto)  
   5.1 [Definição](#def-posto)  
   5.2 [Posto de Linhas e Colunas](#linhas-colunas)  
   5.3 [Posto e Soluções de Sistemas](#sistemas)
6. [Invertibilidade e Posto Máximo](#invertibilidade)
7. [Exemplos Resolvidos](#exemplos)
8. [Exercícios](#exercicios)
9. [Conexões com Outros Capítulos](#conexoes)
10. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

Independência linear e posto são o núcleo da álgebra linear.

Eles determinam:

- quantas direções distintas um conjunto de vetores possui  
- se uma matriz é invertível  
- quantas soluções um sistema linear tem  
- quantas dimensões um subespaço possui  
- quantos parâmetros podem ser estimados em modelos lineares  

Este notebook formaliza esses conceitos com rigor matemático.

---

<a id="combinacoes"></a>
# 2. 🧮 Combinações Lineares

Dado vetores $v_1,\dots,v_k \in \mathbb{R}^n$, uma **combinação linear** é:

$$
c_1 v_1 + \cdots + c_k v_k
$$

onde $c_1,\dots,c_k \in \mathbb{R}$.

---

<a id="independencia"></a>
# 3. 🧩 Independência Linear

<a id="def-independencia"></a>
## 3.1 Definição

Vetores $v_1,\dots,v_k$ são **linearmente dependentes** se existem escalares,
não todos zero, tais que:

$$
c_1 v_1 + \cdots + c_k v_k = 0
$$

Se a única solução é:

$$
c_1 = \cdots = c_k = 0
$$

então os vetores são **linearmente independentes**.

---

<a id="criterio"></a>
## 3.2 Critério Algébrico

Os vetores $v_1,\dots,v_k$ são independentes se e somente se a matriz:

$$
A = [v_1 \; v_2 \; \cdots \; v_k]
$$

tem **posto $k$**.

---

<a id="interpretacao"></a>
## 3.3 Interpretação Geométrica

- Em $\mathbb{R}^2$:  
  dois vetores são independentes se não são paralelos.  

- Em $\mathbb{R}^3$:  
  três vetores são independentes se não estão no mesmo plano.  

- Em geral:  
  independência significa que **nenhum vetor pode ser escrito como combinação dos outros**.

---

<a id="subespaco"></a>
# 4. 🧭 Subespaço Gerado e Dimensão

O **subespaço gerado** por $v_1,\dots,v_k$ é:

$$
\text{span}(v_1,\dots,v_k)
=
\{c_1 v_1 + \cdots + c_k v_k : c_i \in \mathbb{R}\}
$$

A **dimensão** é o número máximo de vetores linearmente independentes no conjunto.

---

<a id="posto"></a>
# 5. 🧱 Posto de uma Matriz

<a id="def-posto"></a>
## 5.1 Definição

O **posto** de uma matriz $A$ é:

> o número de linhas (ou colunas) linearmente independentes.

Equivalentemente:

$$
\text{rank}(A) = \dim(\text{Col}(A)) = \dim(\text{Row}(A))
$$

---

<a id="linhas-colunas"></a>
## 5.2 Posto de Linhas e Colunas

Teorema fundamental:

$$
\text{rank}(A) = \text{rank}(A^T)
$$

Ou seja:

👉 **posto das linhas = posto das colunas**.

---

<a id="sistemas"></a>
## 5.3 Posto e Soluções de Sistemas

Para o sistema $Ax = b$:

- **posto(A) = n** → solução única  
- **posto(A) < n** → infinitas soluções ou nenhuma  
- **posto(A) < posto([A|b])** → sistema inconsistente  

---

<a id="invertibilidade"></a>
# 6. 🔁 Invertibilidade e Posto Máximo

Uma matriz quadrada $A \in \mathbb{R}^{n \times n}$ é invertível se e somente se:

$$
\text{rank}(A) = n
\quad\Longleftrightarrow\quad
\det(A) \ne 0
$$

---

<a id="exemplos"></a>
# 7. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Dependência linear**

Os vetores:

$$
(1,2), \quad (2,4)
$$

são dependentes, pois:

$$
(2,4) = 2(1,2)
$$

---

### **Exemplo 2 — Posto**

Para:

$$
A =
\begin{pmatrix}
1 & 2 \\
2 & 4
\end{pmatrix}
$$

As colunas são múltiplas → posto = 1.

---

### **Exemplo 3 — Invertibilidade**

Para:

$$
A =
\begin{pmatrix}
1 & 3 \\
2 & 5
\end{pmatrix}
$$

$$
\det(A) = 1\cdot5 - 3\cdot2 = -1 \ne 0
$$

Logo, $A$ é invertível e $\text{rank}(A)=2$.

---

<a id="exercicios"></a>
# 8. 📝 Exercícios

1. Determine se os vetores $(1,0,1)$, $(2,1,3)$ e $(3,1,4)$ são independentes.  
2. Encontre o posto de  
   $$
   A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 1 \end{pmatrix}
   $$
3. Mostre que se $v_1,\dots,v_k$ são independentes, então qualquer subconjunto também é.  
4. Determine se o sistema $Ax=b$ tem solução para  
   $$
   A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix},
   \quad
   b = \begin{pmatrix} 3 \\ 6 \end{pmatrix}
   $$
5. Mostre que se $\text{rank}(A)=n$, então $Ax=b$ tem solução única.  

---

<a id="conexoes"></a>
# 9. 🔗 Conexões com Outros Capítulos

- [Matrizes e Vetores](ca://s?q=Criar_Notebook_10_Matrizes) — estrutura básica.  
- [Multiplicação de Matrizes](ca://s?q=Criar_Notebook_11_Multiplicacao) — combinações lineares.  
- [Determinante](ca://s?q=Criar_Notebook_13_Determinante) — invertibilidade.  
- [Autovalores e Autovetores](ca://s?q=Criar_Notebook_15_Autovalores) — decomposições espectrais.  

---

<a id="resumo"></a>
# 10. 🧾 Resumo Final

Este notebook desenvolveu:

- combinações lineares  
- independência linear  
- subespaços e dimensão  
- posto de uma matriz  
- invertibilidade  
- exemplos e exercícios  

Independência linear e posto são a espinha dorsal da álgebra linear.

---

[↑ Voltar ao Índice](#indice)
