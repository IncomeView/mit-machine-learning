<a id="indice"></a>

# 📘 NB11 — Multiplicação de Matrizes

Este notebook desenvolve, com rigor matemático, a operação de multiplicação de
matrizes, incluindo definição formal, interpretação geométrica, produtos interno
e externo, propriedades estruturais e aplicações fundamentais em álgebra linear.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Compatibilidade de Dimensões](#dimensoes)
3. [Definição da Multiplicação de Matrizes](#definicao)
4. [Produto Matriz–Vetor](#matriz-vetor)
5. [Produto Interno](#produto-interno)
6. [Produto Externo](#produto-externo)
7. [Propriedades da Multiplicação](#propriedades)
8. [Interpretação Geométrica](#interpretacao)
9. [Exemplos Resolvidos](#exemplos)
10. [Exercícios](#exercicios)
11. [Conexões com Outros Capítulos](#conexoes)
12. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

A multiplicação de matrizes é a operação central da álgebra linear.

Ela representa:

- composição de transformações lineares  
- mudança de coordenadas  
- projeções  
- rotações  
- sistemas lineares  
- operações fundamentais em ML e estatística  

Este notebook formaliza essa operação com rigor matemático.

---

<a id="dimensoes"></a>
# 2. 📏 Compatibilidade de Dimensões

Se:

- $A$ é $m \times n$  
- $B$ é $n \times k$  

então o produto:

$$
AB \in \mathbb{R}^{m \times k}
$$

A dimensão interna **$n$** deve coincidir.

---

<a id="definicao"></a>
# 3. 🧱 Definição da Multiplicação de Matrizes

O elemento $(i,j)$ do produto $AB$ é:

$$
(AB)_{i,j} = \sum_{r=1}^{n} A_{i,r} B_{r,j}
$$

Interpretação:

- linha $i$ de $A$  
- coluna $j$ de $B$  
- produto interno entre ambos  

---

<a id="matriz-vetor"></a>
# 4. 🔧 Produto Matriz–Vetor

Se $A$ é $m \times n$ e $v$ é $n \times 1$:

$$
Av = v_1 A_{:,1} + v_2 A_{:,2} + \cdots + v_n A_{:,n}
$$

Ou seja:

👉 **o produto matriz–vetor é uma combinação linear das colunas de $A$**.

---

<a id="produto-interno"></a>
# 5. 🎯 Produto Interno

Para vetores $u, v \in \mathbb{R}^n$:

$$
u^T v = \sum_{i=1}^n u_i v_i
$$

É um **escalar**.

Interpretação geométrica:

$$
u^T v = \|u\| \|v\| \cos(\theta)
$$

---

<a id="produto-externo"></a>
# 6. 🧩 Produto Externo

Para $u \in \mathbb{R}^m$ e $v \in \mathbb{R}^n$:

$$
uv^T =
\begin{pmatrix}
u_1 v_1 & \dots & u_1 v_n \\
\vdots & \ddots & \vdots \\
u_m v_1 & \dots & u_m v_n
\end{pmatrix}
$$

Propriedade fundamental:

👉 **$uv^T$ é sempre uma matriz de posto 1** (a menos que $u=0$ ou $v=0$).

---

<a id="propriedades"></a>
# 7. 📐 Propriedades da Multiplicação

### **1. Associatividade**

$$
A(BC) = (AB)C
$$

### **2. Distributividade**

$$
A(B + C) = AB + AC
$$

### **3. Não comutatividade**

Em geral:

$$
AB \ne BA
$$

### **4. Identidade**

$$
AI = IA = A
$$

### **5. Compatibilidade com escalares**

$$
(\alpha A)B = A(\alpha B) = \alpha(AB)
$$

---

<a id="interpretacao"></a>
# 8. 🧭 Interpretação Geométrica

A multiplicação de matrizes representa **composição de transformações lineares**.

Se:

- $A$ aplica uma transformação  
- $B$ aplica outra  

então:

$$
AB
$$

aplica **primeiro $B$, depois $A$**.

Exemplos:

- rotação seguida de escala  
- projeção seguida de rotação  
- cisalhamento seguido de reflexão  

---

<a id="exemplos"></a>
# 9. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Multiplicação simples**

$$
A =
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix},
\quad
B =
\begin{pmatrix}
2 & 0 \\
1 & 2
\end{pmatrix}
$$

Então:

$$
AB =
\begin{pmatrix}
1\cdot2 + 2\cdot1 & 1\cdot0 + 2\cdot2 \\
3\cdot2 + 4\cdot1 & 3\cdot0 + 4\cdot2
\end{pmatrix}
=
\begin{pmatrix}
4 & 4 \\
10 & 8
\end{pmatrix}
$$

---

### **Exemplo 2 — Produto externo**

Para:

$$
u = \begin{pmatrix} 1 \\ 2 \end{pmatrix},
\quad
v = \begin{pmatrix} 3 \\ 4 \\ 5 \end{pmatrix}
$$

$$
uv^T =
\begin{pmatrix}
3 & 4 & 5 \\
6 & 8 & 10
\end{pmatrix}
$$

---

### **Exemplo 3 — Produto interno como caso especial**

Se $u, v \in \mathbb{R}^n$:

$$
u^T v = \text{matriz } 1 \times 1
$$

---

<a id="exercicios"></a>
# 10. 📝 Exercícios

1. Calcule $AB$ para  
   $$
   A = \begin{pmatrix} 1 & 3 \\ 2 & 1 \end{pmatrix},
   \quad
   B = \begin{pmatrix} 0 & 2 \\ 4 & 1 \end{pmatrix}
   $$
2. Mostre que $uv^T$ tem posto 1.  
3. Mostre que $u^T v = v^T u$.  
4. Determine se $AB = BA$ para  
   $$
   A = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix},
   \quad
   B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
   $$
5. Mostre que $A(uv^T) = (Au)v^T$.  

---

<a id="conexoes"></a>
# 11. 🔗 Conexões com Outros Capítulos

- [Matrizes e Vetores](ca://s?q=Criar_Notebook_10_Matrizes) — base estrutural.  
- [Independência Linear e Posto](ca://s?q=Criar_Notebook_12_Posto) — produto externo e posto.  
- [Determinante](ca://s?q=Criar_Notebook_13_Determinante) — multiplicação e escala de volume.  
- [Autovalores e Autovetores](ca://s?q=Criar_Notebook_15_Autovalores) — composição de transformações.  

---

<a id="resumo"></a>
# 12. 🧾 Resumo Final

Este notebook desenvolveu:

- definição da multiplicação de matrizes  
- produto matriz–vetor  
- produto interno  
- produto externo  
- propriedades fundamentais  
- interpretação geométrica  
- exemplos e exercícios  

A multiplicação de matrizes é o núcleo da álgebra linear e da geometria de transformações.

---

[↑ Voltar ao Índice](#indice)
