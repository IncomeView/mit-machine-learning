<a id="indice"></a>

# 📘 NB9 — Gradientes e Otimização Multivariada

Este notebook desenvolve, com rigor matemático, os fundamentos da otimização
em múltiplas variáveis: gradientes, direção de maior crescimento, Hessiana,
condições de otimalidade e métodos de descida.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Funções Multivariadas](#funcoes)
3. [Gradiente](#gradiente)  
   3.1 [Definição](#def-gradiente)  
   3.2 [Interpretação Geométrica](#interpretacao-gradiente)  
   3.3 [Direção de Maior Crescimento](#direcao)
4. [Derivadas Direcionais](#direcionais)
5. [Hessiana](#hessiana)  
   5.1 [Definição](#def-hessiana)  
   5.2 [Teste da Hessiana](#teste-hessiana)
6. [Condições de Otimalidade](#condicoes)
7. [Gradiente Descendente](#gd)
8. [Convexidade em $\mathbb{R}^n$](#convexidade)
9. [Exemplos Resolvidos](#exemplos)
10. [Exercícios](#exercicios)
11. [Conexões com Outros Capítulos](#conexoes)
12. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

Otimização multivariada é a base de:

- regressão  
- PCA  
- SVM  
- redes neurais  
- MLE  
- métodos numéricos  

Antes de algoritmos avançados, precisamos dominar:

- gradientes  
- Hessianas  
- condições de otimalidade  
- convexidade multivariada  

---

<a id="funcoes"></a>
# 2. 🧭 Funções Multivariadas

Uma função multivariada é:

$$
f : \mathbb{R}^n \to \mathbb{R}
$$

Exemplos:

- funções de custo  
- superfícies quadráticas  
- modelos lineares  
- log-likelihoods  

---

<a id="gradiente"></a>
# 3. 📈 Gradiente

<a id="def-gradiente"></a>
## 3.1 Definição

O gradiente de $f$ é o vetor:

$$
\nabla f(x) =
\begin{pmatrix}
\frac{\partial f}{\partial x_1} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{pmatrix}
$$

---

<a id="interpretacao-gradiente"></a>
## 3.2 Interpretação Geométrica

O gradiente aponta para a direção de **maior crescimento** da função.

---

<a id="direcao"></a>
## 3.3 Direção de Maior Crescimento

A derivada direcional na direção unitária $u$ é:

$$
D_u f(x) = \nabla f(x) \cdot u
$$

O máximo ocorre quando $u$ é paralelo ao gradiente.

---

<a id="direcionais"></a>
# 4. 🧭 Derivadas Direcionais

A derivada direcional mede a taxa de variação de $f$ na direção $u$:

$$
D_u f(x) = \lim_{t\to 0} \frac{f(x + tu) - f(x)}{t}
$$

Se $u$ é unitário:

$$
D_u f(x) = \nabla f(x) \cdot u
$$

---

<a id="hessiana"></a>
# 5. 🧮 Hessiana

<a id="def-hessiana"></a>
## 5.1 Definição

A Hessiana é a matriz das segundas derivadas:

$$
H_f(x) =
\begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}
$$

---

<a id="teste-hessiana"></a>
## 5.2 Teste da Hessiana

Se $x^*$ é ponto crítico ($\nabla f(x^*) = 0$):

- **mínimo local** se $H_f(x^*)$ é **definida positiva**  
- **máximo local** se é **definida negativa**  
- **ponto de sela** se é **indefinida**  

---

<a id="condicoes"></a>
# 6. 🧩 Condições de Otimalidade

### **Condição necessária**

Se $x^*$ é mínimo local interior:

$$
\nabla f(x^*) = 0
$$

### **Condição suficiente**

Se:

- $\nabla f(x^*) = 0$  
- $H_f(x^*)$ é definida positiva  

então $x^*$ é mínimo local.

---

<a id="gd"></a>
# 7. 🔽 Gradiente Descendente

O método de gradiente descendente atualiza:

$$
x_{k+1} = x_k - \alpha \nabla f(x_k)
$$

onde $\alpha > 0$ é a taxa de aprendizado.

---

<a id="convexidade"></a>
# 8. 📐 Convexidade em $\mathbb{R}^n$

$f$ é convexa se:

$$
f(\lambda x + (1-\lambda)y)
\le
\lambda f(x) + (1-\lambda)f(y)
$$

Se $H_f(x)$ é definida positiva para todo $x$, então $f$ é convexa.

---

<a id="exemplos"></a>
# 9. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Gradiente**

Para:

$$
f(x,y) = x^2 + y^2
$$

$$
\nabla f = (2x, 2y)
$$

---

### **Exemplo 2 — Hessiana**

$$
H_f =
\begin{pmatrix}
2 & 0 \\
0 & 2
\end{pmatrix}
$$

Definida positiva → mínimo global em $(0,0)$.

---

### **Exemplo 3 — Gradiente Descendente**

Para $f(x) = x^2$:

$$
x_{k+1} = x_k - 2\alpha x_k = (1 - 2\alpha)x_k
$$

Converge se $0 < \alpha < 1$.

---

<a id="exercicios"></a>
# 10. 📝 Exercícios

1. Calcule o gradiente de $f(x,y) = 3x^2 + xy + y^2$.  
2. Encontre a Hessiana de $f(x,y,z) = x^2 + y^2 + z^2 - xy$.  
3. Determine se $f(x,y) = x^2 - y^2$ tem mínimo, máximo ou sela.  
4. Aplique uma iteração de gradiente descendente em $f(x) = x^2$ com $\alpha = 0.1$ e $x_0 = 5$.  
5. Mostre que $f(x) = \|Ax - b\|^2$ é convexa.  

---

<a id="conexoes"></a>
# 11. 🔗 Conexões com Outros Capítulos

- [Convexidade](ca://s?q=Criar_Notebook_3_Propriedades_de_Funcoes) — base para otimalidade.  
- [Otimização 1D](ca://s?q=Criar_Notebook_8_Otimizacao_1D) — caso unidimensional.  
- [Matrizes e Vetores](ca://s?q=Criar_Notebook_10_Matrizes) — Hessiana é uma matriz.  
- [Autovalores](ca://s?q=Criar_Notebook_15_Autovalores) — definidade depende de autovalores.  

---

<a id="resumo"></a>
# 12. 🧾 Resumo Final

Este notebook desenvolveu:

- gradientes  
- derivadas direcionais  
- Hessiana  
- condições de otimalidade  
- gradiente descendente  
- convexidade multivariada  
- exemplos e exercícios  

Este capítulo é o coração da otimização moderna.

---

[↑ Voltar ao Índice](#indice)
