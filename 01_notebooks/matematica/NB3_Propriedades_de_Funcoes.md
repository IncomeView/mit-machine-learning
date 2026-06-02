<a id="indice"></a>

# 📘 NB3 — Propriedades de Funções

Este notebook desenvolve, com rigor matemático, os fundamentos de funções,
monotonicidade, convexidade, funções lineares e hiperplanos.  
É um capítulo essencial para análise, otimização e geometria.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Funções como Mapeamentos](#mapeamentos)
3. [Monotonicidade](#monotonicidade)  
   3.1 [Definição](#def-monotonicidade)  
   3.2 [Teoremas](#teoremas-monotonicidade)  
   3.3 [Exemplos](#exemplos-monotonicidade)  
   3.4 [Exercícios](#exercicios-monotonicidade)
4. [Convexidade](#convexidade)  
   4.1 [Definição](#def-convexidade)  
   4.2 [Teorema da Reta Secante](#reta-secante)  
   4.3 [Caracterização via Derivadas](#derivadas-convexidade)  
   4.4 [Exemplos](#exemplos-convexidade)  
   4.5 [Exercícios](#exercicios-convexidade)
5. [Funções Lineares](#funcoes-lineares)  
   5.1 [Definição](#def-funcao-linear)  
   5.2 [Geometria](#geometria-linear)  
   5.3 [Exemplos](#exemplos-lineares)
6. [Hiperplanos](#hiperplanos)  
   6.1 [Definição](#def-hiperplano)  
   6.2 [Distância ao Hiperplano](#distancia)  
   6.3 [Projeção Ortogonal](#projecao)  
   6.4 [Exercícios](#exercicios-hiperplano)
7. [Conexões com Outros Capítulos](#conexoes)
8. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

Funções são o coração da matemática aplicada.  
Elas descrevem:

- relações entre variáveis,  
- superfícies,  
- curvas,  
- fronteiras de decisão,  
- funções de custo,  
- modelos estatísticos.

Neste notebook, estudamos propriedades fundamentais que aparecem em:

- otimização,  
- geometria,  
- análise,  
- estatística,  
- machine learning.

---

<a id="mapeamentos"></a>
# 2. 🧭 Funções como Mapeamentos

Uma função é uma regra que associa cada entrada a uma saída:

$$
f : \mathbb{R}^n \to \mathbb{R}
$$

Exemplos:

- funções de custo  
- modelos lineares  
- ativações  
- hiperplanos  

---

<a id="monotonicidade"></a>
# 3. 📈 Monotonicidade

<a id="def-monotonicidade"></a>
## 3.1 Definição

Uma função é **monótona crescente** se:

$$
x_1 < x_2 \Rightarrow f(x_1) \le f(x_2)
$$

É **monótona decrescente** se:

$$
x_1 < x_2 \Rightarrow f(x_1) \ge f(x_2)
$$

---

<a id="teoremas-monotonicidade"></a>
## 3.2 Teoremas Fundamentais

### **Teorema 1 — Derivada positiva implica monotonicidade**

Se $f$ é diferenciável e:

$$
f'(x) \ge 0 \quad \forall x
$$

então $f$ é crescente.

---

### **Teorema 2 — Derivada negativa implica monotonicidade**

Se:

$$
f'(x) \le 0
$$

então $f$ é decrescente.

---

<a id="exemplos-monotonicidade"></a>
## 3.3 Exemplos Resolvidos

### **Exemplo 1**

$f(x) = 3x + 1$

$$
f'(x) = 3 > 0
$$

Logo, $f$ é crescente.

---

### **Exemplo 2**

$f(x) = -x^2$

$$
f'(x) = -2x
$$

Não tem sinal fixo → não é monótona.

---

<a id="exercicios-monotonicidade"></a>
## 3.4 Exercícios

1. Determine se $f(x) = x^3$ é crescente.  
2. Mostre que $f(x) = e^x$ é estritamente crescente.  
3. Determine onde $f(x) = x^3 - 3x$ é crescente e decrescente.  

---

<a id="convexidade"></a>
# 4. 📐 Convexidade

<a id="def-convexidade"></a>
## 4.1 Definição

Uma função $f$ é convexa se:

$$
f(\lambda x + (1-\lambda)y)
\le
\lambda f(x) + (1-\lambda)f(y)
$$

para todo $0 \le \lambda \le 1$.

---

<a id="reta-secante"></a>
## 4.2 Teorema da Reta Secante

$f$ é convexa **se e somente se** o gráfico está sempre abaixo da reta secante.

---

<a id="derivadas-convexidade"></a>
## 4.3 Caracterização via Derivadas

### **Teorema 1 — Segunda derivada**

Se $f$ é duas vezes diferenciável, então:

$$
f \text{ é convexa } \iff f''(x) \ge 0
$$

---

### **Teorema 2 — Gradiente Monótono**

$f$ é convexa $\iff$ seu gradiente é crescente:

$$
(f'(x_2) - f'(x_1))(x_2 - x_1) \ge 0
$$

---

<a id="exemplos-convexidade"></a>
## 4.4 Exemplos Resolvidos

### **Exemplo 1**

$f(x) = x^2$

$$
f''(x) = 2 > 0
$$

Logo, é convexa.

---

### **Exemplo 2**

$f(x) = e^x$

$$
f''(x) = e^x > 0
$$

Convexa.

---

<a id="exercicios-convexidade"></a>
## 4.5 Exercícios

1. Mostre que $f(x) = |x|$ é convexa.  
2. Determine se $f(x) = x^3$ é convexa.  
3. Mostre que $f(x) = \ln x$ é côncava.  

---

<a id="funcoes-lineares"></a>
# 5. 📏 Funções Lineares

<a id="def-funcao-linear"></a>
## 5.1 Definição

Uma função linear em $\mathbb{R}^n$ é:

$$
f(x) = \theta \cdot x + \theta_0
$$

---

<a id="geometria-linear"></a>
## 5.2 Geometria

- $\theta$ é o vetor normal  
- $\theta_0$ desloca o hiperplano  
- nível zero define um hiperplano  

---

<a id="exemplos-lineares"></a>
## 5.3 Exemplos

1. $f(x) = 2x + 1$  
2. $f(x_1,x_2) = 3x_1 - x_2 + 4$  

---

<a id="hiperplanos"></a>
# 6. 🧱 Hiperplanos

<a id="def-hiperplano"></a>
## 6.1 Definição

Um hiperplano é o conjunto:

$$
\theta \cdot x + \theta_0 = 0
$$

---

<a id="distancia"></a>
## 6.2 Distância ao Hiperplano

$$
d(x) = \frac{\theta \cdot x + \theta_0}{\|\theta\|}
$$

---

<a id="projecao"></a>
## 6.3 Projeção Ortogonal

$$
v_{\text{proj}}
=
v - \frac{\theta \cdot v + \theta_0}{\|\theta\|^2}\,\theta
$$

---

<a id="exercicios-hiperplano"></a>
## 6.4 Exercícios

1. Encontre a distância do ponto $(1,2)$ ao hiperplano $3x + 4y - 5 = 0$.  
2. Prove que $\theta$ é perpendicular ao hiperplano.  
3. Calcule a projeção de $v=(2,1)$ no hiperplano $x+y-1=0$.  

---

<a id="conexoes"></a>
# 7. 🔗 Conexões com Outros Capítulos

- NB4: vetores e produtos internos são usados aqui.  
- NB5: hiperplanos são aprofundados geometricamente.  
- NB8–NB9: convexidade é essencial para otimização.  
- NB12–NB15: funções lineares aparecem em álgebra linear.  

---

<a id="resumo"></a>
# 8. 🧾 Resumo Final

Este notebook desenvolveu:

- funções como mapeamentos  
- monotonicidade  
- convexidade  
- funções lineares  
- hiperplanos  
- distância e projeção  

Esses conceitos são fundamentais para análise, geometria e otimização.

---

[↑ Voltar ao Índice](#indice)
