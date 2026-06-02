<a id="indice"></a>

# 📘 NB6 — Probabilidade e PDFs

Este notebook desenvolve, com rigor matemático, os fundamentos de probabilidade
contínua, funções densidade de probabilidade (PDFs), propriedades essenciais,
cálculo de probabilidades via integrais e interpretação geométrica.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Variáveis Aleatórias Contínuas](#variaveis)
3. [Função Densidade de Probabilidade (PDF)](#pdf)  
   3.1 [Definição Formal](#def-pdf)  
   3.2 [Propriedades Fundamentais](#propriedades-pdf)  
   3.3 [Probabilidade como Área](#area)
4. [Função Distribuição Acumulada (CDF)](#cdf)
5. [Esperança e Variância](#esperanca)  
   5.1 [Definições](#def-esperanca)  
   5.2 [Propriedades](#prop-esperanca)
6. [Transformações de Variáveis](#transformacoes)
7. [Exemplos Resolvidos](#exemplos)
8. [Exercícios](#exercicios)
9. [Conexões com Outros Capítulos](#conexoes)
10. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

Probabilidade contínua é a base de:

- estatística inferencial  
- modelos probabilísticos  
- Gaussianas  
- MLE  
- análise de incerteza  
- teoria da informação  

A interpretação geométrica (probabilidade = área) é essencial para:

- integrais  
- normalização  
- densidades conjuntas  
- mudança de variáveis  
- distribuições multivariadas  

---

<a id="variaveis"></a>
# 2. 🎲 Variáveis Aleatórias Contínuas

Uma variável aleatória contínua $X$ é uma função que associa resultados de um
experimento a números reais.

Ela é descrita por uma **função densidade de probabilidade** (PDF).

---

<a id="pdf"></a>
# 3. 📈 Função Densidade de Probabilidade (PDF)

<a id="def-pdf"></a>
## 3.1 Definição Formal

Uma função $f_X(x)$ é uma PDF se:

1. **Não-negatividade**  
   $$
   f_X(x) \ge 0
   $$

2. **Área total igual a 1**  
   $$
   \int_{-\infty}^{\infty} f_X(x)\,dx = 1
   $$

---

<a id="propriedades-pdf"></a>
## 3.2 Propriedades Fundamentais

- A PDF **não** precisa estar entre 0 e 1.  
- A probabilidade é dada pela **área**, não pelo valor da PDF.  
- A PDF pode ser maior que 1, desde que a área total seja 1.

Exemplo clássico:

$$
f(x) = 10 \quad \text{em } [0,0.1]
$$

A área é:

$$
\int_0^{0.1} 10\,dx = 1
$$

---

<a id="area"></a>
## 3.3 Probabilidade como Área

Para qualquer intervalo $[a,b]$:

$$
P(a \le X \le b) = \int_a^b f_X(x)\,dx
$$

---

<a id="cdf"></a>
# 4. 📉 Função Distribuição Acumulada (CDF)

A CDF é definida como:

$$
F_X(x) = \int_{-\infty}^x f_X(t)\,dt
$$

Propriedades:

- $F_X$ é crescente  
- $\lim_{x\to -\infty} F_X(x) = 0$  
- $\lim_{x\to \infty} F_X(x) = 1$  
- $F_X'(x) = f_X(x)$ quando $f$ é contínua  

---

<a id="esperanca"></a>
# 5. 📊 Esperança e Variância

<a id="def-esperanca"></a>
## 5.1 Definições

### **Esperança**

$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x)\,dx
$$

### **Variância**

$$
\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]
$$

---

<a id="prop-esperanca"></a>
## 5.2 Propriedades

### Linearidade da esperança

$$
\mathbb{E}[aX + b] = a\mathbb{E}[X] + b
$$

### Variância expandida

$$
\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

---

<a id="transformacoes"></a>
# 6. 🔄 Transformações de Variáveis

Se $Y = g(X)$ e $g$ é monotônica, então:

$$
f_Y(y) = f_X(g^{-1}(y)) \left| \frac{d}{dy} g^{-1}(y) \right|
$$

Este é o fundamento para:

- mudança de variáveis  
- normalização  
- distribuições derivadas  

---

<a id="exemplos"></a>
# 7. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Probabilidade em intervalo**

Se:

$$
f(x) = 2x \quad \text{em } [0,1]
$$

Calcule $P(0.2 \le X \le 0.5)$:

$$
\int_{0.2}^{0.5} 2x\,dx = [x^2]_{0.2}^{0.5} = 0.25 - 0.04 = 0.21
$$

---

### **Exemplo 2 — Esperança**

Para a mesma PDF:

$$
\mathbb{E}[X] = \int_0^1 x(2x)\,dx = 2\int_0^1 x^2\,dx = \frac{2}{3}
$$

---

### **Exemplo 3 — Variância**

$$
\mathbb{E}[X^2] = \int_0^1 x^2(2x)\,dx = 2\int_0^1 x^3\,dx = \frac{1}{2}
$$

$$
\mathrm{Var}(X) = \frac{1}{2} - \left(\frac{2}{3}\right)^2 = \frac{1}{18}
$$

---

<a id="exercicios"></a>
# 8. 📝 Exercícios

1. Mostre que $f(x) = 3x^2$ em $[0,1]$ é uma PDF.  
2. Calcule $P(0.3 \le X \le 0.7)$ para essa PDF.  
3. Encontre $\mathbb{E}[X]$ e $\mathrm{Var}(X)$.  
4. Para $f(x) = 1$ em $[0,1]$, calcule a CDF.  
5. Use mudança de variáveis para encontrar a PDF de $Y = X^2$.  

---

<a id="conexoes"></a>
# 9. 🔗 Conexões com Outros Capítulos

- [Gaussianas Univariadas](ca://s?q=Criar_Notebook_7_Gaussianas) — PDFs específicas.  
- [Otimização](ca://s?q=Criar_Notebook_8_Otimizacao_1D) — integrais e convexidade.  
- [MLE](ca://s?q=Criar_Notebook_7_Gaussianas) — PDFs como likelihoods.  
- [Álgebra Linear](ca://s?q=Criar_Notebook_10_Matrizes) — distribuições multivariadas.  

---

<a id="resumo"></a>
# 10. 🧾 Resumo Final

Este notebook desenvolveu:

- variáveis aleatórias contínuas  
- PDFs e CDFs  
- probabilidade como área  
- esperança e variância  
- mudança de variáveis  
- exemplos e exercícios  

Esses conceitos são fundamentais para estatística, Gaussianas e modelos probabilísticos.

---

[↑ Voltar ao Índice](#indice)
