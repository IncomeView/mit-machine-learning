<a id="indice"></a>

# 📘 NB7 — Gaussianas Univariadas

Este notebook desenvolve, com rigor matemático, a distribuição Gaussiana
(univariada), suas propriedades fundamentais, teoremas, transformações,
normalização, cálculo de momentos e aplicações matemáticas.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Definição da Gaussiana](#definicao)
3. [Propriedades Fundamentais](#propriedades)  
   3.1 [Simetria](#simetria)  
   3.2 [Esperança](#esperanca)  
   3.3 [Variância](#variancia)
4. [Normalização da Gaussiana](#normalizacao)
5. [Padronização (Z-score)](#padronizacao)
6. [Regra 68–95–99.7](#regra)
7. [Momento Gerador e Momentos](#momentos)
8. [MLE para $\mu$ e $\sigma^2$](#mle)
9. [Exemplos Resolvidos](#exemplos)
10. [Exercícios](#exercicios)
11. [Conexões com Outros Capítulos](#conexoes)
12. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

A distribuição Gaussiana é a mais importante da matemática aplicada.

Ela aparece em:

- teoria do erro  
- estatística inferencial  
- modelos probabilísticos  
- análise de ruído  
- otimização  
- teoria da informação  
- processos estocásticos  

E é a base de:

- PCA  
- regressão linear  
- modelos lineares gerais  
- filtros de Kalman  

Este notebook desenvolve a Gaussiana com rigor matemático.

---

<a id="definicao"></a>
# 2. 📈 Definição da Gaussiana

A distribuição Gaussiana univariada com média $\mu$ e variância $\sigma^2$ é:

$$
p(x \mid \mu, \sigma^2)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

Parâmetros:

- $\mu$ controla a **posição**  
- $\sigma^2$ controla a **dispersão**  

---

<a id="propriedades"></a>
# 3. 🧩 Propriedades Fundamentais

<a id="simetria"></a>
## 3.1 Simetria

A Gaussiana é simétrica em torno de $\mu$:

$$
p(\mu + t) = p(\mu - t)
$$

---

<a id="esperanca"></a>
## 3.2 Esperança

$$
\mathbb{E}[X] = \mu
$$

---

<a id="variancia"></a>
## 3.3 Variância

$$
\mathrm{Var}(X) = \sigma^2
$$

---

<a id="normalizacao"></a>
# 4. 🧮 Normalização da Gaussiana

A integral da Gaussiana é 1:

$$
\int_{-\infty}^{\infty}
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
dx
= 1
$$

A prova usa:

- substituição $z = \frac{x-\mu}{\sigma}$  
- integral de Gauss  
- fatorização da integral dupla  

---

<a id="padronizacao"></a>
# 5. 🔄 Padronização (Z-score)

Se:

$$
Z = \frac{X - \mu}{\sigma}
$$

então:

$$
Z \sim \mathcal{N}(0,1)
$$

A PDF padrão é:

$$
\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}
$$

---

<a id="regra"></a>
# 6. 📊 Regra 68–95–99.7

Para $X \sim \mathcal{N}(\mu,\sigma^2)$:

- 68% em $[\mu - \sigma, \mu + \sigma]$  
- 95% em $[\mu - 2\sigma, \mu + 2\sigma]$  
- 99.7% em $[\mu - 3\sigma, \mu + 3\sigma]$  

---

<a id="momentos"></a>
# 7. 🧠 Momento Gerador e Momentos

O momento gerador é:

$$
M_X(t) = \exp\left( \mu t + \frac{1}{2}\sigma^2 t^2 \right)
$$

Momentos:

$$
\mathbb{E}[X^k] = \frac{d^k}{dt^k} M_X(t)\bigg|_{t=0}
$$

---

<a id="mle"></a>
# 8. 📐 MLE para $\mu$ e $\sigma^2$

Dado um conjunto de observações $x_1,\dots,x_n$:

### **Estimador de máxima verossimilhança da média**

$$
\hat{\mu} = \frac{1}{n} \sum_{i=1}^n x_i
$$

### **Estimador da variância**

$$
\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2
$$

---

<a id="exemplos"></a>
# 9. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Padronização**

Se $X \sim \mathcal{N}(10,4)$, encontre $P(X > 12)$.

Padronizando:

$$
Z = \frac{12 - 10}{2} = 1
$$

Logo:

$$
P(X > 12) = P(Z > 1)
$$

---

### **Exemplo 2 — Esperança**

Para $X \sim \mathcal{N}(3,9)$:

$$
\mathbb{E}[X] = 3
$$

---

### **Exemplo 3 — Variância**

$$
\mathrm{Var}(X) = 9
$$

---

<a id="exercicios"></a>
# 10. 📝 Exercícios

1. Mostre que a Gaussiana é simétrica em torno de $\mu$.  
2. Prove que $\mathbb{E}[X] = \mu$.  
3. Prove que $\mathrm{Var}(X) = \sigma^2$.  
4. Padronize $X \sim \mathcal{N}(5,1)$ e calcule $P(X > 6)$.  
5. Derive o MLE de $\mu$ a partir da log-likelihood.  

---

<a id="conexoes"></a>
# 11. 🔗 Conexões com Outros Capítulos

- [Probabilidade e PDFs](ca://s?q=Criar_Notebook_6_Probabilidade) — base teórica.  
- [Otimização](ca://s?q=Criar_Notebook_8_Otimizacao_1D) — MLE usa derivadas.  
- [Matrizes e Vetores](ca://s?q=Criar_Notebook_10_Matrizes) — prepara para Gaussianas multivariadas.  
- [Autovalores](ca://s?q=Criar_Notebook_15_Autovalores) — aparece em PCA.  

---

<a id="resumo"></a>
# 12. 🧾 Resumo Final

Este notebook desenvolveu:

- definição da Gaussiana  
- propriedades fundamentais  
- normalização  
- padronização  
- regra 68–95–99.7  
- momento gerador  
- MLE  
- exemplos e exercícios  

A Gaussiana é a distribuição contínua mais importante da matemática aplicada.

---

[↑ Voltar ao Índice](#indice)
v