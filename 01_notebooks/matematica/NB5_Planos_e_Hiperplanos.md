<a id="indice"></a>

# 📘 NB5 — Planos e Hiperplanos

Este notebook desenvolve, com rigor matemático, os conceitos de planos e
hiperplanos, vetores normais, distâncias, projeções e interpretações geométricas.
É um capítulo essencial para geometria analítica, álgebra linear e otimização.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Planos em $\mathbb{R}^2$](#planos2d)
3. [Planos em $\mathbb{R}^3$](#planos3d)
4. [Hiperplanos em $\mathbb{R}^n$](#hiperplanos)  
   4.1 [Definição](#def-hiperplano)  
   4.2 [Vetor Normal](#vetor-normal)  
   4.3 [Equação Geral](#equacao-geral)
5. [Distância a um Hiperplano](#distancia)
6. [Projeção Ortogonal](#projecao)
7. [Interpretação Geométrica](#interpretacao)
8. [Exemplos Resolvidos](#exemplos)
9. [Exercícios](#exercicios)
10. [Conexões com Outros Capítulos](#conexoes)
11. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

Planos e hiperplanos são objetos fundamentais em:

- geometria analítica  
- álgebra linear  
- otimização convexa  
- análise de dados  
- decomposições matriciais  
- fronteiras de decisão  

Eles representam:

- superfícies lineares,  
- separações entre regiões,  
- restrições lineares,  
- subespaços afins.

Este notebook estabelece a base geométrica para temas como PCA, SVM, regressão
linear e decomposições espectrais.

---

<a id="planos2d"></a>
# 2. 📐 Planos em $\mathbb{R}^2$

Em $\mathbb{R}^2$, um “plano” é uma **reta**.

A forma geral é:

$$
ax + by + c = 0
$$

onde $(a,b)$ é o **vetor normal**.

---

<a id="planos3d"></a>
# 3. 📐 Planos em $\mathbb{R}^3$

Em $\mathbb{R}^3$, um plano é descrito por:

$$
ax + by + cz + d = 0
$$

O vetor normal é:

$$
n = (a,b,c)
$$

---

<a id="hiperplanos"></a>
# 4. 🧱 Hiperplanos em $\mathbb{R}^n$

<a id="def-hiperplano"></a>
## 4.1 Definição

Um hiperplano é o conjunto de pontos $x \in \mathbb{R}^n$ que satisfazem:

$$
\theta \cdot x + \theta_0 = 0
$$

onde $\theta \in \mathbb{R}^n$ é o vetor normal.

---

<a id="vetor-normal"></a>
## 4.2 Vetor Normal

O vetor $\theta$ é perpendicular ao hiperplano.

Propriedade fundamental:

$$
\theta \cdot (x_2 - x_1) = 0
$$

para quaisquer pontos $x_1, x_2$ no hiperplano.

---

<a id="equacao-geral"></a>
## 4.3 Equação Geral

A equação:

$$
\theta \cdot x + \theta_0 = 0
$$

define um hiperplano **afim**.

Se $\theta_0 = 0$, o hiperplano passa pela origem e é um **subespaço linear**.

---

<a id="distancia"></a>
# 5. 📏 Distância a um Hiperplano

A distância assinada de um ponto $x$ ao hiperplano é:

$$
d(x) = \frac{\theta \cdot x + \theta_0}{\|\theta\|}
$$

A distância absoluta é:

$$
|d(x)|
$$

---

<a id="projecao"></a>
# 6. 📐 Projeção Ortogonal

A projeção de um vetor $v$ no hiperplano é:

$$
v_{\text{proj}}
=
v - \frac{\theta \cdot v + \theta_0}{\|\theta\|^2}\,\theta
$$

Essa fórmula remove a componente de $v$ na direção normal.

---

<a id="interpretacao"></a>
# 7. 🧭 Interpretação Geométrica

- O hiperplano divide o espaço em dois lados:  
  $$
  \theta \cdot x + \theta_0 > 0
  $$
  e  
  $$
  \theta \cdot x + \theta_0 < 0
  $$

- A distância mede o quanto um ponto está “acima” ou “abaixo” do hiperplano.  
- A projeção encontra o ponto mais próximo no hiperplano.  
- O vetor normal determina a orientação do hiperplano.  

---

<a id="exemplos"></a>
# 8. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Distância**

Considere o hiperplano:

$$
3x + 4y - 5 = 0
$$

e o ponto $p = (1,2)$.

A distância é:

$$
d(p) = \frac{3(1) + 4(2) - 5}{\sqrt{3^2 + 4^2}}
= \frac{6}{5}
$$

---

### **Exemplo 2 — Projeção**

Projete $v = (2,1)$ no hiperplano $x + y - 1 = 0$.

Aqui:

- $\theta = (1,1)$  
- $\theta_0 = -1$

$$
v_{\text{proj}}
=
v - \frac{(1,1)\cdot(2,1) - 1}{2}(1,1)
=
(2,1) - \frac{2}{2}(1,1)
=
(1,0)
$$

---

<a id="exercicios"></a>
# 9. 📝 Exercícios

1. Encontre a distância do ponto $(3,-1)$ ao hiperplano $2x - y + 4 = 0$.  
2. Mostre que $\theta$ é perpendicular ao hiperplano.  
3. Projete $(4,2)$ no hiperplano $x - 2y + 1 = 0$.  
4. Mostre que o conjunto $\{x : \theta \cdot x = 0\}$ é um subespaço.  
5. Determine a equação do hiperplano perpendicular a $(1,2,3)$ que passa por $(1,1,1)$.  

---

<a id="conexoes"></a>
# 10. 🔗 Conexões com Outros Capítulos

- [Vetores](ca://s?q=Criar_Notebook_4_Pontos_e_Vetores) — hiperplanos dependem de vetores normais.  
- [Funções Lineares](ca://s?q=Criar_Notebook_3_Propriedades_de_Funcoes) — hiperplanos são níveis de funções lineares.  
- [Independência Linear e Posto](ca://s?q=Criar_Notebook_12_Posto) — hiperplanos são subespaços de codimensão 1.  
- [Autovalores e Autovetores](ca://s?q=Criar_Notebook_15_Autovalores) — direções normais aparecem em decomposições espectrais.  

---

<a id="resumo"></a>
# 11. 🧾 Resumo Final

Este notebook desenvolveu:

- planos e hiperplanos  
- vetores normais  
- equações gerais  
- distância  
- projeção ortogonal  
- interpretação geométrica  
- exemplos e exercícios  

Esses conceitos são fundamentais para geometria analítica, álgebra linear e otimização.

---

[↑ Voltar ao Índice](#indice)
