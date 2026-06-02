<a id="indice"></a>

# 📘 NB8 — Otimização 1D via Cálculo

Este notebook desenvolve, com rigor matemático, os fundamentos da otimização
em uma variável real: pontos críticos, derivadas, testes de segunda derivada,
otimização em intervalos fechados e condições necessárias e suficientes.

---

# 📑 Índice

1. [Motivação Matemática](#motivacao)
2. [Pontos Críticos](#pontos-criticos)  
   2.1 [Definição](#def-pontos-criticos)  
   2.2 [Como Encontrar Pontos Críticos](#como-encontrar)
3. [Teste da Segunda Derivada](#segunda-derivada)
4. [Otimização em Intervalos Fechados](#intervalos)
5. [Condições Necessárias e Suficientes](#condicoes)
6. [Convexidade e Unicidade do Mínimo](#convexidade)
7. [Exemplos Resolvidos](#exemplos)
8. [Exercícios](#exercicios)
9. [Conexões com Outros Capítulos](#conexoes)
10. [Resumo Final](#resumo)

---

<a id="motivacao"></a>
# 1. 🎯 Motivação Matemática

Otimização é o coração de:

- cálculo  
- análise  
- estatística  
- métodos numéricos  
- machine learning  

Antes de otimizar funções em $\mathbb{R}^n$, precisamos dominar o caso 1D.

Este notebook desenvolve:

- como encontrar máximos e mínimos,  
- como usar derivadas,  
- como analisar curvatura,  
- como garantir unicidade do mínimo.

---

<a id="pontos-criticos"></a>
# 2. 📍 Pontos Críticos

<a id="def-pontos-criticos"></a>
## 2.1 Definição

Um ponto $x^*$ é **crítico** se:

$$
f'(x^*) = 0
$$

ou se a derivada **não existe**.

---

<a id="como-encontrar"></a>
## 2.2 Como Encontrar Pontos Críticos

1. Calcular $f'(x)$.  
2. Resolver $f'(x) = 0$.  
3. Incluir pontos onde $f'$ não existe.  

---

<a id="segunda-derivada"></a>
# 3. 📐 Teste da Segunda Derivada

Se $f$ é duas vezes diferenciável:

- **Mínimo local** se  
  $$
  f''(x^*) > 0
  $$

- **Máximo local** se  
  $$
  f''(x^*) < 0
  $$

- **Inconclusivo** se  
  $$
  f''(x^*) = 0
  $$

---

<a id="intervalos"></a>
# 4. 📏 Otimização em Intervalos Fechados

Para $f : [a,b] \to \mathbb{R}$:

1. Encontrar pontos críticos em $(a,b)$.  
2. Avaliar $f$ nos pontos críticos.  
3. Avaliar $f(a)$ e $f(b)$.  
4. Comparar todos os valores.  

---

<a id="condicoes"></a>
# 5. 🧩 Condições Necessárias e Suficientes

### **Condição necessária para ótimo interior**

Se $x^*$ é mínimo local interior:

$$
f'(x^*) = 0
$$

### **Condição suficiente**

Se:

$$
f'(x^*) = 0 \quad \text{e} \quad f''(x^*) > 0
$$

então $x^*$ é mínimo local.

---

<a id="convexidade"></a>
# 6. 📐 Convexidade e Unicidade do Mínimo

Se $f$ é **convexa** e diferenciável:

- qualquer ponto crítico é **mínimo global**  
- se $f$ é **estritamente convexa**, o mínimo é **único**

Caracterização via segunda derivada:

$$
f''(x) \ge 0 \Rightarrow f \text{ convexa}
$$

---

<a id="exemplos"></a>
# 7. 🧮 Exemplos Resolvidos

### **Exemplo 1 — Mínimo local**

Considere:

$$
f(x) = x^2 - 4x + 5
$$

1. Derivada:

$$
f'(x) = 2x - 4
$$

2. Ponto crítico:

$$
2x - 4 = 0 \Rightarrow x = 2
$$

3. Segunda derivada:

$$
f''(x) = 2 > 0
$$

Logo, $x=2$ é mínimo local (e global, pois $f$ é convexa).

---

### **Exemplo 2 — Máximo local**

$$
f(x) = -x^2 + 3x
$$

1. Derivada:

$$
f'(x) = -2x + 3
$$

2. Ponto crítico:

$$
x = \frac{3}{2}
$$

3. Segunda derivada:

$$
f''(x) = -2 < 0
$$

Logo, é máximo local.

---

### **Exemplo 3 — Intervalo fechado**

Minimize $f(x) = x^3 - 3x$ em $[-2,2]$.

1. Derivada:

$$
f'(x) = 3x^2 - 3
$$

2. Pontos críticos:

$$
3x^2 - 3 = 0 \Rightarrow x = \pm 1
$$

3. Avaliar:

- $f(-2) = -2$  
- $f(-1) = 2$  
- $f(1) = -2$  
- $f(2) = 2$  

Mínimo global: $f(-2) = f(1) = -2$.

---

<a id="exercicios"></a>
# 8. 📝 Exercícios

1. Encontre os pontos críticos de $f(x) = x^3 - 6x^2 + 9x$.  
2. Determine onde $f(x) = x^4 - 2x^2$ tem máximos e mínimos.  
3. Minimize $f(x) = x + \frac{1}{x}$ em $[0.5, 3]$.  
4. Mostre que $f(x) = e^x$ não tem pontos críticos.  
5. Prove que $f(x) = x^2 + 1$ é estritamente convexa.  

---

<a id="conexoes"></a>
# 9. 🔗 Conexões com Outros Capítulos

- [Convexidade](ca://s?q=Criar_Notebook_3_Propriedades_de_Funcoes) — base para condições de otimalidade.  
- [Gradientes](ca://s?q=Criar_Notebook_9_Gradientes) — generalização para $\mathbb{R}^n$.  
- [Gaussianas](ca://s?q=Criar_Notebook_7_Gaussianas) — MLE usa otimização.  
- [Séries Geométricas](ca://s?q=Criar_Notebook_2_Somas_Produtos_Series) — análise de convergência.  

---

<a id="resumo"></a>
# 10. 🧾 Resumo Final

Este notebook desenvolveu:

- pontos críticos  
- teste da segunda derivada  
- otimização em intervalos fechados  
- condições necessárias e suficientes  
- convexidade e unicidade  
- exemplos e exercícios  

Este é o passo final antes de entrar em otimização multivariada.

---

[↑ Voltar ao Índice](#indice)
