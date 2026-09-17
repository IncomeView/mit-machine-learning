# LSTM — Desenvolvimento Completo e Detalhado

Moacir, agora vamos **desdobrar LSTM de forma explícita**, linha por linha, equação por equação, e resolver **todas** as dúvidas que você levantou:

- como o LSTM decide o que passa e o que é esquecido,  
- por que isso depende da sigmoide,  
- como cada gate é calculado,  
- como o gradiente flui,  
- como o “loop” entre $h_t$ e $c_t$ funciona,  
- e como isso se relaciona com o gated RNN simples.

Vou organizar tudo de forma **cirúrgica**, sem pular etapas.

---

## 1. ✔ O LSTM decide o que passa ou é esquecido usando gates sigmoides
Sim — isso está correto.

A sigmoide:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

tem saída **entre 0 e 1**.

Isso permite que cada gate funcione como um **controle contínuo**:

- 0 → bloquear completamente  
- 1 → deixar passar completamente  
- valores intermediários → deixar passar parcialmente

É literalmente um **interruptor contínuo**.

---

## 2. ✔ Como obtemos $f_t$, $i_t$, $o_t$?

Você escreveu:

$$
f_t = \sigma(W_{f,h} h_{t-1} + W_{f,x} x_t)
$$

$$
i_t = \sigma(W_{i,h} h_{t-1} + W_{i,x} x_t)
$$

$$
o_t = \sigma(W_{o,h} h_{t-1} + W_{o,x} x_t)
$$

Isso está **100% correto**.

Agora vamos explicar **como isso é obtido**.

### Cada gate é apenas um neurônio (ou vários neurônios) com ativação sigmoide.

Por exemplo, o forget gate:

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

Aqui:

- $W_f$ é a matriz que conecta a entrada $x_t$ ao forget gate  
- $U_f$ é a matriz que conecta o estado oculto anterior $h_{t-1}$ ao forget gate  
- $b_f$ é o bias do forget gate

O mesmo vale para:

- input gate $i_t$  
- output gate $o_t$  
- candidato de memória $\tilde{c}_t$ (que usa tanh)

Ou seja:

> **Cada gate é uma camada neural independente dentro do LSTM.**

Eles são calculados exatamente como qualquer neurônio feedforward.

---

## 3. ✔ Como o LSTM usa esses gates para decidir o que passa ou é esquecido?

A célula de memória é atualizada assim:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

Interpretação:

- $f_t$ decide **quanto do passado manter**  
- $i_t$ decide **quanto do novo conteúdo adicionar**  
- $\tilde{c}_t$ é o **novo conteúdo candidato**

O estado visível é:

$$
h_t = o_t \odot \tanh(c_t)
$$

O output gate decide **quanto da memória expor**.

---

## 4. ✔ Sua dúvida: “Como obtemos o f de $W_{f,h}$? E o mesmo para i e o?”

Vamos explicar isso com precisão.

### O forget gate é uma função de $h_{t-1}$ e $x_t$:

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

Então:

- $W_f$ é aprendido por backpropagation  
- $U_f$ é aprendido por backpropagation  
- $b_f$ é aprendido por backpropagation  

O LSTM **não calcula** $W_f$.  
Ele **aprende** $W_f$ ao longo do treinamento.

O que ele calcula é:

$$
W_f x_t + U_f h_{t-1} + b_f
$$

e aplica sigmoide.

O mesmo vale para:

- $W_i, U_i, b_i$  
- $W_o, U_o, b_o$  
- $W_c, U_c, b_c$

---

## 5. ✔ Sua dúvida: “h é calculado com base em c_t — como é o looping?”

Sim, o LSTM tem dois loops:

### Loop da memória:

$$
c_t \to c_{t+1} \to c_{t+2} \to \dots
$$

### Loop do estado visível:

$$
h_t \to h_{t+1} \to h_{t+2} \to \dots
$$

O forget gate controla o loop da memória:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

O output gate controla o loop do estado visível:

$$
h_t = o_t \odot \tanh(c_t)
$$

Então:

- $h_t$ alimenta os gates do próximo passo  
- $c_t$ alimenta a memória do próximo passo

---

## 6. ✔ O que significa “sigmoid” explicitamente?

A sigmoide é:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Ela transforma qualquer número real em algo entre 0 e 1.

Por isso ela é usada como gate:

- 0 → bloquear  
- 1 → deixar passar  
- valores intermediários → deixar passar parcialmente

---

## 7. ✔ Por que o gated RNN simples também usa sigmoide?

Você escreveu:

$$
g_t = \sigma(W_{g,s} s_{t-1} + W_{g,x} x_t)
$$

Sim — isso é exatamente o mesmo princípio.

O gated RNN simples é uma versão reduzida do LSTM:

- tem apenas **um gate**  
- controla **quanto do estado anterior manter**  
- não tem célula de memória separada  
- não tem input/output gates

Ele é uma “proto-LSTM”.

---

## 8. ✔ Resumo direto para fixar tudo

### A sigmoide é usada porque:

- produz valores entre 0 e 1  
- funciona como gate  
- controla fluxo de informação  
- controla fluxo de gradiente

### Cada gate é calculado como:

$$
\text{gate} = \sigma(W x_t + U h_{t-1} + b)
$$

### O LSTM tem dois loops:

- memória $c_t$  
- estado visível $h_t$

### O LSTM aprende os pesos $W$ e $U$ por backpropagation.

---

