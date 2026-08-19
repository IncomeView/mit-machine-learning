notes.md

<a id="indice"></a>


<a id="unit1"></a>

=====================================  
Unit 1. | Linear Classifiers and Generalizations
=====================================

### 1. Fundamentos do Aprendizado Supervisionado
1. [Feature Vectors](#features)  
2. [Labels](#labels)  
3. [Training Set](#trainset)  
4. [Training Error](#trainerror)  
5. [Generalização e Test Error](#testerror)  
6. [Set of Classifiers](#setclassifiers)  
7. [Classificação Linear e Separação Linear](#linearsep)

### 2. Perceptron e Hinge Loss — O Começo dos Classificadores Lineares
8. [Hinge Loss](#hinge)  
9. [Perceptron](#perceptron)  
10. [SGD — Gradiente Descendente Estocástico](#SGD)

### 3. Margem, SVM e Métodos Online
11. [SVM (Formulação Primal)](#SVM)  
12. [$argmin_\theta$ — Atualização Ótima](#arg)  
13. [PA — Passive‑Aggressive Algorithm](#PA)  
14. [Pegasos — SGD para SVM](#pegasos)

### 4. Risco, Generalização e Regularização
15. [Risco Empírico](#empirico)  
16. [Trade‑off Viés–Variância](#trade)  
17. [Abordagem por Gradiente](#grad)  
18. [Solução Fechada — Equação Normal](#fechada)  
19. [Regularização e Generalização](#generalizacao)


<a id="unit2"></a>

=====================================  
Unit 2. | Nonlinear Classification, Linear regression, Collaborative Filtering
=====================================

### 5. — Linear Regression
1. [Overview da Unit 2](#overview2)  
1. [Regressão Linear — Intuição e Motivação](#reg_intuicao)  
2. [Risco Empírico na Regressão](#risk_reg)  
2. [Perda Quadrática](#quadratic_loss)  
3. [Gradiente para Regressão Linear](#grad_reg)  
3. [SGD para Regressão Linear](#sgd_reg)  
4. [Solução Fechada — Equação Normal](#fechada_reg)  
5. [Regularização L2 (Ridge Regression)](#ridge)  
6. [Generalização na Regressão](#gen_reg)  
6. [Conexão com Lernel Methods e Matrix Completion](#conexao_kernel_matrix)


### 6. — Nonlinear Classification (Kernel Methods)
7. [Kernel Trick](#kernel)  
8. [Kernel Dual](#dual)  
9. [Kernel Rules](#regraskernel)  
10. [Kernel Examples](#exemplos_kernel)  
11. [SVM Kernelizada](#svm_kernelizada)  
12. [Perceptron Kernel](#perceptron_kernel)  
13. [Kernel Softmax](#kernel_softmax)

### 7. — Recommender Systems (Matrix Completion & Low Rank Models)
14. [Matrix Completion](#matrix)  
15. [Modelo Ingênuo — Independência entre Entradas](#ingenuo)  
16. [Low Rank Models](#lowrank)  
17. [Função Objetivo Completa](#objetivo)  
18. [Alternating Minimization](#alter_min)  
19. [Ridge Regression 1D](#ridge_1d)  
20. [Interpretação Bayesiana (Prior Gaussiano)](#bayesiana)  
21. [Fatores Latentes](#latente)


<a id="unit3"></a>

=====================================  
Unit 3. | Neural Networks
=====================================

### 8. Redes Neurais
1. [One‑hot Encoding](#onehot)  
2. [Softmax](#softmax)  
3. [Modelos de Markov (n‑gram)](#markov)  

### 9. Redes Feedforward
4. [Feedforward Networks para Linguagem](#feedforward)  
5. [ReLU, Fronteiras e Regiões Lineares](#relu)  
6. [Embeddings](#embeddings)  
7. [Retropropagação (Backpropagation)](#backprop)  

### 10. (RNNs) Redes Neurais Recorrentes 
8. [Redes Neurais Recorrentes (RNN)](#rnn)  
9. [Gating em Redes Recorrentes](#gating)  
10. [LSTM — Long Short‑Term Memory](#lstm)  
11. [Seq2Seq — Codificação e Decodificação](#seq2seq)  
12. [Probabilidade de Sentenças](#prob_sentencas)  
13. [MLE para n‑gram](#mle_ngram)

### 11. (CNNs) Redes Neurais Convolucionais 




<a id="unit4"></a>

=====================================  
Unit 4. | Probabilistic Models, Mixture Models & EM Algorithm
=====================================

### 12. Modelos Generativos e Misturas
1. [Modelos Generativos](#modelos_generativos)  
2. [Misturas de Distribuições](#misturas_distribuicoes)  
3. [Mistura de Gaussianas (GMM)](#gmm)  
4. [Responsabilidades e Posterior](#responsabilidades_posterior)

### 13. Expectation–Maximization (EM)
5. [Intuição do EM](#intuicao_em)  
6. [Passo E — Expectation](#passo_e)  
7. [Passo M — Maximization](#passo_m)  
8. [Convergência do EM](#convergencia_em)

### 14. Aplicações de GMM
9. [Clustering Probabilístico](#clustering_probabilistico)  
10. [Classificação com GMM](#classificacao_gmm)  
11. [GMM para Imagens (MNIST)](#gmm_mnist)  
12. [Comparação com K‑means](#comparacao_kmeans)

### 15. Variantes e Extensões
13. [Covariâncias Diagonais vs. Completas](#covariancias)  
14. [Misturas de Bernoulli e Multinomial](#misturas_bernoulli_multinomial)  
15. [EM para PCA Probabilístico](#em_pca)  
16. [Mistura de Experts](#mistura_experts)

- ### 15.1 Conteúdos Extras (Expansões)
	- 16.1 [Mistura de Experts para Regressão](#moe_regressao)  
	- 16.2 [Mistura de Experts para Classificação](#moe_classificacao)  
	- 16.3 [Mistura de Experts para Classificação Multiclasse](#moe_multiclasse)  
	- 16.4 [Mistura de Experts para Regressão Não Linear](#moe_regressao_nao_linear)

### 16. Conexões Avançadas
17. [EM em Modelos Latentes](#em_latentes)  
18. [Relação com Variational Inference](#relacao_vi)  
19. [Modelos Generativos Modernos](#modelos_modernos)  
20. [Da GMM aos VAEs](#gmm_vae)


<a id="unit5"></a>

=====================================  
Unit 5. | Reinforcement Learning  
=====================================

### 17. Reinforcement Learning 1 — MDPs e Bellman Equations
1. [Overview — Motivação e Diferenças entre RL e Supervisionado](#lecture17_parte1)
2. [Learning to Control — Intuição do RL](#lecture17_parte2)
3. [Terminologia Fundamental — $S$, $A$, $T$, $R$](#lecture17_parte3)
4. [Função de Utilidade — Recompensas Descontadas](#lecture17_parte4)
5. [Políticas e Funções de Valor — $V^\pi(s)$ e $V^*(s)$](#lecture17_parte5)
6. [Equações de Bellman — Intuição](#lecture17_parte6)
7. [Bellman Optimality — Forma Completa](#lecture17_parte7)
8. [Preparação para Value Iteration](#lecture17_parte8)

### 18. Reinforcement Learning 2 — Value Iteration
9. [Bellman Optimality — Revisão Formal](#lecture18_parte1)
10. [Value Iteration — Atualização Recursiva](#lecture18_parte2)
11. [Convergência — Contração e Ponto Fixo](#lecture18_parte3)
12. [Extração da Política Ótima — $ \pi^*(s) = \arg\max_a [...] $](#lecture18_parte4)
13. [Exemplos em Gridworld — Propagação de Valores](#lecture18_parte5)
14. [Conexão com Planejamento e Controle](#lecture18_parte6)

### 19. Reinforcement Learning 3 — Q-Learning
15. [Função Q — $Q(s,a)$ e Relação com $V(s)$](#lecture19_parte1)
16. [Bellman para Ações — $Q^*(s,a)$](#lecture19_parte2)
17. [Q-Learning — Atualização Estocástica](#lecture19_parte3)
18. [Exploração vs. Exploração — $\varepsilon$-greedy](#lecture19_parte4)
19. [Convergência do Q-Learning](#lecture19_parte5)
20. [Q-Tables — Implementação em Gridworld](#lecture19_parte6)
21. [Conexão com RL Completo — Aprender sem conhecer $T$ e $R$](#lecture19_parte7)
22. [Preparação para o Projeto Final — Agentes que Jogam Jogos de Texto](#lecture19_parte8)


1. [proximo](#px)


<br><br>

----

<a id="features"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Feature Vectors  
=============================

Modelos de aprendizado de máquina não entendem filmes, imagens ou textos diretamente.  
Eles entendem **vetores numéricos**.  
Por isso, o primeiro passo em qualquer tarefa supervisionada é transformar cada exemplo em um **feature vector**.

Um feature vector é:

$$
x \in \mathbb{R}^d
$$

onde cada coordenada representa uma característica mensurável do exemplo.

---

### Como construímos features?

Para cada exemplo (filme, imagem, documento), fazemos perguntas sistemáticas:

- é ação? → 1 ou 0  
- é comédia? → 1 ou 0  
- tem ficção científica? → 1 ou 0  
- duração > 2h? → 1 ou 0  
- contém a palavra “excelente”? → 1 ou 0  

Cada resposta se torna uma coordenada do vetor.

Exemplo:

$$
x = [1, 0, 1, 0, 1]
$$

Esse vetor não é “sobre o filme”.  
Ele é **uma representação numérica padronizada** que o algoritmo consegue manipular.

---

### Geometria dos feature vectors

Cada exemplo vira um ponto no espaço:

- se $x \in \mathbb{R}^2$, é um ponto no plano  
- se $x \in \mathbb{R}^3$, é um ponto no espaço  
- se $x \in \mathbb{R}^{784}$ (MNIST), é um ponto em um espaço de 784 dimensões  

O classificador aprende **fronteiras geométricas** nesse espaço.

---

### Por que feature vectors são fundamentais?

1. **Transformam dados brutos em números.**  
   Sem isso, não existe aprendizado supervisionado.

2. **Permitem comparar exemplos geometricamente.**  
   Distâncias, ângulos e hiperplanos só fazem sentido em $\mathbb{R}^d$.

3. **Permitem generalização.**  
   Dois filmes diferentes podem ter vetores semelhantes → o modelo aprende padrões.

4. **São o ponto de partida para todos os métodos da Unit 1.**  
   Perceptron, hinge loss, SVM, PA, Pegasos — todos operam sobre $x$.

---

### Pensamento aplicado

O feature vector é a ponte entre o mundo real e o modelo matemático.

Ele determina:

- o que o modelo pode aprender  
- o que o modelo não pode aprender  
- a geometria da separação  
- a capacidade de generalização  

Um bom conjunto de features pode transformar um problema difícil em um problema simples.  
Um conjunto ruim pode tornar um problema simples impossível.

---

### Conclusão

Feature vectors são a base do aprendizado supervisionado.  
Eles transformam exemplos em pontos no espaço, permitindo que classificadores lineares aprendam fronteiras geométricas para separar classes.



<br><br>



<a id="labels"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Labels  
=============================

Em aprendizado supervisionado, cada exemplo possui um **label** — o alvo que queremos prever.  
Ele é a resposta correta fornecida durante o treinamento.

Para classificação binária:

$$
y \in \{-1, +1\}
$$

Para classificação multiclasse:

$$
y \in \{1, 2, \dots, K\}
$$

Para regressão:

$$
y \in \mathbb{R}
$$

---

### Intuição

O label é a “verdade” que o modelo tenta aprender.  
Ele define o comportamento desejado:

- +1 → filme que você gosta  
- −1 → filme que você não gosta  
- 1 → imagem de gato  
- 2 → imagem de cachorro  
- 3 → imagem de carro  

Sem labels, não existe aprendizado supervisionado.

---

### Labels e geometria

Quando colocamos os feature vectors no espaço, os labels dizem:

- quais pontos devem ficar de um lado da fronteira  
- quais pontos devem ficar do outro lado  

Assim, o label é o que **orienta a geometria da separação**.

---

### Pensamento aplicado

Labels são simples, mas críticos:

1. **Definem a tarefa.**  
   O modelo não sabe o que é “positivo” ou “negativo” — o label diz.

2. **Guiam o aprendizado.**  
   Cada atualização depende de comparar a predição com o label.

3. **Permitem medir erro.**  
   Sem label, não existe training error.

4. **São o elo entre dados e objetivo.**  
   O modelo aprende a mapear features → labels.

---

### Conclusão

Labels são os alvos do aprendizado supervisionado.  
Eles definem o comportamento desejado e orientam a geometria da separação entre classes.




<br><br>




<a id="trainset"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Training Set  
=============================

O **training set** é o conjunto de exemplos rotulados usado para ensinar o modelo.  
Cada exemplo contém:

- um feature vector  
- um label  

Formalmente:

$$
S_n = \{(x^{(i)}, y^{(i)})\}_{i=1}^n
$$

onde:

- $x^{(i)} \in \mathbb{R}^d$  
- $y^{(i)} \in \{-1, +1\}$  

---

### Intuição

O training set é a “experiência passada” do modelo.  
Ele mostra como o modelo deve se comportar.

O algoritmo nunca vê os dados futuros — apenas o training set.  
Por isso, ele precisa **generalizar**.

---

### Geometria

Cada exemplo do training set é um ponto no espaço com um rótulo.  
O classificador deve encontrar uma fronteira que:

- separa corretamente os pontos  
- generaliza para pontos novos  

---

### Pensamento aplicado

O training set é fundamental porque:

1. **Define o que o modelo aprende.**  
   Se os features forem ruins, o modelo aprende padrões ruins.

2. **Define a dificuldade da tarefa.**  
   Dados muito misturados → fronteira difícil.  
   Dados bem separados → fronteira simples.

3. **Define o risco de overfitting.**  
   Poucos exemplos → modelo memoriza.  
   Muitos exemplos → modelo generaliza.

4. **É a única fonte de informação do algoritmo.**  
   O modelo nunca vê o test set durante o treinamento.

---

### Conclusão

O training set é o conjunto de exemplos rotulados que ensina o modelo.  
Ele define a tarefa, a geometria da separação e a capacidade de generalização.




<br><br>




<a id="trainerror"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Training Error  
=============================

O **training error** mede quantos exemplos do conjunto de treinamento o classificador erra.  
Ele é a primeira métrica que usamos para avaliar um modelo.

Se o classificador é $h(x)$, o training error é:

$$
E_n(h) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[h(x^{(i)}) \neq y^{(i)}]
$$

onde:

- $\mathbf{1}[\cdot]$ é 1 quando a condição é verdadeira  
- $h(x^{(i)})$ é a predição  
- $y^{(i)}$ é o label correto  

---

### Intuição

O training error diz **quão bem o modelo aprendeu os exemplos que já viu**.

- Se $E_n = 0$: o modelo acertou todos os exemplos do treino  
- Se $E_n = 0.5$: o modelo está tão ruim quanto jogar uma moeda  
- Se $E_n$ é alto: o modelo não aprendeu a tarefa nem no treino  

---

### Geometria

Cada exemplo é um ponto no espaço.  
O training error mede quantos pontos estão do lado errado da fronteira linear.

Se a fronteira separa perfeitamente os pontos:

- training error = 0  
- o conjunto é **linearmente separável**

Se não separa:

- training error > 0  
- o conjunto **não é separável**  

---

### Pensamento aplicado

Training error é útil, mas limitado:

1. **Ele só mede desempenho no treino.**  
   Não diz nada sobre dados novos.

2. **Pode ser enganoso.**  
   Um modelo pode ter training error = 0 e ainda assim ser péssimo no test set.

3. **Modelos muito complexos tendem a ter training error baixo.**  
   Mas isso pode significar overfitting.

4. **É apenas o primeiro passo.**  
   O objetivo real é minimizar o test error.

---

### Conclusão

Training error mede quantos exemplos do treino o modelo erra.  
É essencial para avaliar aprendizado inicial, mas não garante boa generalização.




<br><br>


<a id="testerror"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Generalização e Test Error  
=============================

O objetivo do aprendizado supervisionado não é acertar o treino —  
é acertar **dados novos**, nunca vistos.

O **test error** mede exatamente isso:

$$
E_{\text{test}}(h) = \frac{1}{m} \sum_{j=1}^m \mathbf{1}[h(x_{\text{test}}^{(j)}) \neq y_{\text{test}}^{(j)}]
$$

Ele é calculado **apenas** após o treinamento, usando exemplos que o modelo nunca viu.

---

### Intuição

Generalização é a capacidade do modelo de funcionar bem em dados novos.

- training error baixo + test error baixo → modelo bom  
- training error baixo + test error alto → overfitting  
- training error alto + test error alto → underfitting  

---

### Geometria

Um classificador linear aprende uma fronteira baseada nos pontos do treino.  
Se essa fronteira:

- **captura padrões reais** → generaliza  
- **memoriza detalhes específicos do treino** → não generaliza  

Generalização é sobre **como a fronteira se comporta fora dos pontos vistos**.

---

### Por que generalização é difícil?

Porque o modelo só vê o training set, mas precisa funcionar no test set.

Há sempre uma discrepância:

- treino → conhecido  
- teste → desconhecido  

O modelo precisa aprender **padrões gerais**, não detalhes específicos.

---

### Pensamento aplicado

Generalização é o coração do aprendizado de máquina:

1. **O test error é a métrica que realmente importa.**  
   É ele que define se o modelo funciona no mundo real.

2. **Regularização controla generalização.**  
   (Ver itens posteriores.)

3. **Mais dados geralmente melhoram generalização.**

4. **Generalização conecta todas as metodologias da Unit 1.**  
   Perceptron, SVM, PA, Pegasos — todos buscam boa generalização.

---

### Conclusão

Test error mede desempenho em dados novos.  
Generalização é a capacidade de acertar o que o modelo nunca viu — o objetivo final do aprendizado supervisionado.




<br><br>




<a id="setclassifiers"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Set of Classifiers  
=============================

Em aprendizado supervisionado, não escolhemos um único classificador.  
Escolhemos **um conjunto de classificadores possíveis** — uma família de funções —  
e o algoritmo deve encontrar, dentro dessa família, o melhor classificador para o training set.

Chamamos essa família de:

$$
\mathcal{H} = \{h_\theta : \theta \in \Theta\}
$$

onde:

- $\Theta$ é o espaço de parâmetros  
- cada $\theta$ define um classificador diferente  
- $\mathcal{H}$ é o conjunto de todos os classificadores possíveis  

---

### Exemplos de conjuntos de classificadores

- **Classificadores lineares**    $$ h_\theta(x) = \text{sign}(\theta^\top x + \theta_0) $$  
  Aqui, $$\Theta = \mathbb{R}^{d+1}$$

- **Perceptron**  
  Mesmo conjunto linear, mas com atualização específica.

- **SVM**  
  Mesmo conjunto linear, mas com margem máxima.

- **PA / Pegasos**  
  Mesmo conjunto linear, mas com regras de atualização diferentes.

---

### Intuição

O conjunto de classificadores define **o que o modelo é capaz de aprender**.

Se $\mathcal{H}$ contém apenas funções lineares:

- só podemos aprender fronteiras lineares  
- não podemos aprender fronteiras curvas  
- não podemos capturar relações não lineares  

O poder do modelo depende diretamente de $\mathcal{H}$.

---

### Geometria

Cada classificador linear corresponde a um hiperplano:

$$
\theta^\top x + \theta_0 = 0
$$

O conjunto de classificadores lineares é o conjunto de **todos os hiperplanos possíveis** no espaço.

O algoritmo escolhe, dentro desse conjunto, o hiperplano que melhor separa os dados.

---

### Pensamento aplicado

O conjunto de classificadores é a “linguagem” do modelo:

1. **Define a capacidade do modelo.**  
   Se $\mathcal{H}$ é pequeno → modelo simples → alto viés.  
   Se $\mathcal{H}$ é grande → modelo complexo → alta variância.

2. **Define o tipo de fronteira.**  
   Linear, polinomial, radial, etc.

3. **Define o que é possível aprender.**  
   Se a fronteira verdadeira não é linear, nenhum classificador linear funcionará bem.

4. **Conecta com generalização.**  
   Um conjunto muito grande pode levar a overfitting.

---

### Conclusão

O set of classifiers é a família de funções que o modelo pode escolher.  
Ele define a capacidade, a geometria e os limites do aprendizado supervisionado.



<br><br>



<a id="linearsep"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Classificação Linear e Separação Linear  
=============================

Um **classificador linear** decide a classe de um exemplo usando um hiperplano:

$$
h(x) = \text{sign}(\theta^\top x + \theta_0)
$$

Esse hiperplano divide o espaço em duas regiões:

- lado positivo → classe +1  
- lado negativo → classe −1  

---

### Separação linear

Um conjunto de dados é **linearmente separável** se existe algum hiperplano que separa perfeitamente os pontos:

$$
y^{(i)}(\theta^\top x^{(i)} + \theta_0) > 0
\quad \text{para todos os } i
$$

Se isso é possível:

- training error = 0  
- perceptron converge  
- SVM encontra uma margem positiva  

Se não é possível:

- training error > 0  
- perceptron não converge  
- SVM ainda funciona, mas com violações de margem  

---

### Geometria

Cada ponto é um vetor $x \in \mathbb{R}^d$.  
O hiperplano é uma superfície de dimensão $d-1$.

Exemplos:

- em $\mathbb{R}^2$: uma linha  
- em $\mathbb{R}^3$: um plano  
- em $\mathbb{R}^{784}$: um hiperplano de alta dimensão  

A classificação linear é sempre geométrica.

---

### Intuição

Classificadores lineares são simples, rápidos e interpretáveis:

- cada coordenada de $\theta$ indica a importância de uma feature  
- a fronteira é fácil de visualizar (em baixa dimensão)  
- o modelo é eficiente mesmo em alta dimensão  

Mas eles só funcionam bem quando a separação é aproximadamente linear.

---

### Pensamento aplicado

Classificação linear é a base de toda a Unit 1:

1. **Perceptron** aprende um hiperplano por atualizações.  
2. **Hinge loss** mede violações da margem linear.  
3. **SVM** escolhe o hiperplano com maior margem.  
4. **PA** ajusta o hiperplano online.  
5. **Pegasos** usa SGD para otimizar o hiperplano.  

Toda a Unit 1 é sobre **como aprender um hiperplano**.

---

### Conclusão

Classificação linear usa um hiperplano para separar classes.  
É simples, eficiente e a base para perceptron, hinge loss, SVM, PA e Pegasos.




<br><br>




<a id="hinge"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Hinge Loss  
=============================

A hinge loss mede **quanto um classificador linear viola a margem**.  
Ela é a função de perda usada em SVMs, PA e Pegasos — e é o elo entre geometria e otimização.

Para um exemplo $(x, y)$, com $y \in \{-1, +1\}$:

$$
z = y(\theta^\top x + \theta_0)
$$

Interpretando:

- $z > 1$ → ponto correto e fora da margem  
- $0 < z < 1$ → ponto correto, mas dentro da margem  
- $z \le 0$ → ponto incorreto  

A hinge loss é:

$$
L_{\text{hinge}}(z) = \max(0,\, 1 - z)
$$

---

### Intuição

A hinge loss não penaliza apenas erros.  
Ela penaliza **falta de confiança**.

- Se o ponto está do lado certo **mas muito perto da fronteira**, ainda há penalidade.  
- Isso força o modelo a criar **margem**, não apenas separar.

---

### Geometria

A condição $z = 1$ define a **margem**:

$$
y(\theta^\top x + \theta_0) = 1
$$

A hinge loss mede a distância do ponto até essa margem.

- pontos fora da margem → perda = 0  
- pontos dentro da margem → perda > 0  
- pontos do lado errado → perda grande  

---

### Por que hinge loss é melhor que 0–1 loss?

A perda 0–1 é:

- não diferenciável  
- não convexa  
- impossível de otimizar diretamente  

A hinge loss é:

- convexa  
- diferenciável quase sempre  
- otimizada por métodos de gradiente  
- alinhada com a geometria da margem  

---

### Pensamento aplicado

Hinge loss é o coração da Unit 1:

1. **Perceptron** usa uma versão implícita dela.  
2. **SVM** minimiza hinge loss + regularização.  
3. **PA** resolve um argmin envolvendo hinge loss.  
4. **Pegasos** aplica SGD diretamente na hinge loss.

Ela conecta:

- classificação linear  
- margem  
- otimização  
- generalização  

---

### Conclusão

Hinge loss mede violações de margem e permite treinar classificadores lineares robustos.  
Ela é a base matemática da SVM, PA e Pegasos.




<br><br>



<a id="perceptron"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Perceptron  
=============================

O perceptron é o primeiro algoritmo de classificação linear.  
Ele aprende um hiperplano ajustando os pesos sempre que erra um exemplo.

A regra de atualização é:

$$
\theta \leftarrow \theta + y x
$$

quando:

$$
y(\theta^\top x) \le 0
$$

Ou seja, quando o classificador erra.

---

### Intuição

O perceptron funciona como um “empurrão geométrico”:

- se o ponto está do lado errado → empurra a fronteira  
- se está do lado certo → não faz nada  

Ele tenta encontrar um hiperplano que separa os dados.

---

### Geometria

Cada atualização:

- gira o hiperplano  
- na direção de $y x$  
- corrigindo o erro cometido  

Se os dados são linearmente separáveis:

- o perceptron **converge**  
- encontra algum hiperplano separador  

Se não são:

- o perceptron **não converge**  
- continua atualizando para sempre  

---

### Perceptron e hinge loss

O perceptron usa uma perda implícita:

- penaliza apenas erros  
- não penaliza violações de margem  
- não busca margem máxima  

Por isso, ele é menos robusto que SVM.

---

### Representação dual

Após várias atualizações:

$$
\theta = \sum_{i=1}^n \alpha_i y_i x_i
$$

onde:

- $\alpha_i$ = número de erros no exemplo $i$

Isso conecta perceptron com kernels (na Unit 2).

---

### Pensamento aplicado

O perceptron é importante porque:

1. **É o primeiro algoritmo linear.**  
   Simples, rápido, histórico.

2. **Mostra como aprender um hiperplano por atualizações.**

3. **Motiva hinge loss.**  
   Perceptron penaliza erros; hinge penaliza falta de margem.

4. **Motiva SVM.**  
   SVM é perceptron + margem + regularização.

5. **Motiva PA e Pegasos.**  
   Ambos são evoluções do perceptron.

---

### Conclusão

O perceptron aprende um hiperplano corrigindo erros.  
É simples, elegante e a base conceitual para SVM, PA e Pegasos.



<br><br>




<a id="SGD"></a>
[$\Uparrow$ Índice](#indice)

=============================  
SGD — Stochastic Gradient Descent  
=============================

SGD é o método de otimização mais usado em aprendizado de máquina.  
Ele ajusta os parâmetros do modelo usando **um exemplo por vez**, tornando o treinamento rápido e escalável.

Se queremos minimizar uma função de perda:

$$
J(\theta)
$$

o SGD atualiza:

$$
\theta \leftarrow \theta - \eta_t \nabla_\theta J(\theta; x^{(i)}, y^{(i)})
$$

onde:

- $\eta_t$ é a taxa de aprendizado  
- $(x^{(i)}, y^{(i)})$ é um exemplo escolhido aleatoriamente  
- o gradiente é calculado **somente nesse exemplo**  

---

### Intuição

SGD é como “descer a montanha” usando passos pequenos baseados em informações locais.

- Gradiente completo → caro, usa todos os dados  
- SGD → barato, usa apenas um exemplo por vez  

Ele é ruidoso, mas rápido.

---

### Por que SGD é essencial na Unit 1?

Porque todos os modelos lineares modernos são treinados com SGD:

- perceptron → atualização equivalente a SGD  
- SVM → Pegasos é SGD na hinge loss  
- PA → resolve um argmin que se comporta como SGD  
- regressão linear → pode ser treinada com SGD  

SGD é a ponte entre:

- perda  
- gradiente  
- atualização  
- convergência  

---

### Geometria

Cada atualização move o hiperplano:

- um pouco na direção correta  
- baseado em um único ponto  
- com passos pequenos  

O hiperplano vai “girando” até encontrar uma boa separação.

---

### Taxa de aprendizado

A taxa $\eta_t$ controla:

- velocidade  
- estabilidade  
- convergência  

Pegasos usa:

$$
\eta_t = \frac{1}{\lambda t}
$$

que garante convergência estável.

---

### Pensamento aplicado

SGD é fundamental porque:

1. **Escala para milhões de exemplos.**  
   Perfeito para Bag‑of‑Words.

2. **É simples e eficiente.**

3. **Funciona bem com hinge loss.**

4. **É a base de Pegasos.**

5. **É usado em praticamente todos os modelos modernos.**

---

### Conclusão

SGD atualiza o modelo usando um exemplo por vez.  
É rápido, escalável e a base de perceptron, SVM, PA e Pegasos.



<br><br>




<a id="SVM"></a>
[$\Uparrow$ Índice](#indice)

=============================  
SVM — Formulação Primal  
=============================

A SVM (Support Vector Machine) é o classificador linear mais importante da Unit 1.  
Ela busca um hiperplano que **maximiza a margem** entre as classes.

O hiperplano é:

$$
f(x) = \theta^\top x + \theta_0
$$

A decisão é:

- $f(x) > 0$ → +1  
- $f(x) < 0$ → −1  

---

### Margem geométrica

A margem é:

$$
\text{margem} = \frac{1}{\|\theta\|}
$$

Maximizar a margem = minimizar $\|\theta\|$.

---

### Hinge loss e SVM

Para cada exemplo:

$$
z_i = y_i(\theta^\top x_i + \theta_0)
$$

A hinge loss penaliza violações:

$$
L_h(z_i) = \max(0,\, 1 - z_i)
$$

---

### Formulação primal da SVM

A SVM resolve:

$$
\min_{\theta, \theta_0}
\quad
\frac{\lambda}{2}\|\theta\|^2
+
\frac{1}{n}\sum_{i=1}^n
\max(0,\, 1 - y_i(\theta^\top x_i + \theta_0))
$$

Essa função combina:

1. **Regularização L2**  $$\frac{\lambda}{2}\|\theta\|^2$$  
   controla complexidade e aumenta margem.

2. **Hinge loss**  
   penaliza erros e violações de margem.

---

### Intuição geométrica

A SVM ajusta o hiperplano para:

- **girar** quando há erro  
- **afastar** quando há violação de margem  
- **encolher** quando tudo está correto  

O hiperplano final é determinado pelos **support vectors** —  
os pontos mais próximos da fronteira.

---

### Conexão com perceptron

Perceptron:

- penaliza apenas erros  
- não cria margem  
- não regulariza  

SVM:

- penaliza erros **e violações de margem**  
- cria margem máxima  
- regulariza para evitar overfitting  

SVM é perceptron + margem + regularização.

---

### Conexão com Pegasos

Pegasos implementa **SGD na formulação primal da SVM**.

Atualizações:

- se viola a margem → passo corretivo + regularização  
- se não viola → apenas regularização  

Pegasos é literalmente SVM primal + SGD.

---

### Pensamento aplicado

SVM é fundamental porque:

1. **Maximiza margem → melhor generalização.**  
2. **É convexa → solução única.**  
3. **É robusta → funciona bem em alta dimensão.**  
4. **É base para Pegasos.**  
5. **É ponte para kernels (Unit 2).**

---

### Conclusão

A SVM primal aprende um hiperplano com margem máxima usando hinge loss e regularização.  
É o classificador linear mais robusto e a base para Pegasos e kernels.




<br><br>



<a id="arg"></a>
[$\Uparrow$ Índice](#indice)

=============================  
$argmin_\theta$ — Atualização Ótima  
=============================

A notação **argmin** aparece sempre que queremos encontrar o valor de $\theta$ que **minimiza** uma função.  
Ela não retorna o valor mínimo da função — ela retorna **o parâmetro que atinge esse mínimo**.

Formalmente:

$$
\theta^* = \arg\min_\theta f(\theta)
$$

Isso significa:

> “Qual é o valor de $\theta$ que torna $f(\theta)$ o menor possível?”

---

### Exemplo simples

Se:

$$
f(\theta) = (\theta - 2)^2
$$

então:

- o valor mínimo da função é 0  
- o ponto onde isso ocorre é:

$$
\arg\min_\theta f(\theta) = 2
$$

---

### Por que argmin aparece na Unit 1?

Porque vários algoritmos atualizam $\theta$ **resolvendo um pequeno problema de otimização**.

Exemplo do Passive‑Aggressive (PA):

$$
\theta^{(k+1)} = \arg\min_\theta 
\left[
\frac{\lambda}{2}\|\theta - \theta^{(k)}\|^2
+ L_h(y\,\theta^\top x)
\right]
$$

Aqui:

- o primeiro termo mantém $\theta$ perto do valor anterior  
- o segundo termo corrige o erro ou violação de margem  
- o argmin escolhe **exatamente o melhor $\theta$** que equilibra esses dois objetivos  

---

### Intuição geométrica

Resolver um argmin é como:

- olhar para a superfície da função  
- encontrar o ponto mais baixo  
- mover $\theta$ exatamente para esse ponto  

É uma atualização **ótima**, não apenas um passo de gradiente.

---

### Pensamento aplicado

Argmin é importante porque:

1. **Formaliza atualizações ótimas.**  
   PA usa argmin para encontrar o menor passo que corrige a margem.

2. **Conecta otimização com geometria.**  
   O argmin escolhe o hiperplano que melhor resolve o conflito atual.

3. **É a base de métodos proximais.**  
   PA é um método proximal — e argmin é o operador central.

4. **Mostra a diferença entre PA e perceptron.**  
   Perceptron usa uma regra fixa; PA usa uma atualização ótima.

---

### Conclusão

Argmin retorna o parâmetro que minimiza uma função.  
Ele formaliza atualizações ótimas e é essencial para entender PA e outros métodos proximais.




<br><br>



<a id="PA"></a>
[$\Uparrow$ Índice](#indice)

=============================  
PA — Passive‑Aggressive Algorithm  
=============================

O algoritmo **Passive‑Aggressive (PA)** é um classificador linear **online** que combina:

- perceptron  
- hinge loss  
- regularização proximal  

Ele ajusta o hiperplano **somente quando necessário**, equilibrando dois objetivos:

1. **Manter $\theta$ próximo do valor anterior** (comportamento passivo)  
2. **Corrigir violações de margem** (comportamento agressivo)

---

### A atualização do PA

O PA resolve:

$$
\theta^{(k+1)} = \arg\min_\theta 
\left[
\frac{\lambda}{2}\|\theta - \theta^{(k)}\|^2
+ L_h(y\,\theta^\top x)
\right]
$$

A solução fechada é:

$$
\theta^{(k+1)} = \theta^{(k)} + \eta\,y\,x
$$

onde:

$$
\eta = \frac{\max(0,\, 1 - y\,\theta^{(k)\top}x)}{\lambda + \|x\|^2}
$$

---

### Intuição

O PA decide:

- **passivo**: se o ponto satisfaz a margem ($y\,\theta^\top x \ge 1$), não faz nada  
- **agressivo**: se viola a margem, corrige com o menor passo possível  

Ele é “agressivo” apenas quando precisa.

---

### Geometria

Cada atualização:

- gira o hiperplano na direção de $y x$  
- com intensidade controlada por $\eta$  
- corrigindo apenas o necessário para satisfazer a margem  

O PA é como uma SVM **local**, resolvendo um mini‑problema de margem a cada exemplo.

---

### Relação com perceptron e SVM

| Algoritmo | Atualiza quando | Passo | Margem | Regularização |
|----------|----------------|--------|--------|----------------|
| Perceptron | erro | fixo | não | não |
| SVM | global | ótimo | sim | sim |
| PA | violação de margem | ótimo | sim | sim |

PA é literalmente:

> perceptron + hinge loss + regularização proximal

---

### Pensamento aplicado

PA é importante porque:

1. **É online.**  
   Perfeito para fluxos de dados.

2. **É robusto.**  
   Usa margem e regularização.

3. **É eficiente.**  
   Resolve um problema pequeno por exemplo.

4. **É elegante.**  
   Atualização ótima, não heurística.

5. **É ponte para Pegasos.**  
   Pegasos é PA + SGD + SVM.

---

### Conclusão

PA é um classificador linear online que ajusta o hiperplano apenas quando necessário.  
Ele combina perceptron, hinge loss e regularização proximal, oferecendo atualizações ótimas e robustas.



<br><br>



<a id="pegasos"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Pegasos —  
*Primal Estimated sub-GrAdient SOlver for SVM*  
=============================

Pegasos é o algoritmo que torna a SVM **escalável**.  
Ele aplica **SGD diretamente na formulação primal da SVM**, usando hinge loss + regularização L2.

A função objetivo da SVM primal é:

$$
J(\theta) = \frac{\lambda}{2}\|\theta\|^2 + L_h(y\,\theta^\top x)
$$

Pegasos atualiza $\theta$ usando um único exemplo por vez.

---

### Atualização do Pegasos

A taxa de aprendizado é:

$$
\eta_t = \frac{1}{\lambda t}
$$

E a atualização depende da margem:

#### Caso 1 — Viola a margem  
$$
y(\theta^\top x) \le 1
$$

Atualização:

$$
\theta \leftarrow (1 - \eta_t \lambda)\theta + \eta_t y x
$$

#### Caso 2 — Satisfaz a margem  
$$
y(\theta^\top x) > 1
$$

Atualização:

$$
\theta \leftarrow (1 - \eta_t \lambda)\theta
$$

---

### Intuição

Pegasos é literalmente:

> **SVM + SGD + hinge loss + regularização L2**

Ele faz:

- **encolhimento** dos pesos (regularização)  
- **correção** quando há violação de margem  
- **passos decrescentes** para garantir convergência  

---

### Geometria

Cada atualização:

- gira o hiperplano quando necessário  
- encolhe o vetor $\theta$ para aumentar a margem  
- move a fronteira de forma suave e controlada  

Pegasos busca um hiperplano com **margem máxima**, como a SVM, mas usando apenas um exemplo por vez.

---

### Relação com perceptron, PA e SVM

| Algoritmo | Atualiza quando | Passo | Margem | Regularização | Tipo |
|----------|----------------|--------|--------|----------------|------|
| Perceptron | erro | fixo | não | não | online |
| PA | violação de margem | ótimo | sim | sim | online |
| SVM | global | ótimo | sim | sim | batch |
| Pegasos | violação de margem | $1/(\lambda t)$ | sim | sim | SGD |

Pegasos é o ponto ideal entre:

- simplicidade do perceptron  
- robustez da SVM  
- eficiência do SGD  
- elegância do PA  

---

### Pensamento aplicado

Pegasos é fundamental porque:

1. **Escala para milhões de exemplos.**  
   Perfeito para Bag‑of‑Words.

2. **É simples e eficiente.**

3. **É a implementação prática da SVM primal.**

4. **É usado em aplicações reais.**

5. **Conecta toda a Unit 1.**  
   Perceptron → hinge → SVM → PA → Pegasos.

---

### Conclusão

Pegasos é SGD aplicado à SVM primal.  
Ele combina eficiência, simplicidade e robustez, tornando-se o método mais prático para treinar classificadores lineares modernos.




<br><br>



<a id="empirico"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Risco Empírico  
=============================

O risco empírico mede o erro médio do modelo **no conjunto de treinamento**.  
Ele é a versão matemática do training error.

Para uma função de perda $L$, o risco empírico é:

$$
R_n(\theta)
= \frac{1}{n} \sum_{i=1}^n L\big(y^{(i)}, f_\theta(x^{(i)})\big)
$$

---

### Exemplo: perda quadrática

Na regressão linear:

$$
L(y, f_\theta(x)) = \frac{1}{2}(y - \theta^\top x)^2
$$

Então:

$$
R_n(\theta)
= \frac{1}{n} \sum_{i=1}^n \frac{1}{2}(y^{(i)} - \theta^\top x^{(i)})^2
$$

---

### Intuição

O risco empírico mede **o quão bem o modelo explica os dados que já viu**.

- risco empírico baixo → bom desempenho no treino  
- risco empírico alto → modelo não aprendeu bem  

Mas ele **não garante** bom desempenho no test set.

---

### Geometria

Minimizar risco empírico significa:

- ajustar o hiperplano  
- para reduzir erros nos pontos do treino  
- sem considerar pontos futuros  

Isso pode levar a overfitting.

---

### Risco verdadeiro

O risco verdadeiro é:

$$
R(\theta) = \mathbb{E}\big[L(y, f_\theta(x))\big]
$$

Mas não conhecemos a distribuição real dos dados.  
Por isso usamos $R_n(\theta)$ como aproximação.

---

### Pensamento aplicado

Risco empírico é central porque:

1. **É o objetivo de treinamento.**  
   Todos os algoritmos da Unit 1 minimizam risco empírico (direta ou indiretamente).

2. **Conecta perda, gradiente e atualização.**

3. **Explica overfitting.**  
   Minimizar demais o risco empírico pode prejudicar generalização.

4. **Motiva regularização.**  
   Regularização controla o risco verdadeiro, não apenas o empírico.

---

### Conclusão

Risco empírico é o erro médio no treino.  
Ele guia o aprendizado, mas não garante boa generalização — por isso regularização é essencial.



<br><br>



<a id="trade"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Trade‑off Viés–Variância  
=============================

O erro de um modelo supervisionado tem três componentes:

$$
\text{Erro} = \text{Viés}^2 + \text{Variância} + \text{Erro Irredutível}
$$

Essa decomposição explica **por que modelos simples e modelos complexos falham de maneiras diferentes**.

---

### Modelo verdadeiro e ruído

Assumimos que existe uma função verdadeira:

$$
y = f(x) + \epsilon
$$

onde:

- $\epsilon$ é ruído  
- $\mathbb{E}[\epsilon] = 0$  
- $\mathbb{V}[\epsilon] = \sigma^2$  

O modelo aprende uma aproximação $\hat{f}(x)$.

---

### Viés

$$
\text{Viés}^2 = \big(f(x) - \mathbb{E}[\hat{f}(x)]\big)^2
$$

Viés mede **o quanto o modelo médio se afasta da função verdadeira**.

- modelos simples → alto viés  
- modelos complexos → baixo viés  

---

### Variância

$$
\text{Variância} = \mathbb{V}[\hat{f}(x)]
$$

Variância mede **o quanto o modelo muda quando treinado com diferentes conjuntos de dados**.

- modelos simples → baixa variância  
- modelos complexos → alta variância  

---

### Erro irredutível

$$
\mathbb{V}[\epsilon]
$$

É o ruído dos dados.  
Nenhum modelo consegue reduzir esse termo.

---

### Intuição

Modelos simples:

- não capturam padrões  
- têm alto viés  
- são estáveis (baixa variância)  
- sofrem de **underfitting**

Modelos complexos:

- capturam muitos padrões  
- têm baixo viés  
- são instáveis (alta variância)  
- sofrem de **overfitting**

O objetivo é encontrar o ponto onde:

$$
\text{Viés}^2 + \text{Variância}
$$

é mínimo.

---

### Papel da regularização

Regularização L2:

$$
J(\theta)
= \frac{1}{2n}\|y - X\theta\|^2 + \frac{\lambda}{2}\|\theta\|^2
$$

- aumenta viés  
- reduz variância  
- melhora generalização  

$\lambda$ controla o equilíbrio.

---

### Pensamento aplicado

O trade‑off viés–variância explica:

1. **por que SVM generaliza bem** (margem + regularização)  
2. **por que perceptron pode overfitar** (sem regularização)  
3. **por que Pegasos é estável** (passos decrescentes + regularização)  
4. **por que regularização é essencial**  

Ele conecta toda a Unit 1 com o objetivo final: **generalização**.

---

### Conclusão

O trade‑off viés–variância explica como modelos simples e complexos erram.  
Regularização controla esse equilíbrio e melhora a generalização.




<br><br>



<a id="grad"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Abordagem por Gradiente  
=============================

A abordagem por gradiente é a base de quase todos os algoritmos modernos.  
Ela ajusta os parâmetros na direção que **reduz a perda mais rapidamente**.

Se queremos minimizar:

$$R_n(\theta)= \frac{1}{n} \sum_{t=1}^n L\big(y^{(t)}, f_\theta(x^{(t)})\big)$$

então o gradiente é:

$$\nabla_\theta R_n(\theta)= \frac{1}{n} \sum_{t=1}^n \nabla_\theta L\big(y^{(t)},f_\theta(x^{(t)})\big)$$

---

### Descida de gradiente (batch)

Atualização:

$$
\theta \leftarrow \theta - \eta \nabla_\theta R_n(\theta)
$$

Usa **todos os exemplos** em cada passo.

---

### SGD (estocástico)

Atualização:

$$
\theta \leftarrow \theta - \eta_t \nabla_\theta L\big(y^{(i)}, f_\theta(x^{(i)})\big)
$$

Usa **um único exemplo** por vez.

É mais rápido e escalável.

---

### Exemplo: regressão linear

Perda quadrática:

$$L = \frac{1}{2}(y - \theta^\top x)^2$$

Gradiente:

$$
\nabla_\theta L = - (y - \theta^\top x)x
$$

Atualização:

$$
\theta \leftarrow \theta + \eta (y - \theta^\top x)x
$$

---

### Geometria

Cada atualização:

- move o hiperplano  
- na direção que reduz o erro  
- com intensidade controlada pela taxa de aprendizado  

O hiperplano vai “girando” até encontrar uma boa separação.

---

### Conexão com Unit 1

- **Perceptron** é gradiente da perda 0–1 (aproximado).  
- **SVM** usa gradiente da hinge loss + regularização.  
- **Pegasos** é SGD na SVM primal.  
- **PA** resolve um argmin equivalente a um passo proximal de gradiente.  

Gradiente é o mecanismo unificador da Unit 1.

---

### Pensamento aplicado

A abordagem por gradiente é essencial porque:

1. **Escala para grandes datasets.**  
2. **Funciona com perdas convexas (hinge).**  
3. **É base para SVM, PA, Pegasos.**  
4. **Conecta otimização com geometria.**

Sem gradiente, não existe aprendizado moderno.

---

### Conclusão

A abordagem por gradiente ajusta o modelo na direção que reduz a perda.  
Ela é a base matemática de perceptron, SVM, PA e Pegasos.




<br><br>



<a id="fechada"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Solução Fechada — Equação Normal  
=============================

Alguns modelos lineares permitem uma **solução fechada**, ou seja, uma fórmula direta para encontrar os parâmetros ótimos sem usar gradiente.

O exemplo clássico é a **regressão linear com perda quadrática**.

---

### Forma matricial

Organizamos os dados em:

- $X \in \mathbb{R}^{n \times d}$ — matriz de features  
- $y \in \mathbb{R}^n$ — vetor de respostas  
- $\theta \in \mathbb{R}^d$ — parâmetros  

O modelo é:

$$f_\theta(x) = \theta^\top x$$

O risco empírico quadrático é:

$$R_n(\theta) = \frac{1}{2n}\|y - X\theta\|^2$$

---

### Gradiente

Usando:

$$
\nabla_\theta \|y - X\theta\|^2 = -2X^\top(y - X\theta)
$$

temos:

$$
\nabla_\theta R_n(\theta) = -\frac{1}{n} X^\top (y - X\theta)
$$

No mínimo da função convexa:

$$
\nabla_\theta R_n(\theta) = 0
$$

Logo:

$$X^\top X \theta = X^\top y$$

Essa é a **equação normal**.

---

### Solução fechada

Se $X^\top X$ é invertível:

$$
\theta^* = (X^\top X)^{-1} X^\top y
$$

Essa é a solução exata da regressão linear.

---

### Regularização (ridge regression)

Quando $X^\top X$ é mal condicionado ou não invertível, adicionamos regularização L2:

$$J(\theta)= \frac{1}{2n}\|y - X\theta\|^2 + \frac{\lambda}{2}\|\theta\|^2$$

A condição de otimalidade leva a:

$$
(X^\top X + \lambda I)\theta = X^\top y
$$

E a solução fechada regularizada é:

$$
\theta^* = (X^\top X + \lambda I)^{-1} X^\top y
$$

---

### Intuição geométrica

A solução fechada encontra o $\theta$ que:

- minimiza o erro quadrático  
- projeta $y$ no subespaço gerado pelas colunas de $X$  
- escolhe o hiperplano que melhor se ajusta aos dados  

Com regularização, ela:

- evita hiperplanos instáveis  
- controla variância  
- melhora generalização  

---

### Pensamento aplicado

A solução fechada é importante porque:

1. **Mostra como otimização pode ser exata.**  
2. **Conecta álgebra linear com aprendizado.**  
3. **Motiva regularização L2.**  
4. **Explica por que gradiente é necessário em modelos mais complexos.**  
5. **É a base para regressão linear, ridge e métodos clássicos.**

---

### Conclusão

A equação normal fornece uma solução exata para regressão linear.  
Com regularização, ela se torna estável e melhora generalização.




<br><br>


<a id="generalizacao"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Regularização e Generalização  
=============================

Regularização é o mecanismo que controla a complexidade do modelo.  
Ela evita overfitting e melhora a capacidade de generalização.

---

### O problema

Minimizar apenas o risco empírico:

$$R_n(\theta)$$

pode levar a modelos que:

- funcionam muito bem no treino  
- funcionam mal no teste  

Isso é **overfitting**.

---

### Regularização L2

Adicionamos um termo que penaliza pesos grandes:

$$J(\theta)= R_n(\theta) + \frac{\lambda}{2}\|\theta\|^2$$

onde:

- $\lambda$ controla a força da penalização  
- $\|\theta\|^2$ mede a complexidade do modelo  

---

### Intuição geométrica

Regularização L2 “puxa” o vetor $\theta$ para perto da origem.

Quando $\lambda$ é grande:

$$
\theta \to 0
$$

e o modelo se torna quase constante:

$$f(x) \approx \theta_0$$

Isso reduz variância, mas aumenta viés.

---

### Efeito na generalização

- **$\lambda$ pequeno** → modelo complexo → baixa viés, alta variância → risco de overfitting  
- **$\lambda$ grande** → modelo simples → alta viés, baixa variância → risco de underfitting  

O objetivo é encontrar o equilíbrio.

---

### Conexão com Unit 1

Regularização aparece em:

- **SVM** → controla margem  
- **PA** → controla estabilidade  
- **Pegasos** → controla convergência  
- **Regressão linear** → evita instabilidade  
- **Trade‑off viés–variância** → explica por que regularização funciona  

Regularização é o elo entre:

- risco empírico  
- risco verdadeiro  
- generalização  

---

### Pensamento aplicado

Regularização é essencial porque:

1. **Evita overfitting.**  
2. **Controla variância.**  
3. **Melhora generalização.**  
4. **Estabiliza soluções fechadas.**  
5. **É usada em praticamente todos os modelos modernos.**

---

### Conclusão

Regularização controla a complexidade do modelo e melhora generalização.  
Ela é essencial para evitar overfitting e garantir desempenho em dados novos.





<br><br>




<a id="overview2"></a>
[$\Uparrow$ Índice](#indice)

=====================================  
Overview da Unit 2  
=====================================

A Unit 2 expande o alcance dos modelos lineares da Unit 1.  
Ela mostra que, com as ferramentas certas, métodos lineares podem se tornar:

- **não lineares** (via kernels)  
- **contínuos** (via regressão linear)  
- **colaborativos** (via matrix completion e low rank)  

A Unit 2 é onde percebemos que a simplicidade dos modelos lineares é apenas o começo.  
Com as técnicas certas, eles se tornam **arbitrariamente poderosos**.

---

### Três pilares da Unit 2

A Unit 2 é organizada em três blocos conceituais:

1. **Linear Regression (Lecture 5)**  
   - prever valores contínuos  
   - minimizar erro quadrático  
   - gradiente descendente  
   - solução fechada  
   - regularização L2  
   - generalização

2. **Kernel Methods (Lecture 6)**  
   - transformar classificadores lineares em não lineares  
   - kernel trick  
   - dual  
   - SVM kernelizada  
   - perceptron kernel  
   - kernels clássicos (polinomial, RBF, etc.)

3. **Recommender Systems (Lecture 7)**  
   - matrix completion  
   - modelos low rank  
   - alternating minimization  
   - ridge regression 1D  
   - fatores latentes  
   - interpretação bayesiana

---

### A ponte com a Unit 1

Na Unit 1 aprendemos:

- hiperplanos  
- margem  
- hinge loss  
- perceptron  
- SVM  
- PA  
- Pegasos  
- generalização  
- regularização  

A Unit 2 pega essa base e mostra:

- como prever valores contínuos (regressão)  
- como tornar modelos lineares não lineares (kernels)  
- como usar regressão para recomendação (matrix completion)

---

### Intuição geral

A Unit 2 responde três perguntas fundamentais:

1. **Como prever números reais?**  
   → regressão linear

2. **Como tornar modelos lineares não lineares sem mudar o algoritmo?**  
   → kernel trick

3. **Como prever valores faltantes em uma matriz (ex.: notas de filmes)?**  
   → matrix completion + low rank

Essas três ideias formam a base de muitos sistemas modernos:

- previsão de preços  
- classificação não linear  
- recomendação de filmes, músicas, produtos  
- sistemas de filtragem colaborativa  
- modelos de regressão em larga escala

---

### Conclusão

A Unit 2 mostra que modelos lineares são muito mais poderosos do que parecem.  
Com regressão, kernels e low rank, eles se tornam ferramentas universais para previsão, classificação e recomendação.



<br><br>



<a id="reg_intuicao"></a>
[$\Uparrow$ Índice](#indice)

=====================================  
Regressão Linear — Intuição e Motivação  
=====================================

Na Unit 1 aprendemos a prever **rótulos discretos** (+1 ou −1).  
Mas muitos problemas reais exigem prever **valores contínuos**:

- preço de uma ação amanhã  
- temperatura daqui a 3 horas  
- quantidade de vendas no próximo mês  
- nota que um usuário dará a um filme  
- velocidade de um carro em função do tempo  

Classificação responde “sim ou não”.  
Regressão responde **“quanto?”**.

A regressão linear é o modelo mais simples e mais usado para prever valores contínuos.

---

### O modelo

A regressão linear assume que o valor a ser previsto é uma **combinação linear** das features:

$$f_\theta(x) = \theta^\top x + \theta_0$$

Para simplificar a notação (como na lecture), assumimos:

$$
\theta_0 = 0
$$

Então:

$$f_\theta(x) = \theta^\top x$$

---

### Intuição geométrica

Cada exemplo é um ponto em $\mathbb{R}^d$.  
A regressão linear tenta encontrar um **hiperplano** que melhor se ajusta a esses pontos.

- Na classificação, o hiperplano separa classes.  
- Na regressão, o hiperplano **aproxima valores contínuos**.

Se estivermos em $\mathbb{R}^2$:

- $x$ é um ponto no plano  
- $f_\theta(x)$ é uma linha  
- a regressão linear escolhe a linha que melhor se ajusta aos dados  

Se estivermos em $\mathbb{R}^3$:

- $x$ é um ponto no espaço  
- $f_\theta(x)$ é um plano  

Em alta dimensão, é sempre um hiperplano.

---

### Por que isso funciona?

Mesmo quando a relação real entre $x$ e $y$ é complexa, muitas vezes:

- uma combinação linear **aproxima bem**  
- ou podemos transformar as features para que a relação se torne linear  
- ou podemos usar kernels (Lecture 6) para tornar o modelo não linear sem mudar a fórmula

A regressão linear é simples, mas extremamente poderosa quando combinada com:

- boas features  
- regularização  
- kernels  
- modelos low rank  

---

### Motivação prática

Regressão linear é usada em:

- previsão de preços (ações, imóveis, commodities)  
- previsão de demanda  
- previsão de temperatura  
- previsão de tráfego  
- recomendação (como parte de matrix completion)  
- modelos estatísticos clássicos  
- sistemas de controle  
- análise de risco  

Ela é a base de muitos modelos mais avançados.

---

### Conexão com a Unit 1

A regressão linear compartilha a mesma estrutura dos classificadores lineares:

- $f_\theta(x) = \theta^\top x$  
- otimização convexa  
- gradiente descendente  
- regularização L2  
- solução fechada  

A diferença é apenas o **tipo de saída**:

- Unit 1 → saída discreta (classe)  
- Unit 2 → saída contínua (valor)

---

### Conclusão

Regressão linear é o modelo mais simples para prever valores contínuos.  
Ela usa um hiperplano para aproximar dados e serve como base para kernels e sistemas de recomendação.



<br><br>



<a id="risk_reg"></a>
[$\Uparrow$ Índice](#indice)

=====================================  
Risco Empírico na Regressão  
=====================================

Para treinar regressão linear, precisamos medir **o quanto nossas previsões se afastam dos valores reais**.  
Essa medida é o **risco empírico**, que representa o erro médio no conjunto de treinamento.

Se temos exemplos:

$$S_n = \{(x^{(t)}, y^{(t)})\}_{t=1}^n$$

e o modelo:

$$f_\theta(x) = \theta^\top x$$

então o risco empírico é:

$$R_n(\theta)= \frac{1}{n} \sum_{t=1}^n L\big(y^{(t)}, f_\theta(x^{(t)})\big)$$

---

### A escolha da perda: erro quadrático

A perda mais comum na regressão linear é o **erro quadrático**:

$$L(y, f_\theta(x)) = \frac{1}{2}(y - \theta^\top x)^2$$

Então o risco empírico se torna:

$$R_n(\theta)= \frac{1}{2n} \sum_{t=1}^n (y^{(t)} - \theta^\top x^{(t)})^2$$

---

### Intuição

O erro quadrático mede **o quanto a previsão se afasta do valor real**:

- se a diferença é pequena → penalidade pequena  
- se a diferença é grande → penalidade muito maior  

Isso é desejável porque:

- dados reais têm ruído  
- pequenas diferenças são aceitáveis  
- grandes erros devem ser fortemente penalizados  

O quadrado amplifica erros grandes e estabiliza o aprendizado.

---

### Geometria

Cada exemplo $(x^{(t)}, y^{(t)})$ define um ponto no espaço:

- $x^{(t)}$ é o ponto  
- $y^{(t)}$ é o valor que queremos aproximar  

A regressão linear escolhe o hiperplano $\theta^\top x$ que:

- passa “perto” dos pontos  
- minimiza a soma das distâncias verticais ao hiperplano  
- encontra a melhor aproximação linear possível  

O risco empírico mede **a soma dessas distâncias ao quadrado**.

---

### Conexão com generalização

O risco empírico é apenas o erro no treino.  
O objetivo real é minimizar o **risco verdadeiro**:

$$R(\theta) = \mathbb{E}[L(y, f_\theta(x))]$$

Mas como não conhecemos a distribuição real dos dados, usamos:

- risco empírico → aproximação  
- regularização → controle de complexidade  
- validação → estimativa de generalização  

Lecture 5 introduz regularização exatamente para isso.

---

### Conexão com a Unit 1

Assim como na Unit 1:

- risco empírico é a função que o algoritmo tenta minimizar  
- gradiente descendente ajusta $\theta$ para reduzir esse risco  
- regularização controla viés–variância  
- solução fechada existe porque a função é convexa  

A diferença é apenas o tipo de perda:

- Unit 1 → hinge loss  
- Unit 2 → erro quadrático

---

### Pensamento aplicado

Risco empírico é essencial porque:

1. **Define o objetivo da regressão linear.**  
2. **Permite medir o erro de forma contínua.**  
3. **É convexa → garante solução única.**  
4. **É compatível com gradiente e solução fechada.**  
5. **É base para regressão, kernels e matrix completion.**

---

### Conclusão

O risco empírico mede o erro médio da regressão linear no treino.  
Com erro quadrático, ele se torna uma função convexa que podemos otimizar com gradiente ou solução fechada.




<br><br>



<a id="quadratic_loss"></a>
[$\Uparrow$ Índice](#indice)

=====================================  
Perda Quadrática (Quadratic Loss)  
=====================================

A regressão linear precisa de uma forma de medir **o quanto a previsão se afasta do valor real**.  
A função mais usada para isso é a **perda quadrática**, também chamada de **erro quadrático**.

Para um exemplo $(x, y)$, com previsão:

$$f_\theta(x) = \theta^\top x$$

a perda quadrática é:

$$L(y, f_\theta(x)) = \frac{1}{2}(y - \theta^\top x)^2$$

O fator $\frac{1}{2}$ é apenas conveniência matemática — ele simplifica o gradiente.

---

### Intuição

A perda quadrática mede a **distância vertical** entre:

- o valor real $y$  
- a previsão $\theta^\top x$  

Se a diferença é pequena → penalidade pequena.  
Se a diferença é grande → penalidade muito maior.

Isso é desejável porque:

- dados reais têm ruído  
- pequenas diferenças são aceitáveis  
- grandes erros devem ser fortemente penalizados  

A perda quadrática amplifica erros grandes e estabiliza o aprendizado.

---

### Geometria

Cada exemplo $(x, y)$ define um ponto no espaço:

- $x$ é o ponto  
- $y$ é o valor que queremos aproximar  

A regressão linear escolhe o hiperplano $\theta^\top x$ que:

- passa “perto” dos pontos  
- minimiza a soma das distâncias verticais ao hiperplano  
- encontra a melhor aproximação linear possível  

A perda quadrática mede **a distância vertical ao quadrado**.

---

### Por que quadrática?

A perda quadrática tem três propriedades fundamentais:

1. **É convexa**  
   → garante solução única  
   → permite otimização eficiente

2. **É diferenciável em todos os pontos**  
   → permite gradiente descendente  
   → permite solução fechada

3. **Penaliza fortemente erros grandes**  
   → melhora estabilidade  
   → reduz sensibilidade a ruído pequeno

Essas propriedades tornam a regressão linear simples, elegante e poderosa.

---

### Conexão com risco empírico

O risco empírico é a média da perda quadrática:

$$R_n(\theta)= \frac{1}{2n} \sum_{t=1}^n (y^{(t)} - \theta^\top x^{(t)})^2$$

Minimizar o risco empírico significa:

- ajustar $\theta$  
- para reduzir o erro quadrático médio  
- sobre todos os exemplos do treino

---

### Conexão com a Unit 1

Na Unit 1:

- hinge loss mede violações de margem  
- perda quadrática mede distância contínua

Ambas são funções convexas que:

- definem o objetivo  
- permitem gradiente  
- permitem regularização  
- permitem solução fechada (no caso quadrático)

A regressão linear é a versão contínua da SVM linear.

---

### Pensamento aplicado

A perda quadrática é essencial porque:

1. **É simples e interpretável.**  
2. **Funciona bem com dados contínuos.**  
3. **Permite gradiente e solução fechada.**  
4. **É base para regressão, kernels e matrix completion.**  
5. **É usada em praticamente todos os modelos estatísticos clássicos.**

---

### Conclusão

A perda quadrática mede a distância entre previsão e valor real.  
Ela é convexa, diferenciável e amplifica erros grandes — tornando a regressão linear eficiente e robusta.




<br><br>


<a id="grad_reg"></a>
[$\Uparrow$ Índice](#indice)

=====================================  
Gradiente para Regressão Linear  
=====================================

Para minimizar o risco empírico da regressão linear, precisamos ajustar os parâmetros $\theta$ na direção que **reduz o erro mais rapidamente**.  
Essa direção é dada pelo **gradiente da perda quadrática**.

---

### Risco empírico

O risco empírico é:

$$R_n(\theta)= \frac{1}{2n} \sum_{t=1}^n (y^{(t)} - \theta^\top x^{(t)})^2$$

Queremos encontrar:

$$
\theta^* = \arg\min_\theta R_n(\theta)
$$

---

### Gradiente da perda quadrática

Para um único exemplo $(x, y)$:

$$L = \frac{1}{2}(y - \theta^\top x)^2$$

O gradiente em relação a $\theta$ é:

$$
\nabla_\theta L = - (y - \theta^\top x)x
$$

Essa expressão é extremamente importante:

- se a previsão está **abaixo** do valor real → $(y - \theta^\top x)$ é positivo → o gradiente empurra $\theta$ **para cima**  
- se a previsão está **acima** do valor real → o gradiente empurra $\theta$ **para baixo**  

O gradiente sempre empurra $\theta$ na direção que **corrige o erro**.

---

### Atualização por gradiente descendente

A regra de atualização é:

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

Substituindo o gradiente:

$$
\theta \leftarrow \theta + \eta (y - \theta^\top x)x
$$

Essa é a forma mais simples e mais usada de regressão linear via gradiente.

---

### Intuição geométrica

Cada atualização:

- move o hiperplano $\theta^\top x$  
- na direção do vetor $x$  
- com intensidade proporcional ao erro $(y - \theta^\top x)$  

Se o modelo erra muito:

- o passo é grande  
- o hiperplano gira bastante  

Se o modelo erra pouco:

- o passo é pequeno  
- o hiperplano ajusta suavemente  

O gradiente é um mecanismo de **auto‑correção contínua**.

---

### Conexão com SGD

Se usarmos **todos os exemplos** em cada atualização:

$$
\nabla_\theta R_n(\theta) = -\frac{1}{n} \sum_{t=1}^n (y^{(t)} - \theta^\top x^{(t)})x^{(t)}
$$

Isso é **gradiente batch**.

Se usarmos **um exemplo por vez**:

$$
\theta \leftarrow \theta + \eta (y^{(t)} - \theta^\top x^{(t)})x^{(t)}
$$

Isso é **SGD**, que:

- é mais rápido  
- escala melhor  
- funciona bem em grandes datasets  

Lecture 5 mostra exatamente essa versão.

---

### Conexão com a Unit 1

A fórmula de atualização da regressão linear é extremamente parecida com:

- perceptron  
- PA  
- Pegasos  

A diferença é apenas o tipo de perda:

- Unit 1 → hinge loss  
- Unit 2 → perda quadrática

Mas o mecanismo é o mesmo:

> gradiente ajusta o hiperplano para reduzir o erro.

---

### Pensamento aplicado

O gradiente é essencial porque:

1. **Permite treinar regressão linear em larga escala.**  
2. **Funciona mesmo quando a solução fechada é cara ou impossível.**  
3. **É base para redes neurais (Unit 3).**  
4. **É base para matrix completion (Lecture 7).**  
5. **É simples, eficiente e universal.**

---

### Conclusão

O gradiente da perda quadrática empurra $\theta$ na direção que reduz o erro.  
Ele é a base do aprendizado contínuo e conecta regressão linear com kernels, low rank e redes neurais.




<br><br>


<a id="sgd_reg"></a>
[$\Uparrow$ Índice](#indice)

=====================================  
SGD para Regressão Linear  
=====================================

O gradiente descendente tradicional usa **todos os exemplos** para calcular o gradiente.  
Isso funciona bem em datasets pequenos, mas se torna caro quando:

- n é grande  
- os dados chegam em fluxo  
- queremos atualizações rápidas  
- o modelo precisa se adaptar continuamente  

Por isso, assim como na Unit 1, usamos **SGD — Stochastic Gradient Descent**.

---

### Atualização estocástica

Para um único exemplo $(x^{(t)}, y^{(t)})$, a perda é:

$$L_t = \frac{1}{2}(y^{(t)} - \theta^\top x^{(t)})^2$$

O gradiente é:

$$
\nabla_\theta L_t = - (y^{(t)} - \theta^\top x^{(t)})x^{(t)}
$$

A atualização por SGD é:

$$
\theta \leftarrow \theta + \eta_t (y^{(t)} - \theta^\top x^{(t)})x^{(t)}
$$

Essa é a forma mais prática de treinar regressão linear.

---

### Intuição geométrica

Cada atualização:

- **gira o hiperplano** na direção de $x^{(t)}$  
- com intensidade proporcional ao erro $(y^{(t)} - \theta^\top x^{(t)})$  
- corrigindo apenas o necessário para aquele exemplo  

Se o modelo erra muito:

- o passo é grande  
- o hiperplano se ajusta rapidamente  

Se o modelo erra pouco:

- o passo é pequeno  
- o hiperplano se ajusta suavemente  

SGD é um mecanismo de **auto‑correção contínua**.

---

### Taxa de aprendizado

A taxa $\eta_t$ controla:

- velocidade  
- estabilidade  
- convergência  

Uma escolha comum é:

$$
\eta_t = \frac{1}{1 + t}
$$

que diminui ao longo do tempo, garantindo estabilidade.

---

### Comparação com batch gradient

| Método | Usa quantos exemplos? | Custo por passo | Estabilidade | Escalabilidade |
|-------|------------------------|------------------|--------------|----------------|
| Batch Gradient | todos | alto | alta | baixa |
| SGD | 1 | baixo | média | alta |

SGD é preferido em:

- datasets grandes  
- streaming  
- sistemas de recomendação  
- modelos online  
- deep learning (Unit 3)

---

### Conexão com a Unit 1

A atualização da regressão linear via SGD é extremamente parecida com:

- perceptron  
- PA  
- Pegasos  

A diferença é apenas o tipo de perda:

- Unit 1 → hinge loss  
- Unit 2 → perda quadrática

Mas o mecanismo é o mesmo:

> usar um exemplo por vez para ajustar o hiperplano.

---

### Pensamento aplicado

SGD é essencial porque:

1. **Escala para milhões de exemplos.**  
2. **Funciona bem com dados ruidosos.**  
3. **Permite aprendizado online.**  
4. **É base para redes neurais (Unit 3).**  
5. **É usado em matrix completion (Lecture 7).**

---

### Conclusão

SGD ajusta a regressão linear usando um exemplo por vez.  
É rápido, escalável e o método mais prático para treinar modelos contínuos.




<br><br>



<a id="fechada_reg"></a>
[$\Uparrow$ Índice](#indice)

=====================================  
Solução Fechada — Equação Normal  
=====================================

A regressão linear tem uma propriedade especial:  
**sua função de perda é convexa e diferenciável em todos os pontos**.

Isso significa que podemos encontrar a solução **exata**, sem gradiente, sem iterações, sem SGD.

Essa solução é chamada de **Equação Normal**.

---

### Risco empírico

O risco empírico é:

$$R_n(\theta)= \frac{1}{2n} \sum_{t=1}^n (y^{(t)} - \theta^\top x^{(t)})^2$$

Queremos encontrar:

$$
\theta^* = \arg\min_\theta R_n(\theta)
$$

---

### Forma matricial

Organizamos os dados em:

- $X \in \mathbb{R}^{n \times d}$ — matriz de features  
- $y \in \mathbb{R}^n$ — vetor de valores reais  
- $\theta \in \mathbb{R}^d$ — parâmetros  

O modelo é:

$$
f_\theta(x) = \theta^\top x
$$

O risco empírico pode ser escrito como:

$$R_n(\theta)= \frac{1}{2n}\|y - X\theta\|^2$$

---

### Gradiente do risco empírico

O gradiente é:

$$
\nabla_\theta R_n(\theta)
= -\frac{1}{n} X^\top (y - X\theta)
$$

No mínimo da função convexa:

$$
\nabla_\theta R_n(\theta) = 0
$$

Logo:

$$X^\top X \theta = X^\top y$$

Essa é a **equação normal**.

---

### Solução fechada

Se $X^\top X$ é invertível:

$$
\theta^* = (X^\top X)^{-1} X^\top y
$$

Essa é a solução exata da regressão linear.

Sem gradiente.  
Sem iterações.  
Sem SGD.

Apenas álgebra linear.

---

### Intuição geométrica

A solução fechada encontra o vetor $\theta$ que:

- projeta $y$ no subespaço gerado pelas colunas de $X$  
- escolhe o hiperplano que minimiza a soma dos erros quadráticos  
- é a melhor aproximação linear possível dos dados  

Geometricamente, é a **projeção ortogonal** de $y$ no espaço das combinações lineares de $X$.

---

### Quando usar a solução fechada?

Ela é ótima quando:

- o número de features $d$ é pequeno  
- o número de exemplos $n$ não é gigantesco  
- $X^\top X$ é bem condicionado  
- queremos uma solução exata e rápida  

Ela é ruim quando:

- $d$ é muito grande  
- $X^\top X$ é mal condicionado  
- os dados são ruidosos  
- precisamos de aprendizado online  
- queremos escalabilidade (SGD é melhor)

---

### Conexão com regularização

Quando $X^\top X$ não é invertível, adicionamos regularização L2:

$$
(X^\top X + \lambda I)\theta = X^\top y
$$

A solução fechada regularizada é:

$$
\theta^* = (X^\top X + \lambda I)^{-1} X^\top y
$$

Essa é a **ridge regression**, que veremos no próximo item.

---

### Conexão com a Unit 1

Assim como:

- SVM tem solução dual  
- PA tem solução ótima por argmin  
- Pegasos tem solução por SGD  

A regressão linear tem uma solução **exata** por álgebra linear.

---

### Pensamento aplicado

A solução fechada é essencial porque:

1. **Mostra como otimização pode ser exata.**  
2. **Conecta regressão com álgebra linear.**  
3. **Motiva regularização L2.**  
4. **Explica por que gradiente é necessário em alta dimensão.**  
5. **É base para matrix completion (Lecture 7).**

---

### Conclusão

A equação normal fornece uma solução exata para regressão linear.  
Ela é elegante, eficiente e conecta otimização com álgebra linear.




<br><br>



<a id="ridge"></a>
[$\Uparrow$ Índice](#indice)

=====================================  
Regularização L2 (Ridge Regression)  
=====================================

A regressão linear funciona bem quando temos muitos dados e pouco ruído.  
Mas quando:

- os dados são ruidosos  
- o número de features é grande  
- o número de exemplos é pequeno  
- as features são correlacionadas  
- $X^\top X$ é mal condicionado  

o modelo pode **superajustar** (overfitting) ou até mesmo ficar **instável**.

A solução é adicionar **regularização L2**, também chamada de **ridge regression**.

---

### O problema sem regularização

Sem regularização, a solução fechada é:

$$
\theta^* = (X^\top X)^{-1} X^\top y
$$

Mas se $X^\top X$ for:

- mal condicionado  
- quase singular  
- com colunas quase redundantes  

então:

- a inversão é instável  
- pequenos ruídos nos dados geram grandes mudanças em $\theta$  
- o modelo generaliza mal  

Precisamos estabilizar a solução.

---

### Regularização L2

Adicionamos um termo que penaliza pesos grandes:

$$J(\theta)= \frac{1}{2n}\|y - X\theta\|^2 + \frac{\lambda}{2}\|\theta\|^2$$

onde:

- $\lambda > 0$ controla a força da penalização  
- $\|\theta\|^2$ mede a complexidade do modelo  

---

### Solução fechada regularizada

A condição de otimalidade leva a:

$$
(X^\top X + \lambda I)\theta = X^\top y
$$

A solução é:

$$
\theta^* = (X^\top X + \lambda I)^{-1} X^\top y
$$

Essa é a **ridge regression**.

---

### Intuição geométrica

Regularização L2 “puxa” o vetor $\theta$ para perto da origem.

Quando $\lambda$ é grande:

$$
\theta \to 0
$$

O hiperplano se torna mais simples:

- menos inclinado  
- menos sensível ao ruído  
- mais estável  

Regularização reduz **variância**, mas aumenta **viés**.

O objetivo é encontrar o equilíbrio ideal.

---

### Conexão com viés–variância

Regularização L2 controla o trade‑off:

- **$\lambda$ pequeno** → modelo complexo → baixa viés, alta variância → risco de overfitting  
- **$\lambda$ grande** → modelo simples → alta viés, baixa variância → risco de underfitting  

A escolha de $\lambda$ determina a capacidade de generalização.

---

### Conexão com a Unit 1

Regularização L2 aparece em:

- SVM (margem máxima)  
- Pegasos (SGD regularizado)  
- PA (passo proximal)  
- regressão linear (ridge)  
- matrix completion (Lecture 7)  

Ela é o mecanismo universal para:

- controlar complexidade  
- evitar overfitting  
- estabilizar soluções  
- melhorar generalização  

---

### Conexão com matrix completion (Lecture 7)

Ridge regression é usada em:

- alternating minimization  
- regressão 1D  
- estimativa de fatores latentes  
- modelos low rank  

Ela é a base matemática dos sistemas de recomendação.

---

### Pensamento aplicado

Regularização L2 é essencial porque:

1. **Evita overfitting.**  
2. **Controla variância.**  
3. **Estabiliza a solução fechada.**  
4. **Melhora generalização.**  
5. **É usada em regressão, kernels e matrix completion.**

---

### Conclusão

Regularização L2 adiciona estabilidade e melhora a generalização da regressão linear.  
Ela transforma a solução fechada em um método robusto e confiável, mesmo em dados ruidosos ou de alta dimensão.



<br><br>



<a id="gen_reg"></a>
[$\Uparrow$ Índice](#indice)

=====================================  
Generalização na Regressão  
=====================================

O objetivo da regressão linear não é apenas ajustar bem os dados de treinamento.  
O objetivo real é **prever corretamente valores novos**, nunca vistos.

Assim como na Unit 1, a diferença entre:

- **risco empírico** (erro no treino)  
- **risco verdadeiro** (erro esperado em novos dados)  

é o que define a capacidade de generalização.

---

### Risco verdadeiro

O risco verdadeiro é:

$$R(\theta) = \mathbb{E}\left[\frac{1}{2}(y - \theta^\top x)^2\right]$$

Mas não conhecemos a distribuição real dos dados.  
Por isso, usamos o risco empírico como aproximação:

$$R_n(\theta)= \frac{1}{2n} \sum_{t=1}^n (y^{(t)} - \theta^\top x^{(t)})^2$$

Minimizar $R_n$ **não garante** minimizar $R$.

Precisamos entender o que afeta essa diferença.

---

### Dois tipos de erro

Lecture 5 explica que existem dois tipos de erro fundamentais:

---

#### 1. Erro estrutural (bias)

O erro estrutural ocorre quando:

- a relação entre $x$ e $y$ **não é linear**  
- o modelo linear é **incapaz** de capturar a verdadeira função  
- mesmo com infinitos dados, o erro não desaparece  

Exemplo:

- tentar prever temperatura usando apenas uma reta  
- tentar prever preço usando apenas uma combinação linear simples  

Esse erro é causado pela **simplicidade do modelo**.

---

#### 2. Erro de estimação (variance)

O erro de estimação ocorre quando:

- temos poucos dados  
- temos muitas features  
- os dados são ruidosos  
- o modelo se ajusta demais ao treino  

Mesmo que a relação seja linear, o modelo não consegue estimar $\theta$ corretamente.

Esse erro é causado pela **falta de dados ou excesso de complexidade**.

---

### O trade‑off fundamental

Esses dois erros puxam em direções opostas:

- modelos simples → alto bias, baixo variance  
- modelos complexos → baixo bias, alto variance  

Generalização depende de equilibrar esses dois fatores.

---

### Papel da regularização

Regularização L2 reduz variance:

$$J(\theta)= R_n(\theta) + \frac{\lambda}{2}\|\theta\|^2$$

Ela:

- estabiliza a solução  
- reduz sensibilidade ao ruído  
- melhora generalização  
- evita overfitting  
- controla complexidade  

Mas aumenta bias.

O valor ideal de $\lambda$ equilibra bias e variance.

---

### Conexão com kernels (Lecture 6)

Kernel methods ampliam o espaço de features:

- aumentam capacidade  
- reduzem bias  
- aumentam variance  

Por isso, **regularização é essencial** em modelos kernelizados.

---

### Conexão com matrix completion (Lecture 7)

Modelos low rank reduzem variance:

- impõem estrutura  
- reduzem dimensionalidade  
- evitam overfitting  
- melhoram generalização em sistemas de recomendação  

Ridge regression aparece novamente como mecanismo de estabilidade.

---

### Intuição geométrica

Generalização depende de como o hiperplano se comporta **fora dos pontos vistos**.

- hiperplanos muito inclinados → variância alta  
- hiperplanos muito planos → viés alto  

Regularização controla essa inclinação.

---

### Pensamento aplicado

Generalização na regressão é essencial porque:

1. **Define desempenho real do modelo.**  
2. **Explica por que regularização é necessária.**  
3. **Conecta regressão com kernels e low rank.**  
4. **Mostra que minimizar erro no treino não é suficiente.**  
5. **É o coração de todos os modelos supervisionados.**

---

### Conclusão

Generalização na regressão depende do equilíbrio entre erro estrutural e erro de estimação.  
Regularização controla esse equilíbrio, garantindo estabilidade e bom desempenho em dados novos.




<br><br>



<a id="conexao_kernel_matrix"></a>
[$\Uparrow$ Índice](#indice)

=====================================  
Conexão com Kernel Methods e Matrix Completion  
=====================================

A regressão linear não é apenas um modelo simples para prever valores contínuos.  
Ela é a **base matemática** que sustenta as duas próximas partes da Unit 2:

- **Kernel Methods (Lecture 6)**  
- **Matrix Completion & Low Rank Models (Lecture 7)**  

Este item mostra como tudo se conecta.

---

## 1. Conexão com Kernel Methods (Lecture 6)

Kernel Methods transformam modelos lineares em modelos **não lineares**.  
Mas, no fundo, tudo ainda é regressão linear — só que em um espaço transformado.

### A ponte é o produto interno

A regressão linear usa:

$$
\theta^\top x.
$$

Kernel Methods substituem isso por:

$$K(x_i, x_j) = \phi(x_i)^\top \phi(x_j).$$

Ou seja:

- regressão linear → produto interno no espaço original  
- kernel methods → produto interno em um espaço transformado  

A matemática é a mesma.  
A diferença é apenas **onde** o produto interno é calculado.

### A solução dual da SVM é regressão linear kernelizada

SVM dual:

$$f(x) = \sum_i \alpha_i y_i K(x_i, x).$$

Perceptron kernelizado:

$$f(x) = \sum_i \alpha_i y_i K(x_i, x).$$

Softmax kernelizado:

$$h_j(x) = \sum_i \alpha_j^{(i)} K(x^{(i)}, x).$$

Todos são **regressões lineares no espaço kernelizado**.

---

## 2. Conexão com Matrix Completion (Lecture 7)

Matrix Completion parece um problema totalmente diferente:

- prever notas de filmes  
- preencher valores faltantes  
- aprender fatores latentes  
- decompor matrizes em baixa dimensão  

Mas a base matemática é **regressão linear com regularização**.

### Alternating Minimization é regressão linear repetida

No modelo low rank:

$$R_{ij} \approx u_i^\top v_j.$$

Fixamos $u_i$ e resolvemos para $v_j$:

→ isso é **regressão linear 1D** com regularização L2.

Fixamos $v_j$ e resolvemos para $u_i$:

→ isso é **regressão linear 1D** com regularização L2.

O algoritmo inteiro é:

> regressão linear + regressão linear + regressão linear + …

### Ridge Regression aparece novamente

A solução para cada fator é:

$$v_j = (U^\top U + \lambda I)^{-1} U^\top r_j.$$

Isso é exatamente a **equação normal regularizada** da Lecture 5.

---

## 3. A visão unificada da Unit 2

A Unit 2 parece ter três temas diferentes:

1. regressão linear  
2. kernels  
3. matrix completion  

Mas, na verdade, todos são variações da mesma ideia:

> **minimizar uma perda quadrática usando produtos internos e regularização.**

- regressão linear → produto interno simples  
- kernel methods → produto interno kernelizado  
- matrix completion → produto interno entre fatores latentes  

A matemática é a mesma.  
A aplicação muda.

---

## 4. Conclusão

A Lecture 5 não é apenas sobre regressão linear.  
Ela é o **fundamento matemático** que permite:

- criar modelos não lineares via kernels  
- construir sistemas de recomendação via low rank  
- resolver problemas quadráticos com solução fechada  
- aplicar regularização para generalização  

A Unit 2 inteira é construída sobre as ideias introduzidas aqui.





<br><br>




<a id="kernel"></a>
[$\Uparrow$ Índice](#indice)


=============================  
Kernel Trick  
=============================

O **kernel trick** é uma das ideias mais elegantes de toda a Unit 2.  
Ele transforma algoritmos lineares em algoritmos **não lineares**, sem nunca calcular explicitamente o mapeamento para o espaço de alta dimensão.

---

## 1. O problema: modelos lineares são limitados

Modelos como:

- perceptron  
- SVM linear  
- regressão logística  
- softmax  

usam funções da forma:

$$
f(x) = w^\top x + b.
$$

Isso significa que a fronteira de decisão é sempre um **hiperplano**.

Mas muitos problemas reais exigem fronteiras **curvas**, **polinomiais**, **radiais**, **complexas**.

---

## 2. A solução clássica (explícita): feature maps

Podemos mapear os dados para um espaço de features mais rico:

$$
x \mapsto \phi(x).
$$

Exemplo: features cúbicas explícitas (como você implementou no Project 2):

$$
\phi(x) = (x_1^3,\; x_1^2 x_2,\; x_1 x_2^2,\; x_2^3,\; \dots).
$$

Então o modelo se torna:

$$
f(x) = w^\top \phi(x).
$$

Isso cria fronteiras **não lineares** no espaço original.

### Problema:
O número de features explode:

- polinômio de grau 3 → centenas de termos  
- polinômio de grau 10 → milhares  
- RBF → infinitas dimensões  

Calcular $\phi(x)$ explicitamente se torna inviável.

---

## 3. A sacada: muitos algoritmos dependem apenas de produtos internos

Perceptron dual:

$$
w = \sum_i \alpha_i y_i x_i.
$$

SVM dual:

$$
f(x) = \sum_i \alpha_i y_i (x_i^\top x).
$$

Softmax kernelizado:

$$
h_j(x) = \sum_i \alpha_j^{(i)} K(x^{(i)}, x).
$$

Todos dependem apenas de:

$$
\phi(x_i)^\top \phi(x_j).
$$

Ou seja: **não precisamos de $\phi(x)$, apenas do produto interno após o mapeamento**.

---

## 4. Kernel Trick

Se existir uma função $K(x,y)$ tal que:

$$
K(x,y) = \phi(x)^\top \phi(y),
$$

então podemos substituir:

$$
\phi(x_i)^\top \phi(x_j)
\quad\longrightarrow\quad
K(x_i, x_j),
$$

sem nunca calcular $\phi(x)$.

Isso é o **kernel trick**:

> **Usar um kernel $K(x,y)$ no lugar do produto interno no espaço de features, evitando calcular $\phi(x)$ explicitamente.**

---

## 5. Exemplos de kernels e seus feature maps implícitos

### Kernel linear
$$
K(x,y) = x^\top y.
$$

Feature map:
$$
\phi(x) = x.
$$

---

### Kernel polinomial
$$
K(x,y) = (x^\top y + c)^p.
$$

Feature map:
- contém todos os termos polinomiais até grau $p$  
- número de features cresce combinatorialmente  
- **não calculamos $\phi(x)$**  

---

### Kernel cúbico
$$
K(x,y) = (x^\top y + 1)^3.
$$

Feature map:
- exatamente equivalente às suas `cubic_features`  
- mas sem custo explícito  

---

### Kernel RBF (Gaussiano)
$$
K(x,y) = \exp(-\gamma \|x - y\|^2).
$$

Feature map:
- **infinito-dimensional**  
- impossível de calcular explicitamente  
- kernel trick torna possível  

---

## 6. Interpretação geométrica

O kernel trick **distorce o espaço**:

- kernel polinomial → dobra o espaço em superfícies polinomiais  
- kernel cúbico → cria curvas cúbicas  
- kernel RBF → cria “bolhas” radiais  
- kernel misto → combina múltiplas geometrias  

No espaço transformado, o modelo continua sendo **linear**:

$$
f(x) = w^\top \phi(x).
$$

Mas no espaço original, a fronteira é **não linear**.

---

## 7. Conexão com SVM

A SVM dual usa:

$$
f(x) = \sum_i \alpha_i y_i K(x_i, x).
$$

O kernel trick permite:

- SVM polinomial  
- SVM cúbico  
- SVM RBF  
- SVM com kernels customizados  

Sem nunca calcular $\phi(x)$.

---

## 8. Conexão com Perceptron Kernel

Perceptron kernelizado:

$$
f(x) = \sum_i \alpha_i y_i K(x_i, x).
$$

Mesma estrutura da SVM dual.

---

## 9. Conexão com Softmax Kernelizado

Softmax kernelizado:

$$
h_j(x) = \sum_i \alpha_j^{(i)} K(x^{(i)}, x).
$$

Mesma estrutura, mas com probabilidades.

---

## 10. Conclusão

O kernel trick permite:

- trabalhar com espaços gigantes ou infinitos  
- sem custo computacional explosivo  
- sem calcular feature maps  
- usando apenas produtos internos kernelizados  

Ele é a ponte entre:

- modelos lineares  
- modelos não lineares  
- SVM  
- perceptron  
- softmax  
- matrix completion (via kernels estruturados)

É uma das ideias mais poderosas de toda a Unit 2.






<br><br>





<a id="dual"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Kernel Dual  
=============================

A formulação **dual** da SVM é onde o kernel realmente aparece.  
Ela transforma a SVM linear em uma SVM **não linear**, sem nunca calcular $\phi(x)$ explicitamente.

---

## 1. Por que existe o dual?

A formulação primal da SVM é:

$$
\min_{w,b}
\quad
\frac{\lambda}{2}\|w\|^2
+
\frac{1}{n}\sum_{i=1}^n
\max(0,\, 1 - y_i(w^\top x_i + b)).
$$

Essa forma é linear em $x$, portanto **não permite kernels**.

Para introduzir kernels, precisamos reescrever o problema em termos de **produtos internos**.  
Isso só acontece no **dual**.

---

## 2. A ponte: representação dual de $w$

A solução ótima $w$ pode ser escrita como:

$$
w = \sum_{i=1}^n \alpha_i y_i x_i.
$$

Essa é a chave:

> **O vetor de pesos é uma combinação linear dos pontos de treinamento.**

Isso significa que a decisão pode ser escrita como:

$$
w^\top x = \sum_{i=1}^n \alpha_i y_i (x_i^\top x).
$$

Agora tudo depende apenas de **produtos internos**.

---

## 3. A formulação dual da SVM linear

O dual é:

$$
\max_{\alpha}
\sum_{i=1}^n \alpha_i
-
\frac{1}{2}
\sum_{i=1}^n \sum_{j=1}^n
\alpha_i \alpha_j y_i y_j (x_i^\top x_j)
$$

sujeito a:

$$
0 \le \alpha_i \le C.
$$

Observe:

- não aparece $w$  
- não aparece $\phi(x)$  
- só aparecem **produtos internos**  

Isso é exatamente o que precisamos para aplicar o kernel trick.

---

## 4. Kernel Trick no dual

Substituímos o produto interno por um kernel:

$$
x_i^\top x_j
\quad\longrightarrow\quad
K(x_i, x_j).
$$

O dual kernelizado se torna:

$$
\max_{\alpha}
\sum_{i=1}^n \alpha_i
-
\frac{1}{2}
\sum_{i=1}^n \sum_{j=1}^n
\alpha_i \alpha_j y_i y_j K(x_i, x_j).
$$

Essa é a **SVM kernelizada**.

---

## 5. A solução final da SVM kernelizada

Depois de resolver o dual, a decisão é:

$$
f(x) = \sum_{i=1}^n \alpha_i y_i K(x_i, x) + b.
$$

Interpretação:

- não existe $w$ explícito  
- não existe $\phi(x)$ explícito  
- tudo depende apenas de $K(x_i, x)$  
- apenas **support vectors** têm $\alpha_i > 0$  

Isso explica o nome “Support Vector Machine”.

---

## 6. Geometria do dual

A SVM kernelizada constrói fronteiras de decisão:

- polinomiais (kernel polinomial)  
- cúbicas (kernel cúbico)  
- radiais (kernel RBF)  
- arbitrárias (kernels customizados)  

No espaço transformado, a fronteira é **linear**:

$$
f(x) = w^\top \phi(x).
$$

No espaço original, ela é **não linear**.

---

## 7. Conexão com Kernel Trick

O dual é o lugar onde o kernel trick acontece:

- primal → depende de $w$  
- dual → depende de $x_i^\top x_j$  
- kernel trick → substitui $x_i^\top x_j$ por $K(x_i, x_j)$  

Por isso:

> **Não existe SVM kernelizada sem o dual.**

---

## 8. Conexão com Perceptron Kernel

Perceptron kernelizado:

$$
f(x) = \sum_i \alpha_i y_i K(x_i, x).
$$

Mesma estrutura da SVM dual.

---

## 9. Conexão com Softmax Kernelizado

Softmax kernelizado:

$$
h_j(x) = \sum_i \alpha_j^{(i)} K(x^{(i)}, x).
$$

Mesma estrutura, mas com probabilidades.

---

## 10. Conexão com cubic_features

Você implementou explicitamente:

$$
\phi(x) = \text{cubic\_features}(x).
$$

O kernel cúbico:

$$
K(x,y) = (x^\top y + 1)^3
$$

produz **exatamente o mesmo espaço**, mas sem custo explícito.

---

## 11. Conexão com kernel RBF

O kernel RBF:

$$
K(x,y) = \exp(-\gamma \|x - y\|^2)
$$

corresponde a um espaço **infinito-dimensional**.

O dual permite usar esse espaço sem nunca calcular $\phi(x)$.

---

## 12. Conclusão

O Kernel Dual é:

- a ponte entre primal e kernel trick  
- o lugar onde kernels entram na SVM  
- a razão pela qual SVM depende apenas de support vectors  
- o mecanismo que permite espaços infinitos  
- a base para perceptron kernelizado e softmax kernelizado  

Sem o dual, kernels não existiriam.






<br><br>


<a id="regraskernel"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Regras para uma Função ser Kernel  
=

Uma função $K(x,x')$ é um **kernel válido** quando ela pode ser interpretada como um **produto interno** em algum espaço de features (possivelmente de dimensão muito alta ou até infinita):

$$
K(x,x') = \phi(x)^\top \phi(x')
$$

Esse espaço pode ser explícito ou implícito.  
Quando é implícito, usamos o **kernel trick** para evitar calcular $\phi(x)$ diretamente.

Para verificar se uma função é kernel, usamos as regras abaixo.

---

## 1. Kernel deve ser PSD (positive semidefinite)

Uma função $K$ é kernel se, para qualquer conjunto de pontos $\{x_1, \dots, x_n\}$, a matriz:

$$
K_{ij} = K(x_i, x_j)
$$

é **semidefinida positiva**, isto é:

$$
\sum_{i,j} c_i c_j K(x_i, x_j) \ge 0
\quad\text{para qualquer vetor } c.
$$

Essa é a definição formal.

---

## 2. Soma de kernels → kernel

Se $K_1$ e $K_2$ são kernels, então:

$$
K(x,x') = K_1(x,x') + K_2(x,x')
$$

também é kernel.

Exemplo:

$$
1 + x^\top x'
$$

é kernel porque:

- $1$ é kernel  
- $x^\top x'$ é kernel  

---

## 3. Produto de kernels → kernel

Se $K_1$ e $K_2$ são kernels, então:

$$
K(x,x') = K_1(x,x') \cdot K_2(x,x')
$$

também é kernel.

Exemplo:

$$
(1 + x^\top x')^2
$$

é kernel porque é o produto de dois kernels.

---

## 4. Multiplicação por função positiva → kernel

Se $K$ é kernel e $f(x) > 0$, então:

$$
\tilde{K}(x,x') = f(x)\,K(x,x')\,f(x')
$$

é kernel.

Essa regra é usada para construir kernels adaptativos.

---

## 5. Composição com funções monotônicas → kernel

Se $K$ é kernel e $g$ é uma função que preserva PSD (como exponencial, polinômios com coeficientes positivos), então:

$$
g(K(x,x'))
$$

é kernel.

Exemplo:

$$
e^{x^\top x'}
$$

é kernel.

---

## 6. Limites de kernels → kernel

Se $K_n$ é uma sequência de kernels e:

$$
K(x,x') = \lim_{n\to\infty} K_n(x,x')
$$

então $K$ é kernel.

Isso permite kernels infinitos, como o RBF.

---

## 7. Exemplos clássicos de kernels

### Kernel linear
$$
K(x,x') = x^\top x'
$$

### Kernel polinomial
$$
K(x,x') = (1 + x^\top x')^d
$$

### Kernel RBF (Gaussiano)
$$
K(x,x') = e^{-\frac{1}{2}\|x - x'\|^2}
$$

### Kernel de Brownian motion
$$
K(x,x') = \min(x, x')
$$

---

## 8. Resumo

Uma função é kernel se:

- é PSD  
- ou pode ser escrita como $\phi(x)^\top \phi(x')$  
- ou é construída por regras válidas:  
  - soma  
  - produto  
  - multiplicação por função positiva  
  - composição monotônica  
  - limite de kernels  

Essas regras permitem criar kernels complexos a partir de kernels simples, e são a base da Lecture 6.


<br><br>


<a id="exemplos_kernel"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Kernels Examples  
=============================

Aqui reunimos os kernels mais usados em aprendizado de máquina.  
Cada kernel define uma geometria diferente no espaço transformado, permitindo que modelos lineares se tornem **não lineares** no espaço original.

---

## 1. Kernel Linear

$$
K(x,y) = x^\top y.
$$

### Intuição  
- mede similaridade via produto interno  
- não altera o espaço  
- equivalente ao modelo linear original  

### Uso  
- SVM linear  
- perceptron linear  
- regressão logística linear  

---

## 2. Kernel Polinomial

$$
K(x,y) = (x^\top y + c)^p.
$$

### Intuição  
- cria superfícies polinomiais  
- inclui termos quadráticos, cúbicos, etc.  
- controla complexidade via $p$  

### Casos especiais  
- $p=2$ → kernel quadrático  
- $p=3$ → kernel cúbico  

### Uso  
- SVM polinomial  
- perceptron polinomial  
- softmax kernelizado polinomial  

---

## 3. Kernel Cúbico (caso especial do polinomial)

$$
K(x,y) = (x^\top y + 1)^3.
$$

### Intuição  
- cria superfícies cúbicas  
- muito usado em MNIST  
- aproxima bem fronteiras curvas  

### Uso  
- SVM cúbico  
- cubic features (equivalente explícito)  

---

## 4. Kernel RBF (Gaussiano)

$$
K(x,y) = \exp(-\gamma \|x - y\|^2).
$$

### Intuição  
- cria “bolhas” radiais  
- altamente não linear  
- espaço de features **infinito**  
- controla suavidade via $\gamma$  

### Uso  
- SVM RBF  
- perceptron RBF  
- softmax kernelizado RBF  

---

## 5. Kernel Laplaciano

$$
K(x,y) = \exp(-\gamma \|x - y\|_1).
$$

### Intuição  
- similar ao RBF, mas usa norma L1  
- mais robusto a outliers  

### Uso  
- SVM para dados com ruído  
- problemas de visão computacional  

---

## 6. Kernel Sigmoide

$$
K(x,y) = \tanh(\alpha x^\top y + c).
$$

### Intuição  
- inspirado em redes neurais  
- não é PSD para todos os parâmetros  
- precisa respeitar Kernel Rules  

### Uso  
- modelos que imitam MLP  
- experimentação  

---

## 7. Kernel de Interseção de Histogramas

$$
K(x,y) = \sum_i \min(x_i, y_i).
$$

### Intuição  
- mede sobreposição entre histogramas  
- muito usado em visão computacional  

### Uso  
- SVM para imagens  
- bag-of-words  

---

## 8. Kernel Chi-Square

$$
K(x,y) = \sum_i \frac{2 x_i y_i}{x_i + y_i}.
$$

### Intuição  
- mede similaridade entre distribuições  
- muito usado em visão computacional  

### Uso  
- classificação de imagens  
- reconhecimento de padrões  

---

## 9. Kernel de Strings (N-gram Kernel)

$$
K(s,t) = \text{número de n-gramas compartilhados}.
$$

### Intuição  
- mede similaridade textual  
- usado em NLP  

### Uso  
- SVM para texto  
- classificação de documentos  

---

## 10. Kernel de Grafos

$$
K(G_1, G_2) = \text{similaridade estrutural entre grafos}.
$$

### Intuição  
- mede padrões estruturais  
- usado em química, biologia, redes sociais  

### Uso  
- previsão de propriedades moleculares  
- análise de redes  

---

## 11. Kernel Customizado (via Kernel Rules)

Se $K_1$ e $K_2$ são kernels válidos:

- soma:  
  $$
  K = K_1 + K_2
  $$
- produto:  
  $$
  K = K_1 K_2
  $$
- composição:  
  $$
  K = \exp(K_1)
  $$

Isso permite criar kernels sob medida para qualquer problema.

---

## 12. Conclusão

Os kernels acima permitem transformar modelos lineares em modelos altamente não lineares:

- polinômios  
- cúbicos  
- radiais  
- histogramas  
- distribuições  
- strings  
- grafos  

Cada kernel define uma geometria diferente, e a escolha correta pode melhorar drasticamente o desempenho do modelo.




<br><br>




<a id="svm_kernelizada"></a>
[$\Uparrow$ Índice](#indice)


=============================  
SVM Kernelizada  
=============================

A SVM kernelizada é a aplicação mais famosa do **kernel trick**.  
Ela transforma a SVM linear em um classificador **não linear**, sem nunca calcular $\phi(x)$.

---

## 1. SVM linear (revisão)

A SVM linear aprende um vetor de pesos $\theta$ e um bias $b$:

$$
f(x) = \theta^\top x + b.
$$

A decisão é:

- $f(x) > 0$ → classe positiva  
- $f(x) < 0$ → classe negativa  

A função objetivo primal é:

$$
\min_\theta \;\; 
\frac{\lambda}{2}\|\theta\|^2
+ \sum_{i=1}^n \max(0,\, 1 - y_i\,\theta^\top x_i).
$$

Essa forma é **linear** em $x$.

---

## 2. A formulação dual (onde o kernel aparece)

O dual da SVM linear é:

$$
\max_{\alpha}
\sum_{i=1}^n \alpha_i
- \frac{1}{2}
\sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^\top x_j)
$$

com restrições:

$$
0 \le \alpha_i \le C.
$$

Aqui aparece a chave:

> **A solução depende apenas de produtos internos entre pontos de treinamento.**

---

## 3. Kernel Trick

Substituímos o produto interno por um kernel:

$$
x_i^\top x_j
\quad\longrightarrow\quad
K(x_i, x_j).
$$

O dual kernelizado se torna:

$$
\max_{\alpha}
\sum_{i=1}^n \alpha_i
- \frac{1}{2}
\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j).
$$

Essa é a SVM kernelizada.

---

## 4. A solução final

Depois de resolver o dual, a decisão é:

$$
f(x) = \sum_{i=1}^n \alpha_i y_i K(x_i, x) + b.
$$

Observe:

- não existe $\theta$ explícito  
- não existe $\phi(x)$ explícito  
- tudo depende apenas de $K(x_i, x)$  
- apenas os **support vectors** têm $\alpha_i > 0$

---

## 5. Exemplos de kernels usados na SVM

### Kernel linear

$$
K(x,y) = x^\top y.
$$

### Kernel polinomial

$$
K(x,y) = (x^\top y + c)^p.
$$

### Kernel cúbico

$$
K(x,y) = (x^\top y + 1)^3.
$$

### Kernel RBF (Gaussiano)

$$
K(x,y) = \exp(-\gamma \|x - y\|^2).
$$

Esse kernel corresponde a um espaço de features **infinito**.

---

## 6. Interpretação geométrica

A SVM kernelizada continua sendo linear no espaço transformado:

$$
f(x) = \theta^\top \phi(x),
$$

mas esse espaço pode ser:

- muito grande  
- ou até infinito  

O kernel trick evita calcular $\phi(x)$ explicitamente.

No espaço original, a fronteira se torna:

- polinomial (kernel polinomial)  
- cúbica (kernel cúbico)  
- altamente não linear (kernel RBF)  

---

## 7. Relação com Perceptron Kernel e Softmax Kernelizado

### Perceptron Kernel

$$
f(x) = \sum_i \alpha_i y_i K(x_i, x).
$$

### SVM Kernelizada

$$
f(x) = \sum_i \alpha_i y_i K(x_i, x) + b.
$$

### Softmax Kernelizado

$$
h_j(x) = \sum_i \alpha_j^{(i)} K(x^{(i)}, x).
$$

A estrutura é sempre a mesma:

> **combinação linear de kernels entre o novo ponto e os pontos de treinamento.**

---

## 8. Conclusão

A SVM kernelizada é:

- linear no espaço transformado  
- não linear no espaço original  
- eficiente  
- poderosa  
- baseada apenas em kernels  
- dependente apenas dos support vectors  

Ela é o exemplo mais clássico e elegante do kernel trick.




<br><br>


<a id="perceptron_kernel"></a>
[$\Uparrow$ Índice](#indice)


=============================  
Perceptron Kernel  
=============================

O perceptron kernelizado é a versão **online** do kernel trick.  
Ele transforma o perceptron linear em um classificador **não linear**, sem nunca calcular $\phi(x)$.

---

## 1. Perceptron clássico (revisão)

O perceptron linear atualiza:

$$
\theta \leftarrow \theta + y_i x_i
$$

quando erra no exemplo $(x_i, y_i)$.

Após várias atualizações:

$$
\theta = \sum_{i=1}^n \alpha_i y_i x_i,
$$

onde $\alpha_i$ é o número de erros no exemplo $i$.

Essa é a **representação dual** do perceptron.

---

## 2. Perceptron no espaço de features

Se quisermos tornar o perceptron não linear, aplicamos um mapeamento:

$$
x \mapsto \phi(x).
$$

A atualização se torna:

$$
\theta \leftarrow \theta + y_i\,\phi(x_i).
$$

E a forma dual:

$$
\theta = \sum_{i=1}^n \alpha_i y_i \phi(x_i).
$$

---

## 3. Kernel Trick

A predição é:

$$
f(x) = \theta^\top \phi(x)
= \sum_{i=1}^n \alpha_i y_i \big[\phi(x_i)^\top \phi(x)\big].
$$

Substituímos o produto interno por um kernel:

$$
K(x_i, x) = \phi(x_i)^\top \phi(x).
$$

Logo:

$$
f(x) = \sum_{i=1}^n \alpha_i y_i K(x_i, x).
$$

Essa é a forma kernelizada do perceptron.

---

## 4. Atualização kernelizada

Quando o perceptron kernelizado erra no ponto $(x_t, y_t)$:

- ele **não atualiza $\theta$ diretamente**  
- ele **incrementa $\alpha_t$**  

$$
\alpha_t \leftarrow \alpha_t + 1.
$$

A predição futura passa a incluir esse novo termo:

$$
f(x) = \sum_{i=1}^n \alpha_i y_i K(x_i, x).
$$

---

## 5. Intuição geométrica

O perceptron kernelizado:

- adiciona uma “bolha” (kernel RBF)  
- ou uma “superfície cúbica” (kernel polinomial)  
- ou uma “superfície polinomial” (kernel polinomial geral)  

cada vez que erra.

Com o tempo, a fronteira de decisão se torna **não linear**, moldada pelos erros.

---

## 6. Comparação com SVM kernelizada

### Perceptron Kernel

- atualiza apenas quando erra  
- passo fixo  
- online  
- não usa margem  
- não usa regularização  
- pode oscilar  

### SVM Kernelizada

- otimização convexa  
- usa margem  
- usa regularização  
- depende apenas dos support vectors  
- mais estável  
- melhor generalização  

Ambos usam:

$$
f(x) = \sum_i \alpha_i y_i K(x_i, x).
$$

Mas a SVM aprende $\alpha_i$ de forma **ótima**, enquanto o perceptron aprende $\alpha_i$ de forma **incremental**.

---

## 7. Exemplos de kernels usados no perceptron

### Kernel linear

$$
K(x,y) = x^\top y.
$$

### Kernel polinomial

$$
K(x,y) = (x^\top y + c)^p.
$$

### Kernel cúbico

$$
K(x,y) = (x^\top y + 1)^3.
$$

### Kernel RBF

$$
K(x,y) = \exp(-\gamma \|x - y\|^2).
$$

---

## 8. Conclusão

O perceptron kernelizado é:

- simples  
- online  
- não linear  
- eficiente  
- baseado apenas em kernels  
- precursor da SVM kernelizada  

Ele mostra que o kernel trick funciona mesmo em algoritmos extremamente simples.




<br><br>

<a id="kernel_softmax"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Kernel Softmax  
=============================

A regressão softmax é um modelo **linear** no espaço de features:

$$
h_j(x) = \theta_j^\top x
$$

onde cada classe $j$ possui um vetor de pesos $\theta_j$.

Para torná‑la **não linear**, podemos aplicar o mesmo truque usado no perceptron e na SVM:  
representar $\theta_j$ como uma combinação linear de vetores de treinamento.

---

## 1. Representação dual da regressão softmax

Assim como no perceptron kernelizado:

$$
\theta = \sum_i \alpha^{(i)} y^{(i)} x^{(i)},
$$

na regressão softmax kernelizada cada classe $j$ possui:

$$
\theta_j = \sum_{i=1}^n \alpha_j^{(i)} \phi(x^{(i)}),
$$

onde:

- $\phi(x)$ é o mapeamento para o espaço de alta dimensão  
- $\alpha_j^{(i)}$ são coeficientes aprendidos durante o treinamento  
- $x^{(i)}$ são os pontos de treinamento

---

## 2. A predição kernelizada

A predição softmax tradicional usa:

$$
h_j(x) = \theta_j^\top \phi(x).
$$

Substituindo a forma dual:

$$
h_j(x)
= \left( \sum_{i=1}^n \alpha_j^{(i)} \phi(x^{(i)}) \right)^\top \phi(x)
= \sum_{i=1}^n \alpha_j^{(i)} \big[ \phi(x^{(i)})^\top \phi(x) \big].
$$

Agora aplicamos o **kernel trick**:

$$
K(x^{(i)}, x) = \phi(x^{(i)})^\top \phi(x).
$$

Logo:

$$
h_j(x) = \sum_{i=1}^n \alpha_j^{(i)} K(x^{(i)}, x).
$$

Essa é a forma kernelizada da regressão softmax.

---

## 3. A função softmax completa

A probabilidade da classe $j$ é:

$$
P(y=j \mid x)
=
\frac{
\exp\left( \frac{1}{\tau}
\sum_{i=1}^n \alpha_j^{(i)} K(x^{(i)}, x)
\right)
}{
\sum_{k=1}^K
\exp\left( \frac{1}{\tau}
\sum_{i=1}^n \alpha_k^{(i)} K(x^{(i)}, x)
\right)
}.
$$

Observe que:

- **não precisamos calcular $\phi(x)$**  
- **não precisamos gerar features polinomiais ou RBF**  
- **tudo depende apenas de $K(x^{(i)}, x)$**

---

## 4. Conexão com kernels específicos

### Kernel polinomial

$$
K(x,y) = (x^\top y + c)^p.
$$

### Kernel cúbico

$$
K(x,y) = (x^\top y + 1)^3.
$$

### Kernel RBF

$$
K(x,y) = \exp(-\gamma \|x - y\|^2).
$$

Todos podem ser usados dentro da regressão softmax kernelizada.

---

## 5. Interpretação geométrica

A regressão softmax kernelizada:

- continua sendo **linear** no espaço transformado  
- mas esse espaço pode ser **gigante** ou **infinito**  
- e o kernel trick evita calcular esse espaço explicitamente

Assim como a SVM kernelizada, o softmax kernelizado aprende fronteiras **não lineares** no espaço original.

---

## 6. Relação com SVM kernelizada

A SVM kernelizada usa:

$$
f(x) = \sum_i \alpha_i y_i K(x^{(i)}, x).
$$

A regressão softmax kernelizada usa:

$$
h_j(x) = \sum_i \alpha_j^{(i)} K(x^{(i)}, x).
$$

A diferença é:

- SVM → decisão por margem  
- Softmax → decisão probabilística

Mas **ambas** dependem apenas do kernel.

---

## 7. Conclusão

A regressão softmax kernelizada é:

> **um modelo linear no espaço de features, mas não linear no espaço original, graças ao kernel trick.**

Ela generaliza:

- perceptron kernelizado  
- SVM kernelizada  
- cubic features  
- polynomial kernel  
- RBF kernel  

e fecha o ciclo dos modelos lineares → kernelizados.




<br><br>




<a id="matrix"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Matrix Completion  
=============================

Matrix Completion é o problema de **preencher uma matriz parcialmente observada**, assumindo que ela possui **estrutura de baixa dimensão**.  
É o fundamento matemático por trás de sistemas de recomendação (como Netflix, Amazon, Spotify).

---

## 1. O problema

Temos uma matriz $M \in \mathbb{R}^{n \times m}$, mas apenas alguns elementos são observados:

$$
(i,j) \in \Omega \subseteq \{1,\dots,n\} \times \{1,\dots,m\}.
$$

Queremos reconstruir toda a matriz:

$$
\hat{M}_{ij} \approx M_{ij}
\quad\text{para todos } i,j.
$$

Mas isso é impossível sem alguma hipótese estrutural.

---

## 2. A hipótese fundamental: baixa dimensão (low rank)

Assumimos que $M$ pode ser aproximada por uma matriz de **baixo rank**:

$$
M \approx U V^\top,
$$

onde:

- $U \in \mathbb{R}^{n \times k}$  
- $V \in \mathbb{R}^{m \times k}$  
- $k \ll \min(n,m)$  

Interpretando:

- cada linha de $U$ representa **fatores latentes de um usuário**  
- cada linha de $V$ representa **fatores latentes de um item**  

E a predição é:

$$
\hat{M}_{ij} = U_i^\top V_j.
$$

---

## 3. Intuição geométrica

A matriz original pode ser enorme (milhões × milhares), mas ela vive em um **subespaço de baixa dimensão**.

Exemplo:

- usuários têm preferências em poucos eixos (ação, comédia, drama…)  
- filmes têm características em poucos eixos (leve, pesado, romântico…)  

A matriz de notas é apenas o produto desses fatores.

---

## 4. O problema de otimização

Queremos encontrar $U$ e $V$ que melhor explicam os valores observados:

$$
\min_{U,V}
\sum_{(i,j)\in\Omega}
\left(M_{ij} - U_i^\top V_j\right)^2.
$$

Para evitar overfitting, adicionamos regularização:

$$
\min_{U,V}
\sum_{(i,j)\in\Omega}
\left(M_{ij} - U_i^\top V_j\right)^2
+
\lambda\left(\|U\|^2 + \|V\|^2\right).
$$

Essa é a **objective function completa** da matrix completion.

---

## 5. Alternating Minimization

O problema não é convexo em $(U,V)$ juntos, mas é convexo em cada um separadamente.

A solução clássica é:

1. Fixar $V$ e otimizar $U$  
2. Fixar $U$ e otimizar $V$  
3. Repetir até convergir

Cada passo é uma **ridge regression**:

### Atualizando $U_i$:

$$
U_i \leftarrow \arg\min_u
\sum_{j:(i,j)\in\Omega}
(M_{ij} - u^\top V_j)^2 + \lambda\|u\|^2.
$$

### Atualizando $V_j$:

$$
V_j \leftarrow \arg\min_v
\sum_{i:(i,j)\in\Omega}
(M_{ij} - U_i^\top v)^2 + \lambda\|v\|^2.
$$

Esse processo converge para uma solução de baixa dimensão.

---

## 6. Conexão com SVD

Se a matriz fosse totalmente observada, a solução ideal seria:

$$
M \approx U_k \Sigma_k V_k^\top,
$$

onde $U_k, \Sigma_k, V_k$ são as $k$ maiores componentes da SVD.

Matrix completion é uma versão **parcial** desse problema:

- não temos todos os valores  
- não podemos usar SVD diretamente  
- precisamos otimizar apenas sobre os elementos observados  

---

## 7. Conexão com recomendação (Netflix)

Cada usuário $i$ tem um vetor de fatores latentes $U_i$.  
Cada filme $j$ tem um vetor de fatores latentes $V_j$.

A predição da nota é:

$$
\hat{M}_{ij} = U_i^\top V_j.
$$

Isso permite:

- recomendar filmes  
- prever notas  
- identificar similaridade entre usuários  
- identificar similaridade entre itens  

Tudo baseado em **low rank**.

---

## 8. Conexão com independência entre entradas

O modelo ingênuo assume:

$$
M_{ij} = a_i + b_j,
$$

ou seja, **independência entre entradas**.

Matrix completion é a versão **sofisticada**:

$$
M_{ij} = U_i^\top V_j,
$$

que captura interações complexas entre usuários e itens.

---

## 9. Conexão com prior gaussiano

A regularização L2:

$$
\lambda(\|U\|^2 + \|V\|^2)
$$

equivale a assumir um **prior gaussiano** sobre $U$ e $V$:

$$
U_i \sim \mathcal{N}(0, \sigma^2 I),
\qquad
V_j \sim \mathcal{N}(0, \sigma^2 I).
$$

Isso estabiliza o treinamento e evita overfitting.

---

## 10. Conclusão

Matrix Completion é:

- reconstrução de matrizes parcialmente observadas  
- baseada em **low rank**  
- resolvida via **alternating minimization**  
- equivalente a múltiplas **ridge regressions**  
- conectada a **SVD**, **fatores latentes**, **prior gaussiano**  
- base de sistemas de recomendação modernos  

É uma das aplicações mais elegantes de otimização e álgebra linear na Unit 2.



<br><br>

<a id="ingenuo"></a>
[$\Uparrow$ Índice](#indice)


=============================  
Modelo Ingênuo (Independência entre Entradas)  
=  

Antes de impor qualquer estrutura de baixo posto, podemos tentar resolver o problema de matrix completion tratando **cada entrada da matriz X como independente** das demais.

O objetivo ingênuo é minimizar:

$$
J(X) =
\sum_{(a,i)\in D} \frac{(Y_{ai} - X_{ai})^2}{2}
+
\frac{\lambda}{2} \sum_{a,i} X_{ai}^2
$$

Aqui:

- o **primeiro termo** mede o erro quadrático apenas nas entradas observadas  
- o **segundo termo** é a regularização L2, que incentiva valores pequenos em $X_{ai}$  

Como **não há nenhuma relação entre diferentes entradas de X**, podemos minimizar cada $X_{ai}$ separadamente.

---

### Minimização para entradas observadas

Para um par $(a,i)\in D$, a função relevante é:

$$
\frac{(Y_{ai} - X_{ai})^2}{2} + \frac{\lambda}{2} X_{ai}^2
$$

Derivando em relação a $X_{ai}$:

$$
\frac{\partial}{\partial X_{ai}}
\left[
\frac{(Y_{ai} - X_{ai})^2}{2}
\right]
= X_{ai} - Y_{ai}
$$

e

$$
\frac{\partial}{\partial X_{ai}}
\left[
\frac{\lambda}{2} X_{ai}^2
\right]
= \lambda X_{ai}
$$

Somando:

$$
(X_{ai} - Y_{ai}) + \lambda X_{ai}
= (1+\lambda)X_{ai} - Y_{ai}
$$

Igualando a zero:

$$
(1+\lambda)X_{ai} - Y_{ai} = 0
$$

Logo:

$$
X_{ai} = \frac{Y_{ai}}{1+\lambda}
$$

---

### Minimização para entradas faltantes

Se $(a,i)\notin D$, o termo de erro quadrático não aparece.  
A função é apenas:

$$
\frac{\lambda}{2} X_{ai}^2
$$

Derivada:

$$
\lambda X_{ai}
$$

Igualando a zero:

$$
X_{ai} = 0
$$

---

### Resultado do modelo ingênuo

- Para entradas observadas:  
  $$X_{ai} = \frac{Y_{ai}}{1+\lambda}$$

- Para entradas faltantes:  
  $$X_{ai} = 0$$

---

### Por que isso é ruim?

Esse modelo:

- **não generaliza**  
- **não descobre estrutura**  
- **não relaciona usuários entre si**  
- **não relaciona itens entre si**  
- **não prevê nada útil para entradas faltantes**  

Ele simplesmente “encolhe” as entradas observadas e zera as faltantes.

Esse fracasso motiva a introdução da **restrição de posto reduzido**, que será o próximo item.




<br><br>

<a id="lowrank"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Low Rank  
=============================

O conceito de **low rank** é o coração matemático da Matrix Completion.  
Ele explica por que podemos reconstruir uma matriz gigante a partir de poucos valores observados.

---

## 1. O que significa “rank”?

O **rank** de uma matriz $M$ é o número de direções independentes que ela contém.

Formalmente:

$$
\text{rank}(M) = \dim(\text{espaço gerado pelas colunas de } M).
$$

Intuição:

- rank alto → matriz complexa, cheia de variação  
- rank baixo → matriz simples, com estrutura comprimida  

---

## 2. Interpretação geométrica

Uma matriz de rank $k$ vive em um **subespaço de dimensão $k$**.

Exemplo:

- rank 1 → todas as linhas são múltiplos umas das outras  
- rank 2 → todas as linhas vivem em um plano  
- rank 10 → todas as linhas vivem em um subespaço de dimensão 10  

Mesmo que a matriz tenha milhões de linhas, ela pode viver em um espaço muito menor.

---

## 3. Fatoração de baixa dimensão

Se $M$ tem rank $k$, podemos escrever:

$$
M = U V^\top,
$$

onde:

- $U \in \mathbb{R}^{n \times k}$  
- $V \in \mathbb{R}^{m \times k}$  

Cada linha de $U$ é um vetor de **fatores latentes do usuário**.  
Cada linha de $V$ é um vetor de **fatores latentes do item**.

A predição é:

$$
M_{ij} = U_i^\top V_j.
$$

Essa é a base dos sistemas de recomendação.

---

## 4. Por que low rank é uma hipótese razoável?

### Em recomendação (Netflix, Amazon, Spotify):

- usuários têm poucos eixos de preferência  
- filmes têm poucos eixos de características  
- notas são combinações desses eixos  

Logo:

$$
\text{rank}(M) \ll \min(n,m).
$$

### Em imagens:

- imagens naturais vivem em subespaços de baixa dimensão  
- ruído aumenta o rank, mas a estrutura é simples  

### Em dados tabulares:

- variáveis são frequentemente correlacionadas  
- correlações reduzem o rank efetivo  

---

## 5. Conexão com SVD

A decomposição SVD escreve:

$$
M = U \Sigma V^\top.
$$

Se mantemos apenas os $k$ maiores valores singulares:

$$
M_k = U_k \Sigma_k V_k^\top,
$$

então $M_k$ é a melhor aproximação de rank $k$ de $M$.

Matrix Completion tenta recuperar **essa estrutura**, mas com dados faltantes.

---

## 6. Conexão com PCA

PCA encontra as direções de maior variância:

$$
M \approx U_k \Sigma_k V_k^\top.
$$

Isso é exatamente uma aproximação de rank $k$.

Matrix Completion é PCA **com buracos na matriz**.

---

## 7. Conexão com Matrix Completion

Matrix Completion assume:

$$
M \approx U V^\top,
$$

e otimiza:

$$
\min_{U,V}
\sum_{(i,j)\in\Omega}
\left(M_{ij} - U_i^\top V_j\right)^2
+
\lambda(\|U\|^2 + \|V\|^2).
$$

Ou seja:

- $U$ e $V$ são fatores latentes  
- rank = número de fatores latentes  
- reconstrução = produto $U V^\top$  

---

## 8. Conexão com Alternating Minimization

O problema não é convexo em $(U,V)$ juntos, mas é convexo em cada um separadamente.

Por isso usamos:

1. Fixar $V$ e otimizar $U$  
2. Fixar $U$ e otimizar $V$  
3. Repetir  

Cada passo é uma **ridge regression**.

---

## 9. Conexão com Prior Gaussiano

A regularização L2:

$$
\lambda(\|U\|^2 + \|V\|^2)
$$

equivale a assumir:

$$
U_i \sim \mathcal{N}(0, \sigma^2 I),
\qquad
V_j \sim \mathcal{N}(0, \sigma^2 I).
$$

Isso força $U$ e $V$ a viverem em um subespaço de baixa dimensão.

---

## 10. Conexão com Independência entre Entradas

O modelo ingênuo assume:

$$
M_{ij} = a_i + b_j.
$$

Isso é rank 2.

Matrix Completion assume:

$$
M_{ij} = U_i^\top V_j,
$$

que é rank $k$.

Ou seja:

- modelo ingênuo → rank muito baixo  
- matrix completion → rank baixo, mas expressivo  

---

## 11. Conclusão

Low Rank é:

- a hipótese estrutural que torna Matrix Completion possível  
- a base de sistemas de recomendação  
- a conexão entre SVD, PCA e fatores latentes  
- a razão pela qual Alternating Minimization funciona  
- a forma mais elegante de representar dados de alta dimensão com poucos parâmetros  

Sem low rank, Matrix Completion seria impossível.



<br><br>

<a id="objetivo"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Objective Function - Completa   
(Erro + Regularização + Posto Reduzido)  
=  

Depois de impor a estrutura de **posto reduzido**:

$$
X = U V^T,
$$

a Lecture 7 define a função objetivo completa que queremos minimizar:

$$
J(U, V) =
\sum_{(a,i)\in D} \frac{(Y_{ai} - U_a^\top V_i)^2}{2}
\;+\;
\frac{\lambda}{2} \sum_{a=1}^n \|U_a\|^2
\;+\;
\frac{\lambda}{2} \sum_{i=1}^m \|V_i\|^2.
$$

Essa função tem **três partes**, cada uma com papel essencial.

---

### 1. Termo de erro quadrático

$$
\sum_{(a,i)\in D} \frac{(Y_{ai} - U_a^\top V_i)^2}{2}
$$

Esse termo mede o quão bem o modelo explica as entradas observadas de $Y$.

- Se $U_a^\top V_i$ estiver perto de $Y_{ai}$, o erro é pequeno.  
- Se estiver longe, o erro é grande.  
- Entradas faltantes **não aparecem** aqui (pois não estão em $D$).

Esse termo força o modelo a **reproduzir as avaliações conhecidas**.

---

### 2. Termo de regularização sobre U

$$
\frac{\lambda}{2} \sum_{a=1}^n \|U_a\|^2
$$

Esse termo incentiva vetores de usuários pequenos.

Interpretação:

- evita que o modelo escolha valores absurdos para explicar poucos dados  
- reduz variância  
- melhora generalização  
- corresponde a um **prior Gaussiano** sobre $U$

---

### 3. Termo de regularização sobre V

$$
\frac{\lambda}{2} \sum_{i=1}^m \|V_i\|^2
$$

Mesmo papel do termo anterior, mas agora para os itens.

Ele garante que:

- características dos itens não explodam  
- o modelo permaneça simples  
- múltiplas explicações sejam distribuídas de forma equilibrada

---

### 4. Por que essa função é boa?

Ela combina:

1. **Aderência aos dados**  
   (erro quadrático nas entradas observadas)

2. **Simplicidade do modelo**  
   (regularização L2)

3. **Estrutura compartilhada**  
   (posto reduzido via $X = UV^T$)

Essa combinação resolve todos os problemas do modelo ingênuo:

- entradas faltantes deixam de ser zero  
- usuários e itens passam a ter representações latentes  
- previsões passam a ser coerentes  
- o modelo generaliza  
- evita soluções absurdas como $0.00004 \times 400000$

---

### 5. Forma compacta

Às vezes escrevemos:

$$
J(U,V) = 
\sum_{(a,i)\in D} \frac{(Y_{ai} - U_a^\top V_i)^2}{2}
+ \frac{\lambda}{2}(\|U\|_F^2 + \|V\|_F^2)
$$

onde:

- $\|U\|_F^2 = \sum_a \|U_a\|^2$  
- $\|V\|_F^2 = \sum_i \|V_i\|^2$

---

### 6. Conexão com regressão ridge

Para cada usuário $a$, fixando $V$, o problema vira:

$$
\min_{U_a}
\sum_{i\in D_a} \frac{(Y_{ai} - U_a^\top V_i)^2}{2}
+ \frac{\lambda}{2}\|U_a\|^2
$$

Isso é exatamente **regressão ridge** em dimensão $k$.

O mesmo vale para cada item $i$.

Essa observação é o que permite o algoritmo de **projeções alternadas** (alternating minimization).

---

### 7. Conclusão

A função objetivo completa é o coração da Lecture 7:

- combina erro + regularização + posto reduzido  
- conecta matrix completion com filtragem colaborativa  
- permite prever entradas faltantes  
- é convexa em $U$ quando $V$ é fixo  
- é convexa em $V$ quando $U$ é fixo  
- leva naturalmente ao algoritmo de alternating minimization

O próximo item será **Alternating Minimization (Projeções Alternadas)**.



<br><br>

<a id="alter_min"></a>
[$\Uparrow$ Índice](#indice)



=============================  
Alternating Minimization (Projeções Alternadas)  
=============================  

Depois de definir a função objetivo completa:

$$
J(U, V) =
\sum_{(a,i)\in D} \frac{(Y_{ai} - U_a^\top V_i)^2}{2}
+
\frac{\lambda}{2}\sum_a \|U_a\|^2
+
\frac{\lambda}{2}\sum_i \|V_i\|^2,
$$

a Lecture 7 introduz o algoritmo fundamental para resolver esse problema:

> **Alternating Minimization (ou Projeções Alternadas)**  
> Fixe $V$, otimize $U$.  
> Fixe $U$, otimize $V$.  
> Repita.

Esse método funciona porque, embora o problema completo não seja convexo em $(U,V)$ juntos, ele **é convexo em U quando V é fixo** e **é convexo em V quando U é fixo**.

---

### 1. Ideia central

O algoritmo alterna entre dois subproblemas simples:

#### Passo A — Fixar $V$, otimizar $U$

Para cada usuário $a$, resolvemos:

$$
\min_{U_a}
\sum_{i\in D_a} \frac{(Y_{ai} - U_a^\top V_i)^2}{2}
+
\frac{\lambda}{2}\|U_a\|^2.
$$

Esse problema é **regressão ridge** em dimensão $k$.

#### Passo B — Fixar $U$, otimizar $V$

Para cada item $i$, resolvemos:

$$
\min_{V_i}
\sum_{a\in D_i} \frac{(Y_{ai} - U_a^\top V_i)^2}{2}
+
\frac{\lambda}{2}\|V_i\|^2.
$$

Também é regressão ridge em dimensão $k$.

---

### 2. Solução fechada para cada usuário

Fixando $V$, a solução para cada $U_a$ é:

$$
U_a
=
\frac{\sum_{i\in D_a} Y_{ai} V_i}
{\sum_{i\in D_a} V_i^2 + \lambda}.
$$

Essa fórmula aparece repetidamente na Lecture 7.

---

### 3. Solução fechada para cada item

Fixando $U$, a solução para cada $V_i$ é:

$$
V_i
=
\frac{\sum_{a\in D_i} Y_{ai} U_a}
{\sum_{a\in D_i} U_a^2 + \lambda}.
$$

Simétrica à fórmula anterior.

---

### 4. Estrutura do algoritmo

O algoritmo completo é:

1. Inicialize $U^{(0)}$ e $V^{(0)}$.  
2. Para $t = 0, 1, 2, \dots$:
   - Atualize $U^{(t+1)}$ usando $V^{(t)}$
   - Atualize $V^{(t+1)}$ usando $U^{(t+1)}$
3. Pare quando convergir.

Cada passo reduz o valor de $J(U,V)$.

---

### 5. Interpretação geométrica

- Fixar $V$ significa:  
  “os itens têm características fixas; descubra os gostos dos usuários”.

- Fixar $U$ significa:  
  “os usuários têm gostos fixos; descubra as características dos itens”.

O algoritmo alterna entre essas duas interpretações até que ambos se ajustem mutuamente.

---

### 6. Por que funciona?

Porque cada subproblema é:

- estritamente convexo  
- tem solução fechada  
- reduz o valor da função objetivo  
- aproxima $X = UV^T$ das entradas observadas  
- mantém $U$ e $V$ simples via regularização

---

### 7. Conexão com filtragem colaborativa

Alternating minimization é exatamente o algoritmo usado em:

- sistemas de recomendação  
- fatoração de matrizes  
- modelos de fatores latentes  
- reconstrução de imagens  
- compressão de dados

É o método clássico para aprender $U$ e $V$.

---

### 8. Conclusão

Alternating minimization:

- resolve matrix completion com posto reduzido  
- é eficiente  
- é simples  
- tem solução fechada em cada passo  
- é o núcleo da Lecture 7  
- prepara o terreno para a Lecture 8 (Collaborative Filtering)

O próximo item será **Atualização fechada de U e V (Regressão Ridge 1D)**.




<br><br>

<a id="ridge_1d"></a>
[$\Uparrow$ Índice](#indice)


=============================  
Atualização Fechada de U e V (Regressão Ridge 1D)  
=============================  

A Lecture 7 mostra que, ao fixarmos $V$, cada vetor de usuário $U_a$ pode ser atualizado **independentemente** dos demais.  
Da mesma forma, ao fixarmos $U$, cada vetor de item $V_i$ também é atualizado independentemente.

Cada atualização é exatamente uma **regressão ridge em 1 dimensão**, o que torna o algoritmo extremamente simples e eficiente.

---

## 1. Atualização fechada de $U_a$ (fixando $V$)

Para um usuário $a$, queremos minimizar:

$$
\min_{U_a}
\sum_{i\in D_a} \frac{(Y_{ai} - U_a^\top V_i)^2}{2}
+
\frac{\lambda}{2}\|U_a\|^2.
$$

Como $k = 1$ na Lecture 7, $U_a$ e $V_i$ são escalares.  
Então o problema vira:

$$
\min_{U_a}
\sum_{i\in D_a} \frac{(Y_{ai} - U_a V_i)^2}{2}
+
\frac{\lambda}{2} U_a^2.
$$

Derivando e igualando a zero:

$$
\sum_{i\in D_a} (U_a V_i^2 - Y_{ai} V_i) + \lambda U_a = 0.
$$

Agrupando termos:

$$
U_a \left( \sum_{i\in D_a} V_i^2 + \lambda \right)
=
\sum_{i\in D_a} Y_{ai} V_i.
$$

Portanto:

$$
U_a
=
\frac{\sum_{i\in D_a} Y_{ai} V_i}
{\sum_{i\in D_a} V_i^2 + \lambda}.
$$

Essa é a fórmula usada em todos os exercícios da Lecture 7.

---

## 2. Atualização fechada de $V_i$ (fixando $U$)

Simetricamente, para cada item $i$:

$$
\min_{V_i}
\sum_{a\in D_i} \frac{(Y_{ai} - U_a V_i)^2}{2}
+
\frac{\lambda}{2} V_i^2.
$$

Derivando e igualando a zero:

$$
\sum_{a\in D_i} (V_i U_a^2 - Y_{ai} U_a) + \lambda V_i = 0.
$$

Agrupando:

$$
V_i \left( \sum_{a\in D_i} U_a^2 + \lambda \right)
=
\sum_{a\in D_i} Y_{ai} U_a.
$$

Portanto:

$$
V_i
=
\frac{\sum_{a\in D_i} Y_{ai} U_a}
{\sum_{a\in D_i} U_a^2 + \lambda}.
$$

---

## 3. Interpretação como regressão ridge 1D

Cada atualização é uma regressão ridge:

- variável explicativa: $V_i$ (ou $U_a$)  
- variável alvo: $Y_{ai}$  
- regularização: $\lambda$  

Isso significa:

- atualizações são **estáveis**  
- evitam explosão de valores  
- distribuem crença entre múltiplas explicações  
- garantem boa generalização  

---

## 4. Por que isso é eficiente?

- Cada $U_a$ depende apenas dos itens avaliados pelo usuário $a$.  
- Cada $V_i$ depende apenas dos usuários que avaliaram o item $i$.  
- Não há necessidade de resolver sistemas grandes.  
- Cada passo é $O(|D|)$.  
- O algoritmo escala para milhões de usuários e itens.

---

## 5. Conclusão

As fórmulas fechadas para cada entrada observada:

$$
U_a
=
\frac{\sum_{i\in D_a} Y_{ai} V_i}
{\sum_{i\in D_a} V_i^2 + \lambda} \Rightarrow \quad \forall  U_a\to V = (U^TU + \lambda Ι)^{-1}U^TY ,
\qquad
V_i
=
\frac{\sum_{a\in D_i} Y_{ai} U_a}
{\sum_{a\in D_i} U_a^2 + \lambda}
$$

são o núcleo matemático da Lecture 7.

Elas permitem:

- resolver matrix completion com posto reduzido  
- implementar alternating minimization  
- interpretar o modelo como regressão ridge  
- construir sistemas de recomendação eficientes  

O próximo item será **Interpretação Bayesiana (Prior Gaussiano)**.



<br><br>


<a id="bayesiana"></a>
[$\Uparrow$ Índice](#indice)


=============================  
Interpretação Bayesiana (Prior Gaussiano)  
=============================  

A Lecture 7 mostra que a regularização L2 usada na função objetivo:

$$
J(U,V) =
\sum_{(a,i)\in D} \frac{(Y_{ai} - U_a^\top V_i)^2}{2}
+
\frac{\lambda}{2}\sum_a \|U_a\|^2
+
\frac{\lambda}{2}\sum_i \|V_i\|^2
$$

não é apenas uma técnica de otimização — ela tem uma **interpretação probabilística elegante** baseada em **priors Gaussianos**.

Essa visão é fundamental para entender por que o modelo não explode, por que ele generaliza bem e por que as atualizações fechadas fazem sentido.

---

## 1. O prior Gaussiano sobre U e V

Assumimos que, antes de observar os dados, acreditamos que os vetores $U_a$ e $V_i$ devem ser pequenos:

$$
U_a \sim \mathcal{N}(0, \sigma^2 I),
\qquad
V_i \sim \mathcal{N}(0, \sigma^2 I).
$$

Isso significa:

- média zero  
- variância $\sigma^2$  
- componentes independentes  

Interpretando:

> A priori, acreditamos que usuários e itens não têm características extremas.  
> Vetores pequenos são mais prováveis.

---

## 2. Como isso gera regularização L2

O log da densidade Gaussiana é:

$$
\log p(U_a) = -\frac{1}{2\sigma^2}\|U_a\|^2 + \text{const},
$$

e o mesmo vale para $V_i$.

No aprendizado Bayesiano, buscamos o **MAP (máximo a posteriori)**:

$$
(U^*, V^*) = \arg\max_{U,V}
\left[
\log p(Y|U,V) + \log p(U) + \log p(V)
\right].
$$

Isso é equivalente a minimizar:

$$
\sum_{(a,i)\in D} \frac{(Y_{ai} - U_a^\top V_i)^2}{2}
+
\frac{1}{2\sigma^2}
\left(
\sum_a \|U_a\|^2 + \sum_i \|V_i\|^2
\right).
$$

Identificando:

$$
\lambda = \frac{1}{\sigma^2},
$$

vemos que:

> **Regularização L2 é exatamente um prior Gaussiano sobre U e V.**

---

## 3. Intuição do prior

- $\sigma^2$ grande → prior fraco → vetores podem ser grandes → modelo complexo  
- $\sigma^2$ pequeno → prior forte → vetores próximos de zero → modelo simples  

Quando $\lambda$ é grande (ou $\sigma^2$ pequeno):

- o modelo evita explicações extremas  
- distribui responsabilidade entre vários fatores  
- evita explosões numéricas  
- melhora generalização  

---

## 4. Conexão com o problema ingênuo

Sem prior (sem regularização):

- o modelo pode escolher valores absurdos para explicar poucos dados  
- soluções como $0.00004 \times 400000$ aparecem  
- o modelo não generaliza  
- entradas faltantes ficam incoerentes

Com prior Gaussiano:

- vetores são puxados para perto da origem  
- múltiplas explicações são equilibradas  
- o modelo é estável  
- previsões são coerentes

---

## 5. Conexão com regressão ridge

A atualização fechada de cada usuário:

$$
U_a
=
\frac{\sum_{i\in D_a} Y_{ai} V_i}
{\sum_{i\in D_a} V_i^2 + \lambda}
$$

é exatamente a solução MAP de uma regressão linear com prior Gaussiano.

O mesmo vale para cada item:

$$
V_i
=
\frac{\sum_{a\in D_i} Y_{ai} U_a}
{\sum_{a\in D_i} U_a^2 + \lambda}.
$$

---

## 6. Conclusão

A interpretação Bayesiana explica:

- por que a regularização é necessária  
- por que o modelo não explode  
- por que alternating minimization funciona  
- por que as soluções fechadas são estáveis  
- por que o modelo generaliza bem  

O prior Gaussiano é a base probabilística da Lecture 7 e conecta:

- matrix completion  
- filtragem colaborativa  
- regressão ridge  
- regularização L2  
- aprendizado Bayesiano  

O próximo item será **Conexão com Sistemas de Recomendação (Fatores Latentes)**.





<br><br>

<a id="latente"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Fatores Latentes  
=============================

Os **fatores latentes** são a interpretação humana da decomposição de baixa dimensão:

$$
M \approx U V^\top.
$$

Eles explicam *por que* o modelo funciona e *o que* ele está realmente aprendendo.

---

## 1. O que são fatores latentes?

Cada linha de $U$ representa um vetor de **preferências ocultas** de um usuário:

$$
U_i = (u_{i1}, u_{i2}, \dots, u_{ik}).
$$

Cada linha de $V$ representa um vetor de **características ocultas** de um item:

$$
V_j = (v_{j1}, v_{j2}, \dots, v_{jk}).
$$

A predição é:

$$
\hat{M}_{ij} = U_i^\top V_j.
$$

Interpretando:

- se $U_i$ e $V_j$ são parecidos → nota alta  
- se são diferentes → nota baixa  

---

## 2. Intuição geométrica

Imagine que cada usuário vive em um espaço de dimensão $k$:

- eixo 1 → gosta de ação  
- eixo 2 → gosta de comédia  
- eixo 3 → gosta de drama  
- eixo 4 → gosta de filmes leves  
- eixo 5 → gosta de filmes pesados  

E cada filme também vive nesse espaço:

- eixo 1 → quão “ação” ele é  
- eixo 2 → quão “comédia” ele é  
- eixo 3 → quão “drama” ele é  
- eixo 4 → quão leve  
- eixo 5 → quão pesado  

A nota é o **produto interno** entre esses vetores.

---

## 3. Por que isso funciona?

Porque preferências humanas e características de itens **não são aleatórias**.  
Elas vivem em poucos eixos estruturais.

Exemplos:

- pessoas não têm 784 preferências independentes  
- filmes não têm 784 características independentes  
- músicas não têm 10.000 características independentes  

A estrutura é **low rank**.

---

## 4. Conexão com Matrix Completion

Matrix Completion assume:

$$
M_{ij} = U_i^\top V_j.
$$

Mas só observa alguns valores de $M$.

Os fatores latentes permitem:

- reconstruir valores faltantes  
- prever notas futuras  
- recomendar itens  
- identificar similaridade entre usuários  
- identificar similaridade entre itens  

Tudo isso sem nunca observar a matriz completa.

---

## 5. Conexão com Alternating Minimization

Alternating Minimization aprende os fatores latentes:

1. Fixar $V$ → aprender $U$  
2. Fixar $U$ → aprender $V$  
3. Repetir

Cada passo é uma **ridge regression**.

No final:

- $U$ captura preferências  
- $V$ captura características  
- $U V^\top$ reconstrói a matriz

---

## 6. Conexão com Prior Gaussiano

A regularização L2:

$$
\lambda(\|U\|^2 + \|V\|^2)
$$

equivale a assumir:

$$
U_i \sim \mathcal{N}(0, \sigma^2 I),
\qquad
V_j \sim \mathcal{N}(0, \sigma^2 I).
$$

Isso força os fatores latentes a:

- não explodirem  
- viverem em um subespaço compacto  
- evitar overfitting  

---

## 7. Conexão com SVD e PCA

Se a matriz fosse totalmente observada, os fatores latentes seriam:

$$
U_k \Sigma_k^{1/2},
\qquad
V_k \Sigma_k^{1/2}.
$$

Ou seja:

- SVD → fatores latentes explícitos  
- PCA → fatores latentes das colunas  
- Matrix Completion → fatores latentes com buracos na matriz  

---

## 8. Conexão com modelos modernos

Fatores latentes são usados em:

- recomendação (Netflix, Amazon, Spotify)  
- embeddings de palavras (word2vec, GloVe)  
- embeddings de usuários  
- embeddings de itens  
- modelos de grafos  
- modelos de deep learning (camadas densas são low rank)  

Eles são uma das ideias mais importantes da IA moderna.

---

## 9. Conclusão

Fatores latentes são:

- a interpretação humana da decomposição $U V^\top$  
- preferências ocultas + características ocultas  
- a base de Matrix Completion  
- a base de recomendação  
- a conexão entre SVD, PCA e low rank  
- uma das ideias mais elegantes da Unit 2  

Sem fatores latentes, Matrix Completion não teria significado.



<br><br>



<a id="unit3"></a>


<a id="onehot"></a>
[$\Uparrow$ Índice](#indice)

=============================  
One‑hot Encoding
=============================

Modelos de linguagem precisam transformar palavras em vetores numéricos para que redes neurais possam operar sobre elas.  
A representação mais simples é o **one‑hot encoding**.

Suponha um vocabulário com $K$ palavras.  
Cada palavra é representada por um vetor:

$$
x \in \mathbb{R}^K
$$

onde:

- todos os valores são 0  
- exceto um único valor igual a 1  
- indicando a posição da palavra no vocabulário

Exemplo:

Vocabulário = $\{\text{ML}, \text{course}, \text{is}, \text{UNK}\}$

- ML → $[1,0,0,0]$  
- course → $[0,1,0,0]$  
- is → $[0,0,1,0]$  
- UNK → $[0,0,0,1]$

### Pensamento aplicado

O one‑hot tem duas propriedades fundamentais:

1. **Não contém informação semântica.**  
   Palavras diferentes são ortogonais:  $$x_i^\top x_j = 0 \quad \text{se } i \neq j.$$

2. **Permite indexar diretamente colunas de uma matriz de pesos.**  
   Se a rede tem matriz $W$, então:  $$ Wx $$  
   seleciona exatamente **uma coluna** de $$W$$.

Isso simplifica o cálculo e conecta diretamente modelos de linguagem com redes neurais lineares.

### Conclusão

One‑hot é simples, eficiente para indexação e essencial para entender como modelos de linguagem transformam palavras em números.  
Ele prepara o terreno para embeddings, que substituem essa representação por vetores densos e semânticos.



<br><br>




<a id="softmax"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Softmax
=============================

A função softmax transforma um vetor de valores arbitrários (logits) em uma **distribuição de probabilidade**.

Dado um vetor $z \in \mathbb{R}^K$, definimos:

$$
\text{softmax}(z)_j
= \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
$$

Propriedades fundamentais:

1. $p_j \ge 0$  
2. $\sum_j p_j = 1$  
3. valores maiores em $z_j$ produzem probabilidades maiores  
4. diferenças relativas entre logits são preservadas

### Intuição

O softmax funciona como um “competidor exponencial”:

- cada logit $z_j$ é elevado a $e^{z_j}$  
- valores maiores crescem muito mais rápido  
- a normalização transforma tudo em probabilidades

Assim, o softmax escolhe a palavra mais provável, mas ainda permite incerteza.

### Pensamento aplicado

O softmax é essencial em modelos de linguagem porque:

1. **Permite interpretar a saída da rede como probabilidade.**  
   Sem softmax, a rede produziria apenas números arbitrários.

2. **Permite amostragem.**  
   A próxima palavra é escolhida de acordo com $p_j$.

3. **Permite treinamento via cross‑entropy.**  
   A função de perda compara a distribuição prevista com a distribuição verdadeira (one‑hot).

4. **É usado em todas as arquiteturas:**  
   - feedforward  
   - RNN  
   - seq2seq  
   - transformers

### Conclusão

Softmax é a ponte entre redes neurais e probabilidade.  
Ele transforma logits em distribuições e permite que modelos de linguagem façam predições interpretáveis.






<br><br>



<a id="markov"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Modelos de Markov (n‑gram)
=============================

Um modelo de linguagem precisa estimar a probabilidade de uma sequência de palavras.  
O modelo de Markov (ou n‑gram) faz isso assumindo que cada palavra depende apenas de um número fixo de palavras anteriores.

Para um bigrama (ordem 1), estimamos:

$$
P(w_t \mid w_{t-1})
$$

A probabilidade de uma sentença é:

$$
P(w_1, w_2, \dots, w_T)
= \prod_{t=1}^T P(w_t \mid w_{t-1})
$$

### Estimativa por Máxima Verossimilhança (MLE)

A probabilidade de transição é estimada diretamente das contagens:

$$
P(w_j \mid w_i)
= \frac{\text{count}(w_i \to w_j)}{\text{count}(w_i)}
$$

Ou seja:

- contamos quantas vezes $w_i$ aparece  
- contamos quantas vezes $w_i$ é seguido por $w_j$  
- dividimos uma contagem pela outra

### Pensamento aplicado

O modelo de Markov tem duas características fundamentais:

1. **Histórico fixo.**  
   Ele só olha para as últimas $n$ palavras.  
   Isso limita a capacidade de capturar dependências longas.

2. **Explosão combinatória.**  
   O número de parâmetros cresce com o tamanho do vocabulário: $$K^n$$  
   para um modelo n‑gram.

Isso explica por que modelos de Markov são simples, mas rapidamente se tornam impraticáveis para vocabulários grandes.

### Conclusão

Modelos de Markov são a base histórica dos modelos de linguagem.  
Eles são simples, interpretáveis e fáceis de treinar, mas sofrem com histórico fixo e falta de generalização — motivando o uso de redes neurais.



<br><br>



<a id="feedforward"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Feedforward Networks para Linguagem
=============================

Modelos de Markov têm histórico fixo e não generalizam bem.  
Uma alternativa é usar uma **rede neural feedforward** para prever a próxima palavra.

Dado um vetor one‑hot $x$, a rede calcula:

$$
h = f(Wx + b)
$$

$$
p = \text{softmax}(Vh + c)
$$

onde:

- $W$ transforma a entrada  
- $f$ é uma função de ativação (ReLU)  
- $V$ transforma a camada oculta  
- softmax produz uma distribuição de probabilidade sobre o vocabulário

### Interpretação

A rede aprende a mapear padrões de entrada para distribuições de saída.  
Mesmo com entrada one‑hot, ela generaliza porque os pesos aprendem relações entre palavras.

### Número de parâmetros

Se o vocabulário tem tamanho $K$ e a camada oculta tem tamanho $H$:

- primeira camada: $K \times H + H$  
- segunda camada: $H \times K + K$

Total:

$$
KH + HK + H + K
$$

### Pensamento aplicado

A rede feedforward resolve duas limitações dos modelos de Markov:

1. **Generalização.**  
   Palavras raras podem ser representadas por padrões nos pesos, não apenas por contagens.

2. **Não linearidade.**  
   A ReLU permite que o modelo represente regiões complexas no espaço de entrada.

Por outro lado, ela ainda tem histórico fixo:  
a entrada representa apenas as últimas palavras fornecidas.

### Conclusão

Redes feedforward são uma extensão natural dos modelos n‑gram, com melhor generalização e capacidade não linear.  
Elas preparam o terreno para RNNs, que resolvem o problema do histórico fixo.



<br><br>



<a id="relu"></a>
[$\Uparrow$ Índice](#indice)

=============================  
ReLU, Fronteiras e Regiões Lineares
=============================

A função ReLU é uma das ativações mais usadas em redes neurais modernas.  
Ela é definida como:

$$
f(z) = \max(z, 0)
$$

Ou seja:

- se $z > 0$: o neurônio fica **ativo**  
- se $z \le 0$: o neurônio fica **apagado**  

Cada neurônio com ReLU define uma **fronteira de decisão linear** no espaço de entrada, exatamente onde:

$$
z = 0
$$

Essa fronteira divide o espaço em duas regiões:

- região onde $z > 0$ → saída linear  
- região onde $z < 0$ → saída igual a 0  

### Exemplo das fronteiras

Se a rede tem neurônios:

- $z_1 = x_1 - 1$ → fronteira: $x_1 = 1$  
- $z_2 = x_2 - 1$ → fronteira: $x_2 = 1$  
- $z_3 = -x_1 - 1$ → fronteira: $x_1 = -1$  
- $z_4 = -x_2 - 1$ → fronteira: $x_2 = -1$

Cada equação $z_i = 0$ é uma linha reta que separa regiões onde o neurônio liga ou desliga.

### Regiões lineares

Como cada neurônio liga ou desliga dependendo do sinal de $z_i$, a rede cria **regiões lineares** no espaço de entrada.

Por exemplo, a região onde **todos** os neurônios estão negativos é:

$$
-1 < x_1 < 1, \quad -1 < x_2 < 1
$$

Dentro dessa região, a rede se comporta como uma função **linear fixa**.  
Em outra região, com outros neurônios ativos, a rede usa **outra** função linear.

Assim, a rede é uma função **piecewise linear**.

### Pensamento aplicado

A ReLU é importante porque:

1. **Divide o espaço em regiões lineares.**  
   Cada combinação de neurônios ativos define uma “peça” da função.

2. **Permite que redes profundas representem funções complexas.**  
   Quanto mais neurônios e camadas, mais regiões lineares.

3. **Simplifica o cálculo.**  
   ReLU é barata e evita saturação (como sigmoid).

4. **Explica a geometria das redes neurais.**  
   A fronteira de decisão final é composta por pedaços de hiperplanos.

### Conclusão

A ReLU transforma redes neurais em funções piecewise‑lineares.  
Cada neurônio define uma fronteira $z=0$, e a combinação dessas fronteiras cria regiões onde a rede usa diferentes funções lineares.



<br><br>



<a id="embeddings"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Embeddings
=============================

A representação one‑hot é simples, mas tem duas limitações importantes:

1. não captura nenhuma relação semântica entre palavras  
2. produz vetores de dimensão igual ao vocabulário (muito grandes)

Para resolver isso, usamos **embeddings**, que são vetores densos aprendidos pela rede.

Um embedding é obtido multiplicando o one‑hot por uma matriz de embedding $E$:

$$
x_{\text{embed}} = E x_{\text{onehot}}
$$

Se o vocabulário tem tamanho $K$ e queremos embeddings de dimensão $d$, então:

- $E$ é uma matriz $K \times d$  
- cada linha de $E$ é o embedding de uma palavra  

Assim, o embedding de uma palavra é simplesmente a linha correspondente de $E$.

### Intuição

Embeddings permitem que palavras com significados semelhantes tenham vetores próximos.  
Por exemplo:

- “king” e “queen” ficam próximos  
- “dog” e “cat” ficam próximos  
- “run” e “walk” ficam mais próximos entre si do que de “banana”

Isso acontece porque a rede ajusta $E$ durante o treinamento para minimizar a perda da tarefa (prever a próxima palavra, traduzir, etc.).

### Pensamento aplicado

Embeddings resolvem três problemas fundamentais do one‑hot:

1. **Dimensionalidade reduzida.**  
   Em vez de vetores de tamanho $K$, usamos vetores de tamanho $d \ll K$.

2. **Semântica distribuída.**  
   Significados são representados por padrões contínuos nos vetores.

3. **Generalização.**  
   Palavras raras podem ser representadas por embeddings úteis, mesmo com poucas ocorrências.

Além disso, embeddings são usados em:

- feedforward  
- RNN  
- seq2seq  
- transformers  
- modelos modernos de NLP

### Conclusão

Embeddings substituem one‑hot por vetores densos e semânticos.  
Eles permitem que modelos de linguagem capturem relações entre palavras e generalizem muito melhor.




<br><br>



<a id="backprop"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Retropropagação (Backpropagation)
=============================

A retropropagação é o algoritmo que permite treinar redes neurais ajustando seus pesos para minimizar uma função de perda.  
Ela calcula como cada peso contribui para o erro final e atualiza esses pesos usando gradiente descendente.

Considere uma rede com camadas:

$$
h = f(Wx + b)
$$

$$
p = \text{softmax}(Vh + c)
$$

e uma perda $L(p, y)$.

A retropropagação calcula:

1. o gradiente da perda em relação à saída  
2. o gradiente da saída em relação às ativações  
3. o gradiente das ativações em relação aos pesos  

Aplicando a regra da cadeia:

$$
\frac{\partial L}{\partial W}
= \frac{\partial L}{\partial h}
\cdot
\frac{\partial h}{\partial W}
$$

e de forma geral:

$$
\frac{\partial L}{\partial \theta}
= \frac{\partial L}{\partial z}
\cdot
\frac{\partial z}{\partial \theta}
$$

onde $\theta$ é qualquer peso da rede.

### Gradiente para uma camada linear

Se:

$$
z = Wx + b
$$

então:

$$
\frac{\partial L}{\partial W}
= \frac{\partial L}{\partial z} \, x^\top
$$

$$
\frac{\partial L}{\partial b}
= \frac{\partial L}{\partial z}
$$

### Gradiente através da ReLU

A ReLU é:

$$
f(z) = \max(z, 0)
$$

Seu gradiente é:

$$
f'(z) =
\begin{cases}
1, & z > 0 \\
0, & z \le 0
\end{cases}
$$

Ou seja:

- neurônios ativos passam gradiente  
- neurônios apagados bloqueiam gradiente  

### Pensamento aplicado

A retropropagação é essencial porque:

1. **Permite treinar redes profundas.**  
   Sem ela, não saberíamos como ajustar pesos em camadas internas.

2. **Explica por que RNNs sofrem com vanishing/exploding gradients.**  
   Em uma RNN, o estado é atualizado repetidamente:  $$s_t = f(Wx_t + U s_{t-1})$$  
   O gradiente passa por $U$ muitas vezes.  
   Se $U$ tem autovalores pequenos → gradiente desaparece.  
   Se $U$ tem autovalores grandes → gradiente explode.

3. **Motiva gating e LSTM.**  
   LSTM foi criada para controlar o fluxo de gradiente ao longo do tempo.

4. **Conecta feedforward, RNN e LSTM.**  
   Todos usam retropropagação — apenas a estrutura muda.

### Conclusão

Retropropagação é o mecanismo que permite que redes neurais aprendam.  
Ela calcula gradientes camada por camada e explica tanto o funcionamento quanto as limitações das RNNs, motivando arquiteturas como LSTM.





<br><br>



<a id="rnn"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Redes Neurais Recorrentes (RNN)
=============================

Modelos feedforward têm histórico fixo: só conseguem olhar para as últimas palavras fornecidas na entrada.  
As **Redes Recorrentes (RNNs)** resolvem esse problema introduzindo um **estado oculto** que carrega informação ao longo do tempo.

A RNN processa uma sequência palavra por palavra.  
Para cada passo $t$, ela recebe:

- a palavra atual $x_t$ (one‑hot ou embedding)  
- o estado anterior $s_{t-1}$  

e produz:

$$
s_t = f(Wx_t + U s_{t-1} + b)
$$

onde:

- $W$ transforma a entrada  
- $U$ transforma o estado anterior  
- $f$ é uma ativação (geralmente ReLU ou $\tanh$)  
- $s_t$ é o novo estado, contendo memória da sequência

A predição da próxima palavra é:

$$
p_t = \text{softmax}(V s_t + c)
$$

### Intuição

O estado $s_t$ funciona como uma “memória comprimida” da frase.  
Ele acumula informações ao longo do tempo:

- $s_1$ contém informação de $x_1$  
- $s_2$ contém informação de $(x_1, x_2)$  
- $s_3$ contém informação de $(x_1, x_2, x_3)$  
- e assim por diante

Assim, a RNN consegue capturar dependências longas que modelos n‑gram e feedforward não conseguem.

### Pensamento aplicado

A RNN resolve três limitações fundamentais dos modelos anteriores:

1. **Histórico variável.**  
   O estado $s_t$ cresce com a sequência, não é fixo.

2. **Generalização temporal.**  
   A mesma rede é aplicada em todos os passos, compartilhando parâmetros.

3. **Memória distribuída.**  
   O estado não guarda palavras diretamente, mas sim uma representação contínua da sequência.

Por outro lado, RNNs sofrem com:

- gradientes que desaparecem ou explodem  
- dificuldade em capturar dependências muito longas  
- sensibilidade à inicialização

Esses problemas motivam arquiteturas mais avançadas (LSTM, GRU), mas a RNN simples é a base conceitual.

### Conclusão

A RNN introduz memória recorrente, permitindo que modelos de linguagem processem sequências de qualquer tamanho.  
Ela é o primeiro modelo capaz de capturar dependências longas e preparar o terreno para seq2seq e transformers.



<br><br>



<a id="gating"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Gating em Redes Recorrentes
=============================

RNNs simples sofrem com dois problemas graves:

1. **vanishing gradients** — o gradiente fica cada vez menor ao atravessar muitos passos  
2. **exploding gradients** — o gradiente cresce demais ao atravessar muitos passos  

Isso acontece porque o estado recorrente é atualizado repetidamente:

$$
s_t = f(Wx_t + U s_{t-1})
$$

O gradiente passa por $U$ várias vezes.  
Se os autovalores de $U$ forem:

- menores que 1 → gradiente desaparece  
- maiores que 1 → gradiente explode  

Para resolver isso, surgem os **gates** (portões).

---

### O que é um gate?

Um **gate** é um neurônio especial que controla o fluxo de informação.  
Ele decide **quanto** da informação deve:

- entrar  
- sair  
- ser esquecida  
- ser mantida  

Matematicamente, um gate é uma função sigmoide:

$$
g = \sigma(Wx_t + U s_{t-1} + b)
$$

onde:

- $g \in (0,1)$  
- $g$ funciona como um “interruptor contínuo”  
- $g \approx 0$ → bloqueia informação  
- $g \approx 1$ → deixa passar informação  

---

### Como gates resolvem vanishing/exploding?

Gates permitem que a rede:

- **mantenha** informações importantes por muitos passos  
- **bloqueie** informações irrelevantes  
- **controle** o fluxo de gradiente  

Em vez de depender apenas de $U$, a rede passa a depender de combinações controladas:

$$
\text{informação nova} \times g_{\text{input}}
$$

$$
\text{informação antiga} \times g_{\text{forget}}
$$

Isso evita que o gradiente seja multiplicado repetidamente por valores ruins.

---

### Pensamento aplicado

Gating é a ponte entre:

- RNN simples (memória fraca)  
- LSTM/GRU (memória forte e controlada)  

Ele introduz três ideias fundamentais:

1. **controle de fluxo**  
   A rede decide o que lembrar e o que esquecer.

2. **memória seletiva**  
   Informação importante pode ser mantida por dezenas de passos.

3. **gradiente estável**  
   Gates permitem que o gradiente flua sem desaparecer ou explodir.

Sem gating, LSTM não existiria.

---

### Conclusão

Gates são mecanismos que controlam o fluxo de informação em redes recorrentes.  
Eles resolvem vanishing/exploding gradients e permitem que modelos como LSTM mantenham memória longa de forma estável.




<br><br>



<a id="lstm"></a>
[$\Uparrow$ Índice](#indice)

=============================  
LSTM | Long Short-Term Memory
=============================

A LSTM é uma arquitetura recorrente criada para resolver o problema de **vanishing/exploding gradients** das RNNs simples.  
Ela faz isso usando **gates** (portões) que controlam o fluxo de informação ao longo do tempo.

A LSTM mantém dois estados:

- estado oculto: $h_t$  
- estado de memória: $c_t$  

O estado de memória $c_t$ é o que permite guardar informação por muitos passos.

---

### Gates da LSTM

A LSTM usa três gates sigmoides e uma atualização candidata:

1. **Gate de esquecimento**  $$ f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) $$

2. **Gate de entrada**  $$ i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) $$

3. **Atualização candidata**  $$ \tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c) $$

4. **Gate de saída**  $$ o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) $$

---

### Atualização da memória

A memória é atualizada combinando:

- o que deve ser esquecido  
- o que deve ser adicionado  

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

onde $\odot$ é multiplicação elemento a elemento.

---

### Atualização do estado oculto

O estado oculto é:

$$
h_t = o_t \odot \tanh(c_t)
$$

Ele é usado para prever a próxima palavra:

$$
p_t = \text{softmax}(V h_t + c)
$$

---

### Intuição

A LSTM controla o fluxo de informação usando gates:

- **$f_t$ (forget gate)** decide o que deve ser apagado da memória  
- **$i_t$ (input gate)** decide o que deve ser adicionado  
- **$\tilde{c}_t$** propõe nova informação  
- **$o_t$ (output gate)** decide o que deve ser exposto como estado oculto  

Isso permite que a rede:

- mantenha informações importantes por dezenas de passos  
- descarte informações irrelevantes  
- evite vanishing/exploding gradients  
- aprenda dependências longas (ex.: tradução, contexto, concordância)

---

### Pensamento aplicado

A LSTM resolve três limitações fundamentais da RNN:

1. **Memória curta**  
   RNN simples não consegue manter informação por muitos passos.  
   A LSTM mantém $c_t$ estável graças ao gate de esquecimento.

2. **Vanishing gradients**  
   O gradiente flui através de $c_t$ sem ser multiplicado repetidamente por $U$.  
   Isso preserva gradientes ao longo do tempo.

3. **Controle explícito de fluxo**  
   Gates permitem que a rede aprenda quando lembrar e quando esquecer.

Por isso, LSTM dominou NLP por mais de uma década, até a chegada dos transformers.

---

### Conclusão

A LSTM é uma RNN com gates que controlam memória e gradiente.  
Ela resolve vanishing/exploding gradients e permite capturar dependências longas, tornando-se a base de modelos seq2seq e muitos sistemas clássicos de NLP.




<br><br>



<a id="seq2seq"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Seq2Seq | Codificação e Decodificação
=============================

Modelos recorrentes (RNN/LSTM) conseguem processar sequências, mas ainda não resolvem tarefas onde **entrada e saída têm comprimentos diferentes**, como:

- tradução automática  
- sumarização  
- resposta a perguntas  
- geração condicionada  

Para isso, usamos a arquitetura **Seq2Seq (Sequence‑to‑Sequence)**, composta por:

1. **Codificador (encoder)**  
2. **Decodificador (decoder)**  

O encoder comprime a entrada em um vetor de contexto.  
O decoder expande esse vetor em uma nova sequência.

---

### Codificador (Encoder)

O encoder lê a frase de entrada palavra por palavra.  
Usando uma RNN/LSTM genérica:

$$
s_t = f(Wx_t + U s_{t-1} + b)
$$

Após o último passo, o estado final:

$$
s_T
$$

é o **vetor de contexto** — uma representação comprimida da frase inteira.

---

### Decodificador (Decoder)

O decoder recebe o vetor de contexto e começa a gerar a saída, passo a passo.

Ele usa:

- a palavra anterior gerada $y_{t-1}$  
- o estado anterior $s'_{t-1}$  
- o contexto $s_T$  

para produzir:

$$
s'_t = f(W' y_{t-1} + U' s'_{t-1} + C s_T + b')
$$

A predição da próxima palavra é:

$$
p_t = \text{softmax}(V' s'_t + c')
$$

A palavra gerada é amostrada de $p_t$.

---

### Teacher Forcing

Durante o treinamento, o decoder recebe a **palavra correta** do passo anterior, não a palavra gerada.  
Isso acelera o aprendizado e estabiliza a rede.

---

### Pensamento aplicado

Seq2Seq resolve três limitações fundamentais das RNNs simples:

1. **Comprimentos diferentes**  
   O encoder resume a entrada em um vetor fixo, independentemente do tamanho.

2. **Dependências longas**  
   O vetor de contexto carrega informação da frase inteira.

3. **Geração condicionada**  
   A saída depende tanto do histórico quanto da entrada completa.

Por outro lado, comprimir toda a frase em um único vetor pode ser um gargalo — motivando mecanismos de atenção (transformers).

---

### Conclusão

Seq2Seq é a arquitetura clássica para mapeamento entre sequências.  
O encoder comprime a entrada e o decoder gera a saída, guiado pelo vetor de contexto.



<br><br>




<a id="prob_sentencas"></a>
[$\Uparrow$ Índice](#indice)

=============================  
Probabilidade de Sentenças
=============================

Modelos de linguagem atribuem uma **probabilidade** a uma sequência de palavras.  
Se a sentença é:

$$
w_1, w_2, \dots, w_T
$$

então a probabilidade conjunta é:

$$
P(w_1, w_2, \dots, w_T)
= \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})
$$

Essa é a regra da cadeia da probabilidade.

---

### Modelos n‑gram (Markov)

Modelos n‑gram aproximam:

$$
P(w_t \mid w_1, \dots, w_{t-1})
\approx
P(w_t \mid w_{t-n}, \dots, w_{t-1})
$$

Por exemplo, bigrama:

$$
P(w_1, \dots, w_T)
= \prod_{t=1}^T P(w_t \mid w_{t-1})
$$

---

### Redes neurais (feedforward, RNN, LSTM)

Modelos neurais aprendem a estimar:

$$
P(w_t \mid w_1, \dots, w_{t-1})
$$

usando uma rede que produz logits e aplica softmax:

$$
p_t = \text{softmax}(z_t)
$$

onde:

- $p_t$ é a distribuição sobre o vocabulário  
- $p_t[w_t]$ é a probabilidade da palavra correta  

A probabilidade da sentença é:

$$
P(w_1, \dots, w_T)
= \prod_{t=1}^T p_t[w_t]
$$

---

### Log‑probabilidade

Para evitar underflow, usamos log‑probabilidade:

$$
\log P(w_1, \dots, w_T)
= \sum_{t=1}^T \log p_t[w_t]
$$

Isso é o que modelos realmente otimizam.

---

### Pensamento aplicado

Probabilidade de sentenças é central em modelos de linguagem porque:

1. **Define o objetivo de treinamento**  
   Maximizar a probabilidade da sentença correta.

2. **Permite comparar modelos**  
   Modelos melhores atribuem maior probabilidade a sentenças naturais.

3. **Permite geração**  
   A próxima palavra é amostrada de $p_t$.

4. **Conecta todas as arquiteturas**  
   Markov, feedforward, RNN, LSTM, seq2seq — todos produzem distribuições via softmax.

A diferença entre modelos está **na forma de calcular**  
$P(w_t \mid \text{histórico})$ não na forma de combinar as probabilidades.

---

### Conclusão

A probabilidade de uma sentença é o produto das probabilidades condicionais de cada palavra.  
Modelos neurais estimam essas probabilidades via softmax, e o treinamento maximiza a log‑probabilidade da sequência correta.




<br><br>



<a id="mle_ngram"></a>
[$\Uparrow$ Índice](#indice)

=============================  
MLE para n‑gram
=============================

Modelos n‑gram estimam a probabilidade de uma palavra condicionada às últimas $n-1$ palavras.  
A forma geral é:

$$
P(w_t \mid w_{t-n+1}, \dots, w_{t-1})
$$

Para estimar essas probabilidades, usamos **MLE (Maximum Likelihood Estimation)**.

---

### MLE: ideia central

MLE escolhe os parâmetros que **maximizam a probabilidade dos dados observados**.

Para n‑gram, isso significa:

> “A probabilidade de uma palavra é proporcional ao número de vezes que ela aparece após o contexto.”

---

### MLE para bigramas

Para bigramas:

$$
P(w_t \mid w_{t-1})
= \frac{\text{contagem}(w_{t-1}, w_t)}
       {\text{contagem}(w_{t-1})}
$$

Ou seja:

- numerador: quantas vezes o par aparece  
- denominador: quantas vezes o contexto aparece  

---

### MLE para trigramas

Para trigramas:

$$
P(w_t \mid w_{t-2}, w_{t-1})
= \frac{\text{contagem}(w_{t-2}, w_{t-1}, w_t)}
       {\text{contagem}(w_{t-2}, w_{t-1})}
$$

Mesma lógica: frequência relativa.

---

### MLE geral para n‑gram

$$
P(w_t \mid w_{t-n+1}, \dots, w_{t-1})
= \frac{\text{contagem}(w_{t-n+1}, \dots, w_{t-1}, w_t)}
       {\text{contagem}(w_{t-n+1}, \dots, w_{t-1})}
$$

---

### Por que MLE funciona bem?

Porque MLE é:

- simples  
- computacionalmente barato  
- estatisticamente consistente  
- alinhado com a regra da cadeia da probabilidade  

Mas ele tem limitações:

- **zero counts** → probabilidade zero  
- **dados escassos** → estimativas ruins  
- **explosão de estados** conforme n cresce  

Esses problemas motivam:

- smoothing (Laplace, Kneser‑Ney)  
- modelos neurais (feedforward, RNN, LSTM)

---

### Pensamento aplicado

MLE para n‑gram é a base histórica dos modelos de linguagem:

1. **Define como estimar probabilidades a partir de dados.**  
2. **Mostra a relação entre contagem e probabilidade.**  
3. **Expõe as limitações que motivaram redes neurais.**  
4. **Conecta estatística clássica com modelos modernos.**

Sem MLE, não existe n‑gram.  
Sem n‑gram, não existe a transição para redes neurais.

---

### Conclusão

MLE para n‑gram estima probabilidades como frequências relativas.  
É simples, eficiente e historicamente fundamental, mas sofre com escassez de dados e motivou a evolução para modelos neurais.




<br><br>




<a id="modelos_generativos"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Modelos Generativos  
=============================

Modelos generativos descrevem **como os dados são produzidos**.  
Eles assumem que cada exemplo observado $x$ é gerado por um processo probabilístico envolvendo:

1. uma variável latente (como a classe ou o cluster),  
2. uma distribuição sobre essa variável,  
3. uma distribuição sobre os dados condicionada à variável latente.

Formalmente, um modelo generativo define a distribuição conjunta:

$$
p(x, y) = p(y)\, p(x \mid y)
$$

onde:

- $p(y)$ é a distribuição das classes ou clusters,  
- $p(x \mid y)$ é a distribuição dos dados dentro de cada classe.

---

### Intuição

Um modelo generativo conta uma história:

> “Para gerar um exemplo $x$, primeiro escolhemos uma classe $y$ segundo $p(y)$,  
> depois geramos $x$ segundo $p(x \mid y)$.”

Essa história permite:

- classificação probabilística,  
- clustering não supervisionado,  
- inferência de variáveis latentes,  
- geração de novos exemplos.

---

### Classificação via Regra de Bayes

A partir da distribuição conjunta, obtemos a distribuição posterior:

$$
p(y \mid x)
= \frac{p(y)\, p(x \mid y)}{p(x)}
$$

onde:

$$
p(x) = \sum_{j=1}^K p(j)\, p(x \mid j)
$$

Essa posterior é usada para classificação e para responsabilidades em modelos de mistura.

---

### Exemplos de Modelos Generativos

- Mistura de Gaussianas (GMM)  
- Mistura de Bernoulli  
- Mistura Multinomial (Bag‑of‑Words)  
- Naive Bayes  
- PCA Probabilístico  
- Mixture of Experts  
- Modelos modernos (VAEs, Diffusion Models)

---

### Conexão com Clustering

Em modelos de mistura, cada componente representa um cluster.  
A responsabilidade posterior é:

$$
p(j \mid x)
= \frac{p_j\, p(x \mid j)}{\sum_{\ell=1}^K p_\ell\, p(x \mid \ell)}
$$

Ela indica a probabilidade de o ponto pertencer ao cluster $j$.

Essa é a base do algoritmo EM.

---

### Conclusão

Modelos generativos são fundamentais porque:

- descrevem como os dados são criados,  
- permitem inferir classes e variáveis latentes,  
- conectam probabilidade, otimização e aprendizado,  
- são a base para GMM, EM e modelos generativos modernos.

Eles iniciam a estrutura da Unit 4.




<br><br>




<a id="misturas_distribuicoes"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Misturas de Distribuições  
=============================

Modelos de mistura representam dados como combinações de múltiplas distribuições.  
Eles assumem que cada exemplo observado $x$ foi gerado por **um entre vários componentes**, cada um com sua própria distribuição.

Formalmente, uma mistura com $K$ componentes define:

$$
p(x) = \sum_{j=1}^K p_j \, p(x \mid j)
$$

onde:

- $p_j$ é o peso (ou probabilidade) do componente $j$,  
- $p(x \mid j)$ é a distribuição do componente $j$.

Os pesos satisfazem:

$$
p_j \ge 0, \qquad \sum_{j=1}^K p_j = 1
$$

---

### Intuição

Misturas são úteis quando os dados têm **subestruturas naturais**:

- diferentes grupos,  
- diferentes padrões,  
- diferentes regiões no espaço,  
- diferentes formas de gerar exemplos.

Cada componente da mistura representa um **cluster probabilístico**.

---

### Exemplo geral

Se cada componente é uma Gaussiana:

$$
p(x \mid j) = \mathcal{N}(x; \mu_j, \Sigma_j)
$$

então o modelo completo é uma **Mistura de Gaussianas (GMM)**:

$$
p(x) = \sum_{j=1}^K p_j \, \mathcal{N}(x; \mu_j, \Sigma_j)
$$

Esse é o modelo mais usado em clustering probabilístico.

---

### Variáveis Latentes

Introduzimos uma variável latente $Z$ indicando qual componente gerou o ponto:

$$
Z \in \{1, 2, \dots, K\}
$$

Com isso, o processo generativo é:

1. Escolher um componente $Z = j$ com probabilidade $p_j$.  
2. Gerar $x$ segundo $p(x \mid j)$.

A distribuição conjunta é:

$$
p(x, Z=j) = p_j \, p(x \mid j)
$$

E a marginal sobre $x$ é a mistura:

$$
p(x) = \sum_{j=1}^K p(x, Z=j)
$$

---

### Posterior (Responsabilidade)

A probabilidade de que o componente $j$ tenha gerado o ponto $x$ é:

$$
p(j \mid x)
= \frac{p_j \, p(x \mid j)}
       {\sum_{\ell=1}^K p_\ell \, p(x \mid \ell)}
$$

Essa quantidade é chamada de **responsabilidade** do componente $j$ sobre o ponto $x$.

Ela é central no algoritmo EM.

---

### Exemplos de Misturas

- Mistura de Gaussianas (GMM)  
- Mistura de Bernoulli (para imagens binárias)  
- Mistura Multinomial (para Bag‑of‑Words)  
- Mistura de Poisson (para contagens)  
- Mistura de Experts (para regressão e classificação)  

Cada uma modela diferentes tipos de dados.

---

### Conclusão

Misturas de distribuições permitem modelar dados complexos como combinações de padrões simples.  
Elas introduzem variáveis latentes, responsabilidades e uma estrutura probabilística que será usada no algoritmo EM.

Essa seção prepara o terreno para a próxima: **Mistura de Gaussianas (GMM)**.


<br><br>


<a id="gmm"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Mistura de Gaussianas (GMM)  
=============================

A Mistura de Gaussianas (GMM — *Gaussian Mixture Model*) é o modelo de mistura mais importante da Unit 4.  
Ela assume que os dados foram gerados por **K distribuições Gaussianas**, cada uma representando um cluster probabilístico.

Formalmente, o modelo define:

$$
p(x) = \sum_{j=1}^K p_j \, \mathcal{N}(x; \mu_j, \Sigma_j)
$$

onde:

- $p_j$ são os pesos da mistura (probabilidades dos clusters),  
- $\mu_j$ são as médias dos clusters,  
- $\Sigma_j$ são as matrizes de covariância,  
- $\mathcal{N}(x; \mu_j, \Sigma_j)$ é a densidade Gaussiana multivariada.

Os pesos satisfazem:

$$
p_j \ge 0, \qquad \sum_{j=1}^K p_j = 1.
$$

---

### Intuição

Cada Gaussiana representa um **grupo natural** nos dados:

- um padrão,  
- uma forma,  
- uma região no espaço,  
- uma estrutura estatística.

O modelo não diz explicitamente qual ponto pertence a qual cluster.  
Em vez disso, ele calcula **probabilidades** de pertencimento.

---

### Processo Generativo

O GMM assume que cada ponto $x$ é gerado assim:

1. Escolher um cluster $Z = j$ com probabilidade $p_j$.  
2. Gerar $x$ segundo a Gaussiana do cluster:

$$
x \sim \mathcal{N}(\mu_j, \Sigma_j).
$$

A distribuição conjunta é:

$$
p(x, Z=j) = p_j \, \mathcal{N}(x; \mu_j, \Sigma_j).
$$

A marginal sobre $x$ é a mistura:

$$
p(x) = \sum_{j=1}^K p_j \, \mathcal{N}(x; \mu_j, \Sigma_j).
$$

---

### Posterior (Responsabilidade)

A probabilidade de que o cluster $j$ tenha gerado o ponto $x$ é:

$$
p(j \mid x)
= \frac{p_j \, \mathcal{N}(x; \mu_j, \Sigma_j)}
       {\sum_{\ell=1}^K p_\ell \, \mathcal{N}(x; \mu_\ell, \Sigma_\ell)}.
$$

Essa quantidade é chamada de **responsabilidade** do cluster $j$ sobre o ponto $x$.

Ela é central no algoritmo EM.

---

### Covariâncias

A matriz de covariância $\Sigma_j$ controla:

- a forma do cluster,  
- sua orientação,  
- sua dispersão.

Casos comuns:

- **Diagonal**: clusters elípticos alinhados aos eixos.  
- **Esférica**: variância igual em todas as direções.  
- **Completa**: forma arbitrária (mais expressivo, mais caro).

---

### Por que GMM é mais poderoso que K‑means?

- K‑means usa distâncias → GMM usa probabilidades.  
- K‑means assume clusters esféricos → GMM permite formas arbitrárias.  
- K‑means faz atribuição dura → GMM faz atribuição suave (responsabilidades).  
- K‑means é um caso especial do GMM com covariância esférica.

---

### Conclusão

A Mistura de Gaussianas é o modelo generativo mais usado para clustering probabilístico.  
Ela fornece:

- responsabilidades,  
- médias,  
- covariâncias,  
- pesos da mistura.

O próximo passo é aprender esses parâmetros via **algoritmo EM**.




<br><br>




<a id="#responsabilidades_posterior"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Responsabilidades e Posterior  
=============================

Em uma Mistura de Gaussianas (GMM), cada ponto observado $x$ pode ter vindo de qualquer um dos $K$ componentes da mistura.  
Como não sabemos qual componente gerou cada ponto, introduzimos uma **variável latente discreta**:

$$
z \in \{1, 2, \dots, K\}
$$

que indica qual Gaussiana gerou o dado.

O algoritmo EM precisa calcular, para cada ponto, **a probabilidade de ele ter sido gerado por cada componente**.  
Essas probabilidades são chamadas de **responsabilidades**.

---

### Variável Latente e Posterior

A variável latente $z$ tem distribuição:

$$
p(z = j) = \pi_j,
$$

onde $\pi_j$ é o peso da mistura.

A distribuição conjunta é:

$$
p(x, z = j)
= \pi_j \, \mathcal{N}(x; \mu_j, \Sigma_j).
$$

O que queremos é o **posterior**:

$$
p(z = j \mid x)
$$

que diz: *dado o ponto $x$, qual é a probabilidade de ele ter vindo do componente $j$?*

---

### Responsabilidades

As responsabilidades são definidas como:

$$
\gamma_{j}(x)
=
p(z = j \mid x).
$$

Usando Bayes:

$$
\gamma_{j}(x)
=
\frac{
\pi_j \, \mathcal{N}(x; \mu_j, \Sigma_j)
}{
\sum_{\ell=1}^K
\pi_\ell \, \mathcal{N}(x; \mu_\ell, \Sigma_\ell)
}.
$$

Essa equação é o **Passo E** do EM para GMM.

---

### Intuição

As responsabilidades funcionam como:

> “Para cada ponto, calcule quanto cada Gaussiana é responsável por explicá‑lo.”

Se o ponto está perto da média $\mu_j$, a responsabilidade $\gamma_j$ será alta.  
Se está longe, será baixa.

---

### Propriedades das Responsabilidades

1. **São probabilidades**  
   $$
   0 \le \gamma_j(x) \le 1
   $$

2. **Somam 1**  
   $$
   \sum_{j=1}^K \gamma_j(x) = 1
   $$

3. **Dependem de todos os parâmetros**  
   - pesos $\pi_j$  
   - médias $\mu_j$  
   - covariâncias $\Sigma_j$

4. **São “soft assignments”**  
   Diferente do K‑means, que faz atribuição dura (hard clustering),  
   o GMM faz atribuição suave (soft clustering).

---

### Papel das Responsabilidades no EM

As responsabilidades são usadas no Passo M para atualizar:

- **pesos da mistura**  
  $$
  \pi_j^{\text{novo}}
  =
  \frac{1}{n}
  \sum_{i=1}^n \gamma_{j}(x^{(i)})
  $$

- **médias**  
  $$
  \mu_j^{\text{novo}}
  =
  \frac{
    \sum_{i=1}^n \gamma_j(x^{(i)}) x^{(i)}
  }{
    \sum_{i=1}^n \gamma_j(x^{(i)})
  }
  $$

- **covariâncias**  
  $$
  \Sigma_j^{\text{novo}}
  =
  \frac{
    \sum_{i=1}^n
    \gamma_j(x^{(i)})
    (x^{(i)} - \mu_j^{\text{novo}})
    (x^{(i)} - \mu_j^{\text{novo}})^\top
  }{
    \sum_{i=1}^n \gamma_j(x^{(i)})
  }.
  $$

---

### Interpretação Geométrica

Cada ponto contribui para cada Gaussiana de forma proporcional à responsabilidade.

- Se $\gamma_j(x)$ é alta → ponto está perto da Gaussiana $j$.  
- Se $\gamma_j(x)$ é baixa → ponto está longe.  
- Se várias $\gamma_j(x)$ são altas → ponto está em uma região de sobreposição.

Isso torna o GMM um **clustering suave**, diferente do K‑means.

---

### Conclusão

As responsabilidades são o coração do Passo E do EM para GMM.  
Elas representam o posterior $p(z \mid x)$ e permitem:

- inferir a variável latente,  
- atualizar os parâmetros da mistura,  
- realizar clustering probabilístico,  
- interpretar regiões de sobreposição.

Com isso, o item 4 do índice está completo.





<br><br>




<a id="intuicao_em"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Intuição do EM (Expectation–Maximization)  
=============================

O algoritmo EM é o método fundamental para treinar modelos de mistura, como GMM.  
Ele resolve problemas onde:

- há **variáveis latentes** (como o cluster $Z$),  
- a distribuição completa $p(x, Z)$ é conhecida,  
- mas a distribuição marginal $p(x)$ é difícil de otimizar diretamente.

O EM alterna entre **inferir as variáveis latentes** e **atualizar os parâmetros** do modelo.

---

### O problema central

Queremos maximizar a log‑verossimilhança dos dados:

$$
\log p(x^{(1)}, \dots, x^{(n)} \mid \theta)
$$

Mas, em modelos de mistura:

$$
p(x^{(i)} \mid \theta)
= \sum_{j=1}^K p_j \, p(x^{(i)} \mid j)
$$

Essa soma dentro do log torna a otimização difícil.

O EM resolve isso introduzindo as responsabilidades:

$$
\gamma_{j,i} = p(j \mid x^{(i)}, \theta)
$$

que representam a probabilidade de o ponto $x^{(i)}$ ter sido gerado pelo cluster $j$.

---

### Intuição do ciclo EM

O EM alterna entre dois passos:

1. **Passo E (Expectation)**  
   Estimar as responsabilidades $\gamma_{j,i}$ usando os parâmetros atuais.

2. **Passo M (Maximization)**  
   Atualizar os parâmetros $\theta$ usando as responsabilidades como “pesos”.

O ciclo é:

> **Inferir clusters → atualizar parâmetros → repetir.**

---

### Analogia intuitiva

Imagine que você tem vários grupos de dados misturados, mas não sabe qual ponto pertence a qual grupo.

O EM faz:

- **Passo E:** “Dado o modelo atual, qual é a probabilidade de cada ponto pertencer a cada cluster?”  
- **Passo M:** “Dado essas probabilidades, quais são as melhores médias, covariâncias e pesos?”

É como alternar entre:

- **adivinhar a estrutura oculta**,  
- **melhorar o modelo**,  
- **adivinhar novamente**,  
- **melhorar novamente**.

---

### Conexão com variáveis latentes

O EM funciona porque transforma um problema difícil:

$$
\log \sum_j p_j\, p(x \mid j)
$$

em um problema fácil usando a distribuição completa:

$$
\log p(x, Z=j)
$$

A variável latente $Z$ torna a otimização mais simples.

---

### Por que EM funciona tão bem?

- Ele **nunca diminui** a verossimilhança.  
- Ele lida naturalmente com variáveis latentes.  
- Ele transforma atribuição dura (K‑means) em atribuição suave.  
- Ele é robusto e funciona em alta dimensão.  
- Ele é a base de muitos modelos modernos (VAEs, mixture‑of‑experts, etc.).

---

### Conclusão

A intuição do EM é simples:

1. **Inferir** a estrutura oculta (responsabilidades).  
2. **Atualizar** os parâmetros do modelo.  
3. **Repetir** até convergir.

Ele é o algoritmo que torna possível treinar GMM e outros modelos generativos com variáveis latentes.





<br><br>




<a id="passo_e"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Passo E — Expectation  
=============================

O **Passo E** do algoritmo EM calcula as **responsabilidades** — isto é, a probabilidade de cada componente da mistura ter gerado cada ponto de dados.

Ele corresponde à etapa de **inferência das variáveis latentes**.

---

### Objetivo do Passo E

Dado o conjunto de parâmetros atuais:

- pesos da mistura $p_j$,  
- médias $\mu_j$,  
- covariâncias $\Sigma_j$,

queremos calcular:

$$
\gamma_{j,i} = p(j \mid x^{(i)}, \theta)
$$

que representa a responsabilidade do cluster $j$ sobre o ponto $x^{(i)}$.

---

### Fórmula geral

Para modelos de mistura arbitrários:

$$
\gamma_{j,i}
= \frac{p_j \, p(x^{(i)} \mid j)}
       {\sum_{\ell=1}^K p_\ell \, p(x^{(i)} \mid \ell)}.
$$

Essa é a aplicação direta da **Regra de Bayes**.

---

### Caso específico: Mistura de Gaussianas (GMM)

Quando cada componente é uma Gaussiana:

$$
p(x^{(i)} \mid j)
= \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j),
$$

então:

$$
\gamma_{j,i}
= \frac{
p_j \, \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)
}{
\sum_{\ell=1}^K
p_\ell \, \mathcal{N}(x^{(i)}; \mu_\ell, \Sigma_\ell)
}.
$$

---

### Interpretação geométrica

A responsabilidade $\gamma_{j,i}$ mede:

- **quão perto** o ponto $x^{(i)}$ está da média $\mu_j$,  
- **quão compatível** ele é com a forma da covariância $\Sigma_j$,  
- **quão provável** é o cluster $j$ segundo o peso $p_j$.

Clusters com:

- maior peso $p_j$,  
- maior densidade no ponto $x^{(i)}$,

recebem maior responsabilidade.

---

### Atribuição suave (soft assignment)

Diferente do K‑means, onde cada ponto pertence **exatamente** a um cluster, no Passo E:

- cada ponto pertence **parcialmente** a todos os clusters,  
- com pesos probabilísticos,  
- que somam 1:

$$
\sum_{j=1}^K \gamma_{j,i} = 1.
$$

Isso torna o EM mais estável e mais expressivo que K‑means.

---

### Papel do Passo E no ciclo EM

O Passo E prepara o terreno para o Passo M:

- ele calcula **como cada ponto deve contribuir** para cada cluster,  
- e essas contribuições serão usadas para atualizar médias, covariâncias e pesos.

O Passo E é, portanto, a etapa de **inferência**.

---

### Conclusão

O Passo E calcula as responsabilidades $\gamma_{j,i}$ usando os parâmetros atuais.  
Ele transforma o problema de mistura em um problema com “rótulos suaves”, permitindo que o Passo M atualize os parâmetros de forma eficiente.

O próximo passo é o **Passo M — Maximization**.




<br><br>




<a id="passo_m"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Passo M — Maximization  
=============================

O **Passo M** do algoritmo EM atualiza os parâmetros do modelo usando as responsabilidades calculadas no Passo E.

Ele corresponde à etapa de **maximização da verossimilhança completa esperada**.

---

### Objetivo do Passo M

Dado:

- as responsabilidades $\gamma_{j,i}$,  
- os dados $x^{(i)}$,  
- o número de componentes $K$,

queremos atualizar os parâmetros:

- pesos da mistura $p_j$,  
- médias $\mu_j$,  
- covariâncias $\Sigma_j$.

Esses parâmetros são escolhidos para **maximizar** a função:

$$
Q(\theta) = 
\sum_{i=1}^n \sum_{j=1}^K 
\gamma_{j,i} \, \log p(x^{(i)}, j \mid \theta)
$$

que é a verossimilhança completa ponderada pelas responsabilidades.

---

### Atualização dos Pesos da Mistura

O novo peso do cluster $j$ é proporcional ao total de responsabilidade que ele recebe:

$$
p_j^{\text{novo}}
= \frac{1}{n} \sum_{i=1}^n \gamma_{j,i}.
$$

Isso garante:

$$
\sum_{j=1}^K p_j^{\text{novo}} = 1.
$$

---

### Atualização das Médias

A nova média do cluster $j$ é a média ponderada dos pontos, usando as responsabilidades como pesos:

$$
\mu_j^{\text{novo}}
= \frac{
\sum_{i=1}^n \gamma_{j,i} \, x^{(i)}
}{
\sum_{i=1}^n \gamma_{j,i}
}.
$$

Essa fórmula é idêntica à média de um conjunto de pontos, mas com pesos suaves.

---

### Atualização das Covariâncias

A nova covariância do cluster $j$ é:

$$
\Sigma_j^{\text{novo}}
=
\frac{
\sum_{i=1}^n
\gamma_{j,i}
\left(x^{(i)} - \mu_j^{\text{novo}}\right)
\left(x^{(i)} - \mu_j^{\text{novo}}\right)^\top
}{
\sum_{i=1}^n \gamma_{j,i}
}.
$$

Casos especiais:

- **Covariância esférica:**  
  $$ \Sigma_j = \sigma_j^2 I $$
- **Covariância diagonal:**  
  apenas variâncias individuais.
- **Covariância completa:**  
  forma arbitrária (mais expressiva, mais cara).

---

### Interpretação geométrica

O Passo M faz:

- **pesos:** quanto cada cluster é “grande”,  
- **médias:** onde cada cluster está localizado,  
- **covariâncias:** qual é a forma e orientação de cada cluster.

Ele usa as responsabilidades como “rótulos suaves”, transformando o problema em uma versão ponderada de estimar médias e covariâncias.

---

### Papel do Passo M no ciclo EM

O Passo M usa as responsabilidades para:

- recalcular os parâmetros,  
- melhorar o modelo,  
- preparar o próximo Passo E.

O ciclo completo é:

1. **Passo E:** inferir responsabilidades.  
2. **Passo M:** atualizar parâmetros.  
3. Repetir até convergir.

---

### Conclusão

O Passo M atualiza os parâmetros do modelo usando as responsabilidades do Passo E.  
Ele transforma o problema de maximizar a verossimilhança em atualizações fechadas para pesos, médias e covariâncias.

O próximo tópico é **Convergência do EM**.




<br><br>




<a id="convergencia_em"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Convergência do EM  
=============================

O algoritmo EM possui uma propriedade fundamental que o torna extremamente útil em modelos de mistura:  
**a cada iteração, a verossimilhança dos dados nunca diminui**.

Isso significa que o EM sempre caminha na direção de um modelo melhor — embora não necessariamente o melhor possível.

---

### Propriedade central

Seja $\theta^{(t)}$ o conjunto de parâmetros na iteração $t$.  
O EM garante que:

$$
\log p(X \mid \theta^{(t+1)}) 
\ge 
\log p(X \mid \theta^{(t)})
$$

onde $X = \{x^{(1)}, \dots, x^{(n)}\}$ é o conjunto de dados.

Essa é a propriedade de **monotonicidade da verossimilhança**.

---

### Por que isso acontece?

O EM alterna entre:

1. **Passo E:** constrói uma distribuição auxiliar (responsabilidades).  
2. **Passo M:** maximiza uma função que é uma *lower bound* da verossimilhança verdadeira.

Essa lower bound é chamada de:

$$
Q(\theta, \theta^{(t)})
=
\sum_{i=1}^n \sum_{j=1}^K 
\gamma_{j,i}^{(t)} \,
\log p(x^{(i)}, j \mid \theta)
$$

O Passo M escolhe $\theta^{(t+1)}$ que **maximiza** essa função.  
Como $Q$ é uma bound da verossimilhança, maximizar $Q$ garante que a verossimilhança não diminui.

---

### O que o EM garante?

- **Monotonicidade:** a verossimilhança nunca diminui.  
- **Convergência:** o algoritmo converge para um ponto fixo.  
- **Estabilidade:** pequenas mudanças nos dados não causam explosões numéricas.  
- **Robustez:** funciona bem mesmo em alta dimensão.

---

### O que o EM *não* garante?

Apesar de sempre aumentar a verossimilhança, o EM **não garante**:

- convergência para o **máximo global**,  
- convergência para o **melhor modelo possível**,  
- convergência rápida quando os clusters se sobrepõem muito.

O EM pode ficar preso em **máximos locais**.

---

### Máximos locais

A função de verossimilhança de uma mistura é altamente não convexa:

$$
\log p(X \mid \theta)
= 
\sum_{i=1}^n 
\log \left(
\sum_{j=1}^K p_j \, p(x^{(i)} \mid j)
\right)
$$

Essa função possui:

- múltiplos picos,  
- vales,  
- regiões planas.

O EM pode convergir para qualquer máximo local dependendo da inicialização.

---

### Importância da inicialização

Como o EM pode convergir para diferentes soluções, a inicialização é crítica:

- usar K‑means para inicializar as médias,  
- inicializar covariâncias com variâncias razoáveis,  
- evitar pesos muito pequenos,  
- rodar EM várias vezes com diferentes seeds.

Boas inicializações → convergência mais rápida e melhores soluções.

---

### Critérios de parada

O EM é geralmente interrompido quando:

1. a mudança na verossimilhança é pequena:

$$
\log p(X \mid \theta^{(t+1)}) 
- 
\log p(X \mid \theta^{(t)})
< \varepsilon
$$

2. ou quando as mudanças nos parâmetros são pequenas:

$$
\|\theta^{(t+1)} - \theta^{(t)}\| < \delta
$$

3. ou após um número máximo de iterações.

---

### Conclusão

O EM sempre aumenta a verossimilhança e converge para um ponto fixo.  




<br><br>



<a id="clustering_probabilistico"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Clustering Probabilístico  
=============================

Clustering probabilístico é a abordagem que utiliza **modelos generativos** para agrupar dados.  
Diferente de métodos como K‑means, que fazem atribuição dura (cada ponto pertence a um único cluster), o clustering probabilístico faz **atribuição suave**, usando probabilidades.

A ideia central é modelar os dados como sendo gerados por uma mistura:

$$
p(x) = \sum_{j=1}^K p_j \, p(x \mid j)
$$

onde cada componente representa um cluster.

---

### Atribuição suave (soft assignment)

Em clustering probabilístico, cada ponto $x^{(i)}$ recebe uma distribuição sobre os clusters:

$$
\gamma_{j,i} = p(j \mid x^{(i)})
$$

com:

$$
\sum_{j=1}^K \gamma_{j,i} = 1.
$$

Isso significa:

- o ponto pertence parcialmente a todos os clusters,  
- a responsabilidade $\gamma_{j,i}$ indica o grau de pertencimento,  
- clusters podem se sobrepor,  
- a incerteza é explicitamente modelada.

---

### Comparação com K‑means

| K‑means | Clustering Probabilístico |
|--------|----------------------------|
| atribuição dura | atribuição suave |
| distância euclidiana | densidade probabilística |
| clusters esféricos | clusters com formas arbitrárias |
| sem variáveis latentes | com variáveis latentes |
| sem pesos | pesos da mistura $p_j$ |

K‑means é um caso especial de clustering probabilístico com:

- covariâncias esféricas,  
- responsabilidades binárias (0 ou 1),  
- maximização via distância.

---

### Probabilidade posterior

A responsabilidade é dada por:

$$
\gamma_{j,i}
= \frac{
p_j \, p(x^{(i)} \mid j)
}{
\sum_{\ell=1}^K p_\ell \, p(x^{(i)} \mid \ell)
}.
$$

Essa é a probabilidade de o cluster $j$ ter gerado o ponto $x^{(i)}$.

Ela é usada para:

- atualizar médias,  
- atualizar covariâncias,  
- atualizar pesos da mistura.

---

### Interpretação geométrica

Cada cluster é uma distribuição:

- sua média $\mu_j$ define o centro,  
- sua covariância $\Sigma_j$ define a forma,  
- seu peso $p_j$ define o tamanho.

A responsabilidade $\gamma_{j,i}$ mede:

- quão perto o ponto está da média,  
- quão compatível ele é com a forma do cluster,  
- quão provável é o cluster segundo o peso.

---

### Vantagens do clustering probabilístico

1. **Modela incerteza**  
   Cada ponto tem uma distribuição sobre clusters.

2. **Clusters com formas complexas**  
   Covariâncias completas permitem elipses arbitrárias.

3. **Base probabilística sólida**  
   Usa densidades, likelihood e Bayes.

4. **Integração com EM**  
   Treinamento eficiente e com garantias de monotonicidade.

5. **Aplicações em alta dimensão**  
   Funciona bem em imagens, texto, áudio, embeddings.

---

### Aplicações clássicas

- agrupamento de dígitos (MNIST),  
- segmentação de imagens,  
- agrupamento de documentos,  
- compressão de dados,  
- modelagem de áudio,  
- detecção de anomalias.

---

### Conclusão

Clustering probabilístico é uma abordagem mais rica e expressiva que K‑means.  
Ele usa modelos generativos, atribuição suave e inferência probabilística para descobrir estruturas naturais nos dados.

A próxima seção é **Classificação com GMM**.




<br><br>




<a id="classificacao_gmm"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Classificação com GMM  
=============================

A Mistura de Gaussianas (GMM) não é apenas um método de clustering.  
Ela também pode ser usada como um **classificador probabilístico**, especialmente quando cada componente da mistura representa uma classe.

Essa abordagem é chamada de **classificação generativa**, pois utiliza o modelo:

$$
p(x, y) = p(y)\, p(x \mid y)
$$

para inferir a classe mais provável de um novo ponto.

---

### Modelo generativo para classificação

Suponha que temos $K$ classes.  
Para cada classe $j$, treinamos um modelo GMM que define:

- um peso da classe $p(y=j)$,  
- uma distribuição condicional $p(x \mid y=j)$.

A regra de decisão é baseada na probabilidade posterior:

$$
p(y=j \mid x)
= \frac{p(y=j)\, p(x \mid y=j)}
       {\sum_{\ell=1}^K p(y=\ell)\, p(x \mid y=\ell)}.
$$

A classe escolhida é:

$$
\hat{y}(x) = \arg\max_j \, p(y=j \mid x).
$$

---

### Caso especial: uma Gaussiana por classe

Se cada classe é modelada por uma única Gaussiana:

$$
p(x \mid y=j) = \mathcal{N}(x; \mu_j, \Sigma_j),
$$

então:

$$
p(y=j \mid x)
=
\frac{
p_j \, \mathcal{N}(x; \mu_j, \Sigma_j)
}{
\sum_{\ell=1}^K p_\ell \, \mathcal{N}(x; \mu_\ell, \Sigma_\ell)
}.
$$

Essa é a mesma fórmula das responsabilidades, mas agora interpretada como **probabilidade de classe**.

---

### Interpretação geométrica

Cada classe é representada por uma distribuição Gaussiana:

- $\mu_j$ → centro da classe,  
- $\Sigma_j$ → forma e orientação,  
- $p_j$ → frequência da classe.

A classificação é feita comparando:

- quão provável é o ponto sob cada Gaussiana,  
- ponderado pelo peso da classe.

Isso produz uma fronteira de decisão **não linear**, diferente da SVM ou perceptron.

---

### Vantagens da classificação com GMM

1. **Modela a distribuição completa dos dados**  
   Não apenas a fronteira de decisão.

2. **Fronteiras não lineares**  
   Covariâncias completas produzem fronteiras curvas.

3. **Probabilidades calibradas**  
   A saída é uma distribuição sobre classes.

4. **Bom desempenho em alta dimensão**  
   Especialmente quando os dados são aproximadamente Gaussianos.

5. **Natural para dados multimodais**  
   Uma classe pode ter vários componentes (subgrupos).

---

### Comparação com classificadores discriminativos

| GMM (generativo) | SVM / Perceptron (discriminativo) |
|------------------|-----------------------------------|
| modela $p(x, y)$ | modela apenas $p(y \mid x)$ |
| produz probabilidades | produz margens |
| fronteiras não lineares | fronteiras lineares (sem kernel) |
| sensível à distribuição | focado na separação |
| pode gerar novos dados | não pode gerar |

Ambos têm vantagens dependendo da tarefa.

---

### Aplicação clássica: dígitos manuscritos (MNIST)

Para cada dígito $0$ a $9$:

- treina-se um GMM com $K$ componentes,  
- cada componente captura uma variação do dígito,  
- a classificação é feita via posterior.

Isso produz um classificador não supervisionado extremamente eficaz.

---

### Conclusão

Classificação com GMM utiliza o modelo generativo para inferir a classe mais provável de um ponto.  
Ela fornece probabilidades calibradas, fronteiras não lineares e uma interpretação estatística clara.

A próxima seção é **GMM para Imagens (MNIST)**.




<br><br>




<a id="gmm_mnist"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
GMM para Imagens (MNIST)  
=============================

A Mistura de Gaussianas (GMM) é um dos modelos probabilísticos mais eficazes para trabalhar com imagens de dígitos manuscritos, como o conjunto MNIST.  
Cada imagem é representada como um vetor de alta dimensão, e o GMM modela a distribuição desses vetores como uma mistura de Gaussianas.

---

### Representação das Imagens

Cada imagem MNIST tem:

- 28 × 28 pixels (ou 26 × 26, dependendo da versão),
- tons de cinza,
- um valor por pixel.

Ao achatar a imagem, obtemos um vetor:

$$
x \in \mathbb{R}^d,
\qquad d = 26 \times 26 = 676.
$$

Assim, cada imagem é um ponto em um espaço de 676 dimensões.

---

### Modelando Dígitos com GMM

Para cada dígito $y \in \{0, 1, \dots, 9\}$, podemos treinar um modelo GMM que captura:

- a forma média do dígito,
- sua variabilidade,
- suas deformações,
- suas diferentes maneiras de ser escrito.

O modelo para cada classe é:

$$
p(x \mid y=j)
= \sum_{k=1}^{K_j}
p_{j,k} \,
\mathcal{N}(x; \mu_{j,k}, \Sigma_{j,k}),
$$

onde:

- $K_j$ é o número de componentes para o dígito $j$,
- $\mu_{j,k}$ é a média do componente $k$ da classe $j$,
- $\Sigma_{j,k}$ é a covariância,
- $p_{j,k}$ é o peso do componente.

---

### Intuição

Cada componente da mistura representa uma **variação do dígito**:

- diferentes estilos de escrita,
- diferentes inclinações,
- diferentes espessuras,
- diferentes deformações.

Por exemplo, o dígito “2” pode ter:

- um traço mais curvo,
- um traço mais reto,
- uma base mais larga,
- uma escrita mais inclinada.

Cada variação é capturada por uma Gaussiana.

---

### Classificação com GMM

Para classificar uma nova imagem $x$, calculamos:

$$
p(y=j \mid x)
=
\frac{
p(y=j)\, p(x \mid y=j)
}{
\sum_{\ell=0}^9 p(y=\ell)\, p(x \mid y=\ell)
}.
$$

A classe escolhida é:

$$
\hat{y}(x) = \arg\max_j \, p(y=j \mid x).
$$

Essa abordagem é totalmente probabilística.

---

### Clustering com GMM

Também podemos aplicar GMM **sem rótulos**:

- usamos todos os 100.000 exemplos,
- definimos $K = 10$ clusters,
- cada cluster deve capturar um dígito.

O EM descobre:

- médias que parecem dígitos,
- covariâncias que capturam variações,
- pesos que refletem frequência dos dígitos.

Isso produz um classificador não supervisionado.

---

### Visualização das Médias

As médias $\mu_j$ aprendidas pelo GMM podem ser visualizadas como imagens:

- elas se parecem com versões “suavizadas” dos dígitos,
- são representações estatísticas da classe,
- capturam a forma média do dígito.

Essa é uma das propriedades mais bonitas do GMM aplicado a imagens.

---

### Covariâncias e Variabilidade

A covariância $\Sigma_j$ captura:

- onde o dígito varia mais,
- quais pixels têm maior correlação,
- quais regiões são mais instáveis.

Covariâncias completas revelam:

- padrões de escrita,
- inclinações,
- deformações estruturais.

---

### Vantagens do GMM em Imagens

1. **Modela variações naturais dos dígitos**  
   Cada componente captura um estilo.

2. **Classificação probabilística**  
   Produz $p(y \mid x)$, não apenas uma decisão.

3. **Clustering não supervisionado**  
   Descobre classes sem rótulos.

4. **Interpretação visual**  
   Médias e covariâncias podem ser visualizadas.

5. **Base para modelos modernos**  
   GMM é precursor de VAEs e modelos generativos.

---

### Conclusão

O GMM é uma ferramenta poderosa para modelar imagens de dígitos.  
Ele captura variações, produz classificações probabilísticas e permite clustering não supervisionado.  
Aplicado ao MNIST, ele revela estruturas naturais e padrões de escrita.

A próxima seção é **Comparação com K‑means**.




<br><br>




<a id="comparacao_kmeans"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Comparação com K‑means  
=============================

K‑means e GMM são dois dos métodos de clustering mais usados em aprendizado de máquina.  
Apesar de parecerem semelhantes, eles têm **fundamentos matemáticos diferentes**, produzem **resultados diferentes** e são adequados para **situações diferentes**.

---

### Ideia central de cada método

**K‑means:**  
Minimiza a distância quadrática entre pontos e centros:

$$
\min_{\{\mu_j\}} \sum_{i=1}^n \min_j \|x^{(i)} - \mu_j\|^2.
$$

**GMM:**  
Maximiza a verossimilhança de uma mistura de Gaussianas:

$$
p(x) = \sum_{j=1}^K p_j \, \mathcal{N}(x; \mu_j, \Sigma_j).
$$

---

### Atribuição de clusters

**K‑means:**  
Atribuição **dura** (hard assignment):

$$
\gamma_{j,i} \in \{0, 1\}.
$$

Cada ponto pertence a exatamente um cluster.

**GMM:**  
Atribuição **suave** (soft assignment):

$$
\gamma_{j,i} = p(j \mid x^{(i)}),
\qquad
\sum_{j=1}^K \gamma_{j,i} = 1.
$$

Cada ponto pertence parcialmente a todos os clusters.

---

### Forma dos clusters

**K‑means:**  
Clusters esféricos, iguais em todas as direções.

**GMM:**  
Clusters elípticos, com forma e orientação definidas pela covariância:

- esférica,  
- diagonal,  
- completa.

GMM é muito mais expressivo.

---

### Função objetivo

**K‑means:**  
Minimiza distâncias.

**GMM:**  
Maximiza densidades probabilísticas.

Isso significa que GMM leva em conta:

- variância,  
- correlação entre dimensões,  
- pesos dos clusters.

---

### Conexão matemática entre os dois

K‑means é um caso especial de GMM quando:

- $\Sigma_j = \sigma^2 I$ (covariância esférica),
- $\sigma^2$ é muito pequeno,
- responsabilidades são binárias.

Nesse limite, a densidade Gaussiana se comporta como:

$$
\mathcal{N}(x; \mu_j, \sigma^2 I)
\propto
\exp\left(-\frac{\|x - \mu_j\|^2}{2\sigma^2}\right),
$$

e maximizar a densidade equivale a minimizar a distância.

---

### Comparação lado a lado

| Aspecto | K‑means | GMM |
|--------|---------|-----|
| Atribuição | dura | suave |
| Forma dos clusters | esférica | elíptica |
| Covariância | não existe | diagonal / completa |
| Probabilidades | não | sim |
| Função objetivo | distância | verossimilhança |
| Interpretação | geométrica | estatística |
| Robustez | baixa | alta |
| Expressividade | limitada | alta |

---

### Quando usar cada um?

**Use K‑means quando:**

- você quer algo rápido,  
- clusters são aproximadamente esféricos,  
- não precisa de probabilidades,  
- dados têm baixa variabilidade.

**Use GMM quando:**

- clusters têm formas complexas,  
- você quer probabilidades de pertencimento,  
- dados têm correlações entre dimensões,  
- você quer um modelo generativo.

---

### Conclusão

K‑means é simples, rápido e útil para clusters esféricos.  
GMM é mais poderoso, probabilístico e expressivo, capaz de capturar formas complexas e incerteza.

K‑means é uma aproximação grosseira do GMM — e o EM é a versão probabilística e completa do processo de clustering.

A próxima seção é **Covariâncias Diagonais vs. Completas**.




<br><br>




<a id="covariancias"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Covariâncias Diagonais vs. Completas  
=============================

A matriz de covariância é o elemento que determina **a forma, orientação e dispersão** de cada componente da Mistura de Gaussianas (GMM).  
Ela é um dos fatores mais importantes para controlar a expressividade do modelo.

Cada componente $j$ possui uma covariância:

$$
\Sigma_j \in \mathbb{R}^{d \times d},
$$

onde $d$ é a dimensionalidade dos dados.

---

### Covariância Completa

A covariância completa permite capturar **correlações entre todas as dimensões**:

$$
\Sigma_j =
\begin{bmatrix}
\sigma_{11} & \sigma_{12} & \dots & \sigma_{1d} \\
\sigma_{21} & \sigma_{22} & \dots & \sigma_{2d} \\
\vdots      & \vdots      & \ddots & \vdots      \\
\sigma_{d1} & \sigma_{d2} & \dots & \sigma_{dd}
\end{bmatrix}.
$$

Ela produz clusters com formas elípticas arbitrárias.

#### Vantagens

- Captura correlações entre pixels ou features.  
- Permite formas complexas e orientações variadas.  
- É o modelo mais expressivo.

#### Desvantagens

- Custo computacional alto: $O(d^2)$ por componente.  
- Pode superajustar quando $d$ é grande.  
- Requer muito mais dados para estimar corretamente.

---

### Covariância Diagonal

A covariância diagonal assume que as dimensões são **independentes**:

$$
\Sigma_j =
\begin{bmatrix}
\sigma_{1}^2 & 0 & \dots & 0 \\
0 & \sigma_{2}^2 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & \sigma_{d}^2
\end{bmatrix}.
$$

Cada dimensão tem sua própria variância, mas não há correlação entre elas.

#### Vantagens

- Muito mais rápida: $O(d)$ por componente.  
- Estável em alta dimensão.  
- Menos propensa a overfitting.  
- Ideal para imagens e Bag‑of‑Words.

#### Desvantagens

- Não captura correlações entre pixels.  
- Clusters têm forma elíptica alinhada aos eixos.  
- Menos expressiva que a covariância completa.

---

### Covariância Esférica

Caso especial da diagonal:

$$
\Sigma_j = \sigma_j^2 I.
$$

Todas as dimensões têm a mesma variância.

#### Características

- Clusters são esferas perfeitas.  
- É o caso mais simples.  
- Aproxima K‑means.

---

### Comparação lado a lado

| Tipo de Covariância | Forma do Cluster | Correlações | Custo | Risco de Overfitting |
|---------------------|------------------|-------------|-------|----------------------|
| Completa | elipse arbitrária | sim | alto | alto |
| Diagonal | elipse alinhada aos eixos | não | médio | baixo |
| Esférica | esfera | não | baixo | muito baixo |

---

### Qual escolher?

**Covariância completa**  
Use quando:

- há muitos dados,  
- correlações são importantes,  
- clusters têm formas complexas.

**Covariância diagonal**  
Use quando:

- $d$ é grande (imagens, texto),  
- você quer velocidade e estabilidade,  
- correlações não são essenciais.

**Covariância esférica**  
Use quando:

- você quer algo simples e rápido,  
- clusters são aproximadamente esféricos,  
- está próximo de K‑means.

---

### Conclusão

A escolha da covariância define a expressividade do GMM:

- **completa** → máxima flexibilidade,  
- **diagonal** → equilíbrio entre expressividade e custo,  
- **esférica** → simplicidade e velocidade.

Essa escolha impacta diretamente a qualidade do clustering e da classificação.

A próxima seção é **Misturas de Bernoulli e Multinomial**.




<br><br>




<a id="misturas_bernoulli_multinomial"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Misturas de Bernoulli e Multinomial  
=============================

Nem todos os dados seguem distribuições Gaussianas.  
Para muitos tipos de dados — especialmente **imagens binárias**, **texto**, **contagens** e **Bag‑of‑Words** — modelos baseados em Bernoulli e Multinomial são mais adequados.

Modelos de mistura podem ser construídos sobre essas distribuições, permitindo clustering e classificação probabilística em domínios não contínuos.

---

### Mistura de Bernoulli

A mistura de Bernoulli é usada quando cada dimensão do vetor $x$ é **binária**:

$$
x \in \{0,1\}^d.
$$

Cada componente da mistura define uma distribuição Bernoulli independente para cada dimensão:

$$
p(x \mid j)
=
\prod_{k=1}^d 
\theta_{j,k}^{\, x_k}
(1 - \theta_{j,k})^{\, 1 - x_k},
$$

onde:

- $\theta_{j,k}$ é a probabilidade do pixel/feature $k$ ser 1 no cluster $j$.

A mistura completa é:

$$
p(x) = \sum_{j=1}^K p_j \, p(x \mid j).
$$

---

### Intuição

Cada cluster $j$ possui um vetor de probabilidades:

$$
\theta_j = (\theta_{j,1}, \dots, \theta_{j,d}),
$$

que representa a “imagem média binária” ou “padrão binário” do cluster.

Isso é útil para:

- imagens binárias (ex.: dígitos preto/branco),  
- dados booleanos,  
- presença/ausência de atributos.

---

### Atualização dos parâmetros (EM)

**Passo E:**  
Responsabilidades:

$$
\gamma_{j,i}
=
\frac{
p_j \, p(x^{(i)} \mid j)
}{
\sum_{\ell=1}^K p_\ell \, p(x^{(i)} \mid \ell)
}.
$$

**Passo M:**  
Atualização das probabilidades Bernoulli:

$$
\theta_{j,k}^{\text{novo}}
=
\frac{
\sum_{i=1}^n \gamma_{j,i} \, x^{(i)}_k
}{
\sum_{i=1}^n \gamma_{j,i}
}.
$$

Essa fórmula é idêntica à média ponderada dos valores binários.

---

### Mistura Multinomial

A mistura Multinomial é usada para dados de **contagem**, como Bag‑of‑Words:

- $x_k$ = número de vezes que a palavra $k$ aparece no documento,
- $x$ é um vetor de contagens,
- $\sum_k x_k$ é o tamanho do documento.

A distribuição Multinomial é:

$$
p(x \mid j)
=
\frac{(\sum_k x_k)!}{\prod_k x_k!}
\prod_{k=1}^d
\theta_{j,k}^{\, x_k},
$$

onde:

- $\theta_{j,k}$ é a probabilidade da palavra $k$ no cluster $j$,
- $\sum_k \theta_{j,k} = 1$.

A mistura completa é:

$$
p(x) = \sum_{j=1}^K p_j \, p(x \mid j).
$$

---

### Intuição

Cada cluster $j$ possui uma distribuição de palavras:

$$
\theta_j = (\theta_{j,1}, \dots, \theta_{j,d}),
$$

que representa um “tópico” ou “tema”.

Essa é a base de:

- clustering de documentos,  
- modelos de tópicos simples,  
- classificação de texto generativa.

---

### Atualização dos parâmetros (EM)

**Passo E:**  
Responsabilidades:

$$
\gamma_{j,i}
=
\frac{
p_j \, p(x^{(i)} \mid j)
}{
\sum_{\ell=1}^K p_\ell \, p(x^{(i)} \mid \ell)
}.
$$

**Passo M:**  
Atualização das probabilidades Multinomiais:

$$
\theta_{j,k}^{\text{novo}}
=
\frac{
\sum_{i=1}^n \gamma_{j,i} \, x^{(i)}_k
}{
\sum_{i=1}^n \gamma_{j,i} \sum_{k'} x^{(i)}_{k'}
}.
$$

Essa fórmula é a normalização das contagens ponderadas.

---

### Comparação entre Bernoulli e Multinomial

| Tipo | Dados | Interpretação | Aplicações |
|------|-------|----------------|------------|
| Bernoulli | binários | presença/ausência | imagens binárias, atributos booleanos |
| Multinomial | contagens | frequência | texto, Bag‑of‑Words, eventos |

---

### Conclusão

Misturas de Bernoulli e Multinomial permitem aplicar EM em domínios não contínuos:

- imagens binárias,  
- texto,  
- contagens,  
- atributos booleanos.

Elas ampliam o alcance dos modelos generativos além das Gaussianas.

A próxima seção é **EM para PCA Probabilístico**.




<br><br>




<a id="em_pca"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
EM para PCA Probabilístico  
=============================

O PCA clássico é um método geométrico baseado em autovetores da matriz de covariância.  
O **PCA Probabilístico (PPCA)** reformula o PCA como um **modelo generativo com variáveis latentes**, permitindo que o algoritmo EM seja usado para estimar seus parâmetros.

Essa visão probabilística conecta PCA a modelos de mistura, GMM, variáveis latentes e, mais tarde, a modelos como VAEs.

---

### Modelo Generativo do PPCA

O PPCA assume que cada ponto observado $x \in \mathbb{R}^d$ é gerado a partir de uma variável latente $z \in \mathbb{R}^q$ com $q < d$:

1. Variável latente:
   $$
   z \sim \mathcal{N}(0, I_q)
   $$

2. Geração do dado:
   $$
   x = W z + \mu + \epsilon
   $$

onde:

- $W$ é uma matriz de projeção $d \times q$,  
- $\mu$ é a média dos dados,  
- $\epsilon \sim \mathcal{N}(0, \sigma^2 I_d)$ é ruído isotrópico.

A distribuição marginal de $x$ é:

$$
x \sim \mathcal{N}(\mu, WW^\top + \sigma^2 I_d).
$$

---

### Intuição

O PPCA diz:

> “Os dados vivem perto de um subespaço de dimensão menor, com ruído Gaussiano ao redor.”

Esse subespaço é definido pelas colunas de $W$.

O PCA clássico encontra esse subespaço via autovetores.  
O PPCA encontra via **máxima verossimilhança**, usando EM.

---

### Variáveis Latentes

O PPCA introduz a variável latente $z$ que representa a coordenada do ponto no subespaço de baixa dimensão.

O EM alterna entre:

- **Passo E:** inferir $z$ para cada ponto,  
- **Passo M:** atualizar $W$ e $\sigma^2$.

---

### Passo E — Inferência de $z$

Dado $W$, $\mu$ e $\sigma^2$, a distribuição posterior de $z$ é Gaussiana:

$$
p(z \mid x)
=
\mathcal{N}(M^{-1} W^\top (x - \mu), \, \sigma^2 M^{-1}),
$$

onde:

$$
M = W^\top W + \sigma^2 I_q.
$$

O Passo E calcula:

- a média posterior:
  $$
  \mathbb{E}[z_i] = M^{-1} W^\top (x^{(i)} - \mu)
  $$

- a covariância posterior:
  $$
  \mathbb{E}[z_i z_i^\top] = \sigma^2 M^{-1} + \mathbb{E}[z_i] \mathbb{E}[z_i]^\top
  $$

---

### Passo M — Atualização de $W$ e $\sigma^2$

O Passo M atualiza os parâmetros maximizando a verossimilhança completa esperada.

Atualização de $W$:

$$
W^{\text{novo}}
=
\left(
\sum_{i=1}^n (x^{(i)} - \mu) \, \mathbb{E}[z_i]^\top
\right)
\left(
\sum_{i=1}^n \mathbb{E}[z_i z_i^\top]
\right)^{-1}.
$$

Atualização de $\sigma^2$:

$$
\sigma^{2\,\text{novo}}
=
\frac{1}{nd}
\sum_{i=1}^n
\left[
\|x^{(i)} - \mu\|^2
- 2 \mathbb{E}[z_i]^\top W^{\text{novo}\,\top} (x^{(i)} - \mu)
+ \operatorname{tr}\left(
\mathbb{E}[z_i z_i^\top] W^{\text{novo}\,\top} W^{\text{novo}}
\right)
\right].
$$

---

### Conexão com PCA Clássico

O PPCA recupera o PCA clássico quando:

- $\sigma^2 \to 0$ (ruído muito pequeno),
- $W$ contém os autovetores principais.

O subespaço encontrado pelo PPCA é o mesmo do PCA clássico.

---

### Vantagens do PPCA

1. **Interpretação probabilística**  
   PCA deixa de ser apenas geométrico.

2. **Inferência de variáveis latentes**  
   Cada ponto tem coordenadas latentes $z$.

3. **Base para modelos mais avançados**  
   PPCA é precursor de FA (Factor Analysis) e VAEs.

4. **Treinamento via EM**  
   Permite lidar com dados faltantes.

5. **Robustez**  
   Estimativas mais estáveis em alta dimensão.

---

### Conclusão

O PPCA reformula o PCA como um modelo generativo com variáveis latentes.  
O EM permite estimar $W$ e $\sigma^2$ de forma eficiente, conectando PCA a toda a família de modelos probabilísticos.

A próxima seção é **Mistura de Experts**.




<br><br>



<a id="mistura_experts"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Mistura de Experts  
=============================

A **Mistura de Experts (Mixture of Experts — MoE)** é um modelo generativo e discriminativo ao mesmo tempo.  
Ele combina múltiplos modelos especializados (“experts”) e um **gating network** que decide, para cada entrada, qual expert deve ser mais influente.

A ideia central é decompor um problema complexo em partes mais simples, onde cada expert é responsável por uma região ou padrão específico dos dados.

---

### Estrutura do Modelo

O modelo possui:

1. **Experts**  
   Cada expert é um modelo (regressão, classificação, rede neural, etc.) que produz uma saída:

   $$
   f_j(x)
   $$

2. **Gating Network**  
   Uma rede que produz pesos (probabilidades) para cada expert:

   $$
   p(j \mid x)
   $$

3. **Combinação das saídas**  
   A saída final é uma média ponderada:

   $$
   y(x) = \sum_{j=1}^K p(j \mid x) \, f_j(x).
   $$

O gating network decide **qual expert deve dominar** para cada entrada.

---

### Intuição

A Mistura de Experts funciona como:

> “Para cada entrada, escolha o expert mais adequado — ou combine vários — de forma probabilística.”

Isso permite:

- dividir o espaço de entrada em regiões,  
- especializar cada expert em uma parte do problema,  
- criar modelos mais expressivos e interpretáveis.

---

### Modelo Generativo

A versão probabilística assume:

1. Escolher um expert $Z=j$ com probabilidade $p(j \mid x)$.
2. Gerar a saída $y$ segundo o expert:

   $$
   y \sim p(y \mid x, j).
   $$

A distribuição conjunta é:

$$
p(y, j \mid x)
= p(j \mid x) \, p(y \mid x, j).
$$

A marginal é:

$$
p(y \mid x)
= \sum_{j=1}^K p(j \mid x) \, p(y \mid x, j).
$$

---

### Mistura de Experts e EM

O treinamento pode ser feito com EM quando os experts são modelos probabilísticos.

**Passo E:**  
Responsabilidades dos experts:

$$
\gamma_{j,i}
=
p(j \mid x^{(i)}, y^{(i)}).
$$

**Passo M:**  
Atualização:

- dos parâmetros dos experts,  
- dos parâmetros do gating network.

O gating network é geralmente uma regressão logística ou uma pequena rede neural.

---

### Mistura de Experts para Regressão

Cada expert é uma regressão linear:

$$
f_j(x) = w_j^\top x.
$$

A saída final é:

$$
y(x) = \sum_{j=1}^K p(j \mid x) \, w_j^\top x.
$$

Isso permite modelar funções altamente não lineares como combinações de regressões lineares especializadas.

---

### Mistura de Experts para Classificação

Cada expert é um classificador:

$$
f_j(x) = p(y \mid x, j).
$$

A saída final é:

$$
p(y \mid x)
= \sum_{j=1}^K p(j \mid x) \, p(y \mid x, j).
$$

Isso produz classificadores:

- não lineares,  
- interpretáveis,  
- com regiões especializadas.

---

### Vantagens da Mistura de Experts

1. **Especialização**  
   Cada expert aprende uma parte do problema.

2. **Flexibilidade**  
   Pode combinar modelos lineares, não lineares, redes neurais, etc.

3. **Probabilístico**  
   O gating network produz probabilidades.

4. **Escalável**  
   Pode ser distribuído em múltiplos dispositivos (MoE moderno).

5. **Expressivo**  
   Modela funções complexas como combinações de funções simples.

---

### Conexão com Modelos Modernos

Mistura de Experts é a base de:

- MoE em redes neurais profundas,  
- Switch Transformers,  
- modelos gigantes com roteamento de tokens,  
- arquiteturas eficientes para LLMs.

O conceito clássico da Unit 4 evoluiu para arquiteturas modernas de larga escala.

---

### Conclusão

Mistura de Experts combina múltiplos modelos especializados usando um gating network probabilístico.  
É um modelo poderoso, interpretável e altamente expressivo, com aplicações que vão desde regressão e classificação até arquiteturas modernas de deep learning.

A próxima seção é **Mistura de Experts para Regressão**.




<br><br>




<a id="moe_regressao"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Mistura de Experts para Regressão  
=============================

A Mistura de Experts (MoE) para regressão combina vários modelos de regressão especializados, cada um responsável por uma região ou padrão específico dos dados.  
O gating network decide, para cada entrada, qual expert deve contribuir mais para a predição.

Essa abordagem permite modelar funções altamente não lineares como combinações de regressões lineares simples.

---

### Estrutura do Modelo

O modelo possui:

1. **Experts de regressão**  
   Cada expert $j$ produz uma predição linear:

   $$
   f_j(x) = w_j^\top x.
   $$

2. **Gating network**  
   Produz probabilidades para cada expert:

   $$
   p(j \mid x)
   $$

   Geralmente implementado como uma regressão logística ou uma pequena rede neural.

3. **Combinação das predições**  
   A saída final é:

   $$
   y(x) = \sum_{j=1}^K p(j \mid x) \, w_j^\top x.
   $$

---

### Intuição

A Mistura de Experts para regressão funciona como:

> “Cada expert aprende uma parte da função.  
> O gating network decide qual expert deve dominar para cada entrada.”

Isso permite:

- dividir o espaço de entrada em regiões,  
- especializar cada expert em uma parte da função,  
- capturar não linearidades complexas.

---

### Modelo Probabilístico

Assumimos que a saída é gerada por:

$$
y \sim \mathcal{N}(w_j^\top x, \sigma_j^2)
$$

quando o expert $j$ é escolhido.

A distribuição completa é:

$$
p(y \mid x)
=
\sum_{j=1}^K
p(j \mid x)
\,
\mathcal{N}(y; w_j^\top x, \sigma_j^2).
$$

---

### Treinamento via EM

O EM é usado quando os experts são modelos probabilísticos.

---

#### Passo E — Responsabilidades

A responsabilidade do expert $j$ sobre o ponto $(x^{(i)}, y^{(i)})$ é:

$$
\gamma_{j,i}
=
\frac{
p(j \mid x^{(i)}) \,
\mathcal{N}(y^{(i)}; w_j^\top x^{(i)}, \sigma_j^2)
}{
\sum_{\ell=1}^K
p(\ell \mid x^{(i)}) \,
\mathcal{N}(y^{(i)}; w_\ell^\top x^{(i)}, \sigma_\ell^2)
}.
$$

---

#### Passo M — Atualização dos Experts

Cada expert é atualizado usando regressão ponderada:

$$
w_j^{\text{novo}}
=
\left(
\sum_{i=1}^n
\gamma_{j,i} \, x^{(i)} x^{(i)\top}
\right)^{-1}
\left(
\sum_{i=1}^n
\gamma_{j,i} \, x^{(i)} y^{(i)}
\right).
$$

Atualização das variâncias:

$$
\sigma_j^{2\,\text{novo}}
=
\frac{
\sum_{i=1}^n
\gamma_{j,i}
\left(y^{(i)} - w_j^{\text{novo}\,\top} x^{(i)}\right)^2
}{
\sum_{i=1}^n \gamma_{j,i}
}.
$$

---

#### Atualização do Gating Network

O gating network é treinado para maximizar:

$$
\sum_{i=1}^n \sum_{j=1}^K
\gamma_{j,i} \log p(j \mid x^{(i)}).
$$

Geralmente usando gradiente descendente.

---

### Interpretação Geométrica

Cada expert aprende uma regressão linear:

- uma inclinação,  
- uma direção,  
- uma região do espaço.

O gating network aprende:

- onde cada expert deve atuar,  
- como combinar experts,  
- como dividir o espaço de entrada.

O resultado é uma função altamente não linear composta de pedaços lineares.

---

### Vantagens da Mistura de Experts para Regressão

1. **Modela funções complexas**  
   Combina regressões lineares em regiões diferentes.

2. **Interpretação clara**  
   Cada expert é responsável por uma parte da função.

3. **Probabilístico**  
   Produz incerteza sobre qual expert deve atuar.

4. **Flexível**  
   Pode usar regressões lineares, polinomiais ou redes neurais como experts.

5. **Escalável**  
   Pode ser distribuído em múltiplos dispositivos (MoE moderno).

---

### Conexão com Deep Learning

A versão moderna da Mistura de Experts é usada em:

- Switch Transformers,  
- roteamento de tokens,  
- modelos gigantes com especialistas especializados,  
- arquiteturas eficientes para LLMs.

A ideia clássica da Unit 4 evoluiu para arquiteturas de larga escala.

---

### Conclusão

Mistura de Experts para regressão combina múltiplos modelos especializados usando um gating network probabilístico.  
Ela é poderosa, interpretável e capaz de modelar funções altamente não lineares.

A próxima seção é **Mistura de Experts para Classificação**.




<br><br>
<a id="moe_classificacao"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Mistura de Experts para Classificação  
=============================

A **Mistura de Experts (MoE)** para classificação combina múltiplos classificadores especializados, cada um responsável por uma região ou padrão específico dos dados.  
O gating network decide, para cada entrada, qual expert deve contribuir mais para a predição.

Essa abordagem produz classificadores **não lineares**, **interpretáveis** e **probabilísticos**.

---

### Estrutura do Modelo

O modelo possui:

1. **Experts de classificação**  
   Cada expert $j$ produz uma distribuição sobre as classes:

   $$
   f_j(x) = p(y \mid x, j).
   $$

2. **Gating network**  
   Produz probabilidades para cada expert:

   $$
   p(j \mid x).
   $$

3. **Combinação das predições**  
   A distribuição final sobre as classes é:

   $$
   p(y \mid x)
   = \sum_{j=1}^K p(j \mid x) \, p(y \mid x, j).
   $$

---

### Intuição

A Mistura de Experts para classificação funciona como:

> “Cada expert é bom em classificar uma região do espaço.  
> O gating network decide qual expert deve dominar para cada entrada.”

Isso permite:

- dividir o espaço de entrada em regiões,  
- especializar cada expert em um tipo de padrão,  
- capturar fronteiras de decisão complexas.

---

### Modelo Probabilístico

Assumimos que:

1. O gating network escolhe um expert $Z=j$ com probabilidade $p(j \mid x)$.
2. O expert $j$ gera a classe:

   $$
   y \sim p(y \mid x, j).
   $$

A distribuição final é:

$$
p(y \mid x)
=
\sum_{j=1}^K p(j \mid x) \, p(y \mid x, j).
$$

---

### Treinamento via EM

O EM é usado quando os experts são modelos probabilísticos.

---

#### Passo E — Responsabilidades

A responsabilidade do expert $j$ sobre o ponto $(x^{(i)}, y^{(i)})$ é:

$$
\gamma_{j,i}
=
\frac{
p(j \mid x^{(i)}) \, p(y^{(i)} \mid x^{(i)}, j)
}{
\sum_{\ell=1}^K
p(\ell \mid x^{(i)}) \, p(y^{(i)} \mid x^{(i)}, \ell)
}.
$$

---

#### Passo M — Atualização dos Experts

Cada expert é atualizado para maximizar:

$$
\sum_{i=1}^n \gamma_{j,i} \log p(y^{(i)} \mid x^{(i)}, j).
$$

Se o expert é uma regressão logística:

- atualiza pesos via gradiente,  
- com cada ponto ponderado por $\gamma_{j,i}$.

Se o expert é uma rede neural:

- treina com loss ponderado por $\gamma_{j,i}$.

---

#### Atualização do Gating Network

O gating network é treinado para maximizar:

$$
\sum_{i=1}^n \sum_{j=1}^K
\gamma_{j,i} \log p(j \mid x^{(i)}).
$$

Geralmente usando regressão logística ou uma pequena rede neural.

---

### Interpretação Geométrica

Cada expert aprende uma fronteira de decisão:

- linear,  
- logística,  
- ou não linear (se for uma rede neural).

O gating network aprende:

- onde cada expert deve atuar,  
- como dividir o espaço de entrada,  
- como combinar experts.

O resultado é um classificador altamente não linear composto de pedaços especializados.

---

### Vantagens da Mistura de Experts para Classificação

1. **Fronteiras não lineares complexas**  
   Combina múltiplos classificadores especializados.

2. **Interpretação clara**  
   Cada expert é responsável por uma região do espaço.

3. **Probabilístico**  
   Produz $p(y \mid x)$ e incerteza sobre qual expert atua.

4. **Flexível**  
   Pode usar regressões logísticas, SVMs ou redes neurais como experts.

5. **Escalável**  
   Base para arquiteturas modernas de MoE em deep learning.

---

### Conexão com Deep Learning

A versão moderna da Mistura de Experts é usada em:

- Switch Transformers,  
- roteamento de tokens,  
- modelos gigantes com especialistas especializados,  
- arquiteturas eficientes para LLMs.

A ideia clássica da Unit 4 evoluiu para arquiteturas de larga escala.

---

### Conclusão

Mistura de Experts para classificação combina múltiplos classificadores especializados usando um gating network probabilístico.  
Ela é poderosa, interpretável e capaz de modelar fronteiras altamente não lineares.

A próxima seção é **Mistura de Experts para Classificação Multiclasse**.



<br><Br>



<a id="moe_multiclasse"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

======================================  
Mistura de Experts para Classificação Multiclasse  
======================================

A Mistura de Experts (MoE) para classificação multiclasse estende o modelo de experts para lidar com múltiplas classes simultaneamente.  
Cada expert é responsável por aprender padrões específicos de algumas regiões do espaço de entrada, enquanto o gating network decide, para cada entrada, qual expert deve contribuir mais.

Essa abordagem produz classificadores **não lineares**, **probabilísticos** e **altamente interpretáveis**.

---

### Estrutura Geral

O modelo possui:

1. **Experts multiclasse**  
   Cada expert $j$ produz uma distribuição sobre as classes:

   $$
   p(y = c \mid x, j),
   \qquad c = 1, \dots, C.
   $$

2. **Gating network**  
   Produz probabilidades para cada expert:

   $$
   p(j \mid x),
   \qquad j = 1, \dots, K.
   $$

3. **Combinação das predições**  
   A distribuição final sobre as classes é:

   $$
   p(y = c \mid x)
   =
   \sum_{j=1}^K
   p(j \mid x) \, p(y = c \mid x, j).
   $$

---

### Intuição

A Mistura de Experts multiclasse funciona como:

> “Cada expert é bom em classificar certos tipos de padrões ou regiões.  
> O gating network decide qual expert deve dominar para cada entrada.”

Isso permite:

- dividir o espaço de entrada em regiões especializadas,  
- capturar fronteiras de decisão complexas,  
- combinar múltiplos classificadores de forma probabilística.

---

### Modelo Probabilístico

O processo generativo é:

1. O gating network escolhe um expert $Z=j$ com probabilidade $p(j \mid x)$.
2. O expert $j$ gera a classe:

   $$
   y \sim p(y \mid x, j).
   $$

A distribuição final é:

$$
p(y \mid x)
=
\sum_{j=1}^K p(j \mid x) \, p(y \mid x, j).
$$

---

### Treinamento via EM

O EM é usado quando os experts são modelos probabilísticos.

---

#### Passo E — Responsabilidades

A responsabilidade do expert $j$ sobre o ponto $(x^{(i)}, y^{(i)})$ é:

$$
\gamma_{j,i}
=
\frac{
p(j \mid x^{(i)}) \, p(y^{(i)} \mid x^{(i)}, j)
}{
\sum_{\ell=1}^K
p(\ell \mid x^{(i)}) \, p(y^{(i)} \mid x^{(i)}, \ell)
}.
$$

---

#### Passo M — Atualização dos Experts

Cada expert é atualizado para maximizar:

$$
\sum_{i=1}^n
\gamma_{j,i}
\log p(y^{(i)} \mid x^{(i)}, j).
$$

Se o expert é uma regressão logística multiclasse:

- atualiza pesos via gradiente,  
- com cada ponto ponderado por $\gamma_{j,i}$.

Se o expert é uma rede neural:

- treina com loss ponderado por $\gamma_{j,i}$.

---

#### Atualização do Gating Network

O gating network é treinado para maximizar:

$$
\sum_{i=1}^n \sum_{j=1}^K
\gamma_{j,i} \log p(j \mid x^{(i)}).
$$

Geralmente usando regressão logística ou uma pequena rede neural.

---

### Interpretação Geométrica

Cada expert aprende uma fronteira de decisão multiclasse:

- linear,  
- logística,  
- ou não linear (se for uma rede neural).

O gating network aprende:

- como dividir o espaço de entrada,  
- onde cada expert deve atuar,  
- como combinar experts.

O resultado é um classificador multiclasse altamente não linear composto de pedaços especializados.

---

### Vantagens da Mistura de Experts Multiclasse

1. **Fronteiras altamente não lineares**  
   Combina múltiplos classificadores especializados.

2. **Interpretação clara**  
   Cada expert é responsável por uma região ou padrão.

3. **Probabilístico**  
   Produz $p(y \mid x)$ e incerteza sobre qual expert atua.

4. **Flexível**  
   Pode usar regressões logísticas, SVMs ou redes neurais como experts.

5. **Escalável**  
   Base para arquiteturas modernas de MoE em deep learning.

---

### Conexão com Deep Learning

A versão moderna da Mistura de Experts é usada em:

- Switch Transformers,  
- roteamento de tokens,  
- modelos gigantes com especialistas especializados,  
- arquiteturas eficientes para LLMs.

A ideia clássica da Unit 4 evoluiu para arquiteturas de larga escala.

---

### Conclusão

Mistura de Experts para classificação multiclasse combina múltiplos classificadores especializados usando um gating network probabilístico.  
Ela é poderosa, interpretável e capaz de modelar fronteiras altamente não lineares em problemas com muitas classes.

A próxima seção é **Mistura de Experts para Regressão Não Linear**.




<br><br>




<a id="moe_regressao_nao_linear"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

======================================  
Mistura de Experts para Regressão Não Linear  
======================================

A Mistura de Experts (MoE) para regressão não linear é uma extensão natural da MoE linear.  
Aqui, cada expert deixa de ser uma regressão linear simples e passa a ser um **modelo não linear**, como:

- regressão polinomial,  
- redes neurais,  
- kernels,  
- modelos locais não lineares.

O gating network continua decidindo, para cada entrada, qual expert deve dominar — mas agora os experts são capazes de capturar padrões muito mais complexos.

---

### Estrutura do Modelo

O modelo possui:

1. **Experts não lineares**  
   Cada expert $j$ produz uma predição não linear:

   $$
   f_j(x) = g_j(x; \theta_j),
   $$

   onde $g_j$ pode ser:
   - uma rede neural,  
   - uma regressão polinomial,  
   - um modelo kernel,  
   - um RBF, etc.

2. **Gating network**  
   Produz probabilidades para cada expert:

   $$
   p(j \mid x).
   $$

3. **Combinação das predições**  
   A saída final é:

   $$
   y(x)
   =
   \sum_{j=1}^K p(j \mid x) \, g_j(x; \theta_j).
   $$

---

### Intuição

A MoE para regressão não linear funciona como:

> “Cada expert aprende uma parte não linear da função.  
> O gating network decide qual expert deve atuar em cada região.”

Isso permite:

- decompor funções complexas em partes mais simples,  
- especializar cada expert em padrões específicos,  
- capturar não linearidades profundas.

---

### Modelo Probabilístico

Assumimos que a saída é gerada por:

$$
y \sim \mathcal{N}(g_j(x; \theta_j), \sigma_j^2)
$$

quando o expert $j$ é escolhido.

A distribuição completa é:

$$
p(y \mid x)
=
\sum_{j=1}^K
p(j \mid x)
\,
\mathcal{N}(y; g_j(x; \theta_j), \sigma_j^2).
$$

---

### Treinamento via EM

O EM continua funcionando, desde que os experts sejam probabilísticos.

---

#### Passo E — Responsabilidades

A responsabilidade do expert $j$ sobre o ponto $(x^{(i)}, y^{(i)})$ é:

$$
\gamma_{j,i}
=
\frac{
p(j \mid x^{(i)}) \,
\mathcal{N}(y^{(i)}; g_j(x^{(i)}; \theta_j), \sigma_j^2)
}{
\sum_{\ell=1}^K
p(\ell \mid x^{(i)}) \,
\mathcal{N}(y^{(i)}; g_\ell(x^{(i)}; \theta_\ell), \sigma_\ell^2)
}.
$$

---

#### Passo M — Atualização dos Experts

Cada expert é atualizado para minimizar o erro ponderado:

$$
\theta_j^{\text{novo}}
=
\arg\min_{\theta_j}
\sum_{i=1}^n
\gamma_{j,i}
\left(y^{(i)} - g_j(x^{(i)}; \theta_j)\right)^2.
$$

Se o expert é uma rede neural:

- treina com loss ponderado por $\gamma_{j,i}$,  
- usando gradiente descendente.

Se o expert é um modelo kernel:

- ajusta pesos com regularização ponderada.

Atualização das variâncias:

$$
\sigma_j^{2\,\text{novo}}
=
\frac{
\sum_{i=1}^n
\gamma_{j,i}
\left(y^{(i)} - g_j(x^{(i)}; \theta_j^{\text{novo}})\right)^2
}{
\sum_{i=1}^n \gamma_{j,i}
}.
$$

---

#### Atualização do Gating Network

O gating network é treinado para maximizar:

$$
\sum_{i=1}^n \sum_{j=1}^K
\gamma_{j,i} \log p(j \mid x^{(i)}).
$$

Geralmente usando:

- regressão logística,  
- softmax linear,  
- ou uma pequena rede neural.

---

### Interpretação Geométrica

Cada expert aprende uma função não linear:

- uma curva,  
- uma superfície,  
- uma região complexa do espaço.

O gating network aprende:

- como dividir o espaço de entrada,  
- onde cada expert deve atuar,  
- como combinar experts.

O resultado é uma função altamente não linear composta de pedaços especializados.

---

### Exemplos de Experts Não Lineares

1. **Redes neurais pequenas (MLPs)**  
   Cada expert aprende uma função não linear local.

2. **RBF networks**  
   Cada expert é centrado em uma região do espaço.

3. **Regressão polinomial**  
   Cada expert aprende uma curva diferente.

4. **Modelos kernel**  
   Cada expert aprende padrões complexos via kernels.

---

### Vantagens da MoE Não Linear

1. **Extremamente expressiva**  
   Pode aproximar funções arbitrárias.

2. **Interpretação clara**  
   Cada expert é responsável por uma parte da função.

3. **Probabilística**  
   Produz incerteza sobre qual expert atua.

4. **Modular**  
   Experts podem ser treinados separadamente.

5. **Escalável**  
   Base para MoE modernos em deep learning.

---

### Conexão com Deep Learning

A versão moderna da MoE não linear é usada em:

- Switch Transformers,  
- roteamento de tokens,  
- modelos gigantes com especialistas especializados,  
- arquiteturas eficientes para LLMs.

A ideia clássica evoluiu para arquiteturas de larga escala.

---

### Conclusão

Mistura de Experts para regressão não linear combina múltiplos modelos especializados usando um gating network probabilístico.  
Ela é poderosa, interpretável e capaz de modelar funções altamente não lineares, sendo uma ponte direta para arquiteturas modernas de deep learning.





<br><br>




<a id="em_latentes"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
EM em Modelos Latentes  
=============================

O algoritmo EM é especialmente poderoso quando trabalhamos com **modelos que possuem variáveis latentes** — isto é, partes do modelo que não são observadas diretamente, mas influenciam a geração dos dados.

Modelos latentes aparecem em:

- Misturas de Gaussianas (clusters latentes),  
- PCA Probabilístico (variáveis latentes contínuas),  
- Mistura de Experts (expert latente),  
- Modelos de tópicos,  
- Modelos gráficos,  
- Modelos generativos modernos.

O EM fornece uma forma sistemática de lidar com essas variáveis ocultas.

---

### O Papel das Variáveis Latentes

Um modelo latente assume que cada dado observado $x$ é gerado a partir de uma variável oculta $z$:

$$
p(x, z \mid \theta)
$$

onde $\theta$ são os parâmetros do modelo.

O objetivo é maximizar a verossimilhança marginal:

$$
p(x \mid \theta)
= \sum_z p(x, z \mid \theta)
$$

ou, no caso contínuo:

$$
p(x \mid \theta)
= \int p(x, z \mid \theta) \, dz.
$$

Mas essa soma/integral é frequentemente impossível de calcular diretamente.

É aí que o EM entra.

---

### Como o EM Resolve o Problema

O EM alterna entre:

1. **Passo E (Expectation)**  
   Estimar a distribuição posterior das variáveis latentes:

   $$
   p(z \mid x, \theta^{\text{antigo}})
   $$

2. **Passo M (Maximization)**  
   Atualizar os parâmetros maximizando a expectativa da log‑verossimilhança completa:

   $$
   \theta^{\text{novo}}
   =
   \arg\max_\theta
   \mathbb{E}_{z \mid x, \theta^{\text{antigo}}}
   [\log p(x, z \mid \theta)].
   $$

O EM transforma um problema difícil (maximizar $p(x)$) em dois problemas fáceis:

- inferir $z$,  
- maximizar a verossimilhança completa.

---

### Intuição

O EM funciona como:

> “Adivinhe as variáveis latentes (Passo E).  
> Atualize os parâmetros como se essas variáveis fossem reais (Passo M).  
> Repita.”

Essa alternância converge para um ponto de máximo local da verossimilhança.

---

### Exemplos de Modelos Latentes

#### 1. Mistura de Gaussianas  
Latente: cluster $z$  
Passo E: responsabilidades  
Passo M: atualizar médias, covariâncias e pesos

#### 2. PCA Probabilístico  
Latente: coordenada $z$ no subespaço  
Passo E: inferir $z$  
Passo M: atualizar $W$ e $\sigma^2$

#### 3. Mistura de Experts  
Latente: expert responsável  
Passo E: responsabilidades dos experts  
Passo M: atualizar experts e gating network

#### 4. Modelos de Tópicos  
Latente: tópico de cada palavra  
Passo E: distribuição de tópicos  
Passo M: atualizar distribuições de palavras

---

### EM como Inferência Aproximada

Em muitos modelos latentes, a distribuição posterior $p(z \mid x)$ é:

- difícil de calcular,  
- intractável,  
- impossível de obter analiticamente.

Nesses casos, o EM pode ser:

- aproximado,  
- variacional,  
- estocástico.

Isso conecta o EM a métodos modernos como:

- Variational Inference (VI),  
- Expectation Propagation (EP),  
- Autoencoders Variacionais (VAEs).

---

### EM e Modelos Gráficos

Em modelos gráficos com variáveis latentes:

- HMMs,  
- modelos de Markov,  
- modelos hierárquicos,

o EM aparece como:

- **Forward–Backward** (HMM),  
- **Baum–Welch**,  
- **EM hierárquico**,  
- **EM com mensagens**.

O Passo E é feito via inferência no grafo.  
O Passo M é feito via maximização local.

---

### Conclusão

O EM é a ferramenta fundamental para treinar modelos com variáveis latentes.  
Ele transforma um problema difícil (maximizar $p(x)$) em dois problemas simples:

- inferir $z$,  
- maximizar a verossimilhança completa.

Essa estrutura aparece em praticamente todos os modelos probabilísticos modernos.

A próxima seção é **Relação com Variational Inference**.





<br><br>




<a id="relacao_vi"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Relação com Variational Inference  
=============================

O algoritmo EM é um dos métodos mais antigos e fundamentais para lidar com **variáveis latentes**.  
Variational Inference (VI) é uma generalização moderna que resolve os mesmos problemas, mas em cenários onde o EM clássico não consegue atuar — especialmente quando a distribuição posterior é intractável.

Esta seção explica como EM e VI se relacionam, como EM pode ser visto como um caso especial de VI, e por que VI se tornou essencial em modelos generativos modernos.

---

### O Problema Central

Queremos maximizar a verossimilhança marginal:

$$
\log p(x \mid \theta)
= \log \int p(x, z \mid \theta) \, dz.
$$

Mas essa integral é frequentemente impossível de calcular.

O EM resolve isso quando o posterior $p(z \mid x)$ é fácil de obter.  
O VI resolve isso quando o posterior é **difícil**.

---

### EM como Inferência Exata

No EM, o Passo E calcula:

$$
q(z) = p(z \mid x, \theta^{\text{antigo}})
$$

ou seja, o **posterior exato**.

Isso só é possível quando:

- o modelo é conjugado,  
- a distribuição tem forma fechada,  
- o posterior é analítico.

Exemplos:

- GMM,  
- PCA probabilístico,  
- Mistura de Experts clássica,  
- HMMs com Baum–Welch.

---

### Variational Inference como Inferência Aproximada

Quando o posterior é intractável, o VI introduz uma família de distribuições aproximadoras:

$$
q(z) \in \mathcal{Q}
$$

e escolhe a melhor aproximação minimizando a divergência KL:

$$
q^*(z)
=
\arg\min_{q \in \mathcal{Q}}
\mathrm{KL}(q(z) \,\|\, p(z \mid x)).
$$

Isso substitui o Passo E do EM por uma **inferência aproximada**.

---

### A Conexão Matemática

O EM maximiza a função:

$$
\mathcal{L}(q, \theta)
=
\mathbb{E}_{q(z)}[\log p(x, z \mid \theta)]
+
H(q),
$$

onde $H(q)$ é a entropia.

O VI maximiza exatamente a mesma função, chamada de **ELBO**:

$$
\text{ELBO}(q, \theta)
=
\mathbb{E}_{q(z)}[\log p(x, z \mid \theta)]
+
H(q).
$$

A diferença é:

- **EM:** o Passo E escolhe $q(z) = p(z \mid x)$  
- **VI:** o Passo E escolhe $q(z)$ dentro de uma família restrita (ex.: fatorizada)

Portanto:

### ✔️ EM é um caso especial de VI  
onde a família variacional contém o posterior exato.

---

### EM vs. VI — Comparação

| Aspecto | EM | Variational Inference |
|--------|----|------------------------|
| Posterior | exato | aproximado |
| Família variacional | completa | restrita |
| Convergência | rápida | depende da aproximação |
| Flexibilidade | limitada | muito alta |
| Modelos suportados | conjugados | praticamente qualquer modelo |
| Base para VAEs | não | sim |

---

### Quando EM funciona

- Modelos conjugados  
- Misturas de Gaussianas  
- PCA probabilístico  
- Mistura de Experts clássica  
- HMMs com Baum–Welch  
- Modelos com posterior analítico

---

### Quando VI é necessário

- Modelos com posterior complexo  
- Modelos hierárquicos profundos  
- Modelos gráficos grandes  
- Modelos generativos modernos  
- Autoencoders Variacionais (VAEs)  
- Difusão, flows, transformers probabilísticos

---

### EM e VI em Modelos Modernos

Nos modelos modernos:

- VAEs  
- Bayesian Neural Networks  
- Deep Latent Variable Models  
- Modelos de tópicos avançados  
- Modelos probabilísticos hierárquicos

o posterior é intractável.

Por isso, o EM clássico não funciona.  
VI substitui o Passo E por uma rede neural (encoder) que aproxima o posterior.

---

### Conclusão

EM e VI resolvem o mesmo problema: lidar com variáveis latentes.  
A diferença é:

- **EM:** inferência exata  
- **VI:** inferência aproximada

Matematicamente, EM é um caso especial de VI.  
Conceitualmente, VI é a ponte que leva dos modelos clássicos (GMM, PCA) aos modelos modernos (VAEs, difusão, deep generative models).

A próxima seção é **Modelos Generativos Modernos**.





<br><br>



<a id="modelos_modernos"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Modelos Generativos Modernos  
=============================

Os modelos generativos evoluíram drasticamente desde as primeiras misturas de Gaussianas e o algoritmo EM.  
Hoje, modelos generativos modernos são capazes de:

- gerar imagens realistas,  
- sintetizar áudio e vídeo,  
- criar texto coerente,  
- aprender distribuições complexas,  
- capturar estruturas profundas e hierárquicas.

Esta seção conecta os modelos clássicos da Unit 4 aos modelos generativos contemporâneos.

---

### Linha do Tempo dos Modelos Generativos

1. **GMMs (anos 90)**  
   Misturas de Gaussianas, EM, clustering probabilístico.

2. **Modelos Latentes Lineares (anos 90–2000)**  
   PCA probabilístico, Factor Analysis.

3. **Modelos Gráficos (anos 2000)**  
   HMMs, modelos hierárquicos, EM generalizado.

4. **Modelos Variacionais (anos 2010)**  
   Variational Inference, ELBO, aproximações.

5. **VAEs (2013–2014)**  
   Autoencoders Variacionais — primeira ponte entre deep learning e modelos probabilísticos.

6. **GANs (2014)**  
   Redes adversariais — geração de imagens realistas.

7. **Normalizing Flows (2018)**  
   Modelos invertíveis com densidade exata.

8. **Modelos de Difusão (2020–2023)**  
   Base de modelos como Stable Diffusion.

9. **Transformers Generativos (2020–2024)**  
   Modelos de linguagem, multimodais, grandes modelos generativos.

---

### Conexão com os Modelos Clássicos

Os modelos modernos ainda seguem a mesma lógica dos modelos clássicos:

- **variáveis latentes**,  
- **inferência**,  
- **maximização de verossimilhança**,  
- **modelagem de distribuições complexas**.

A diferença é que agora usamos:

- redes neurais profundas,  
- otimização por gradiente,  
- inferência variacional,  
- arquiteturas complexas.

---

### Categorias de Modelos Generativos Modernos

---

## 1. Autoencoders Variacionais (VAEs)

Os VAEs são descendentes diretos dos modelos latentes treinados com EM.

Eles substituem:

- o Passo E → por uma rede neural que aproxima o posterior  
- o Passo M → por gradiente descendente na ELBO

O VAE é:

$$
\text{ELBO} = \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - \mathrm{KL}(q(z \mid x) \,\|\, p(z)).
$$

VAEs são usados para:

- geração de imagens,  
- compressão,  
- modelos latentes profundos.

---

## 2. GANs — Generative Adversarial Networks

GANs não usam EM nem VI.  
Eles treinam um gerador e um discriminador em um jogo adversarial:

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)]
+
\mathbb{E}_{z \sim p(z)}[\log (1 - D(G(z)))].
$$

GANs produzem imagens extremamente realistas.

---

## 3. Normalizing Flows

Modelos invertíveis com densidade exata:

$$
p(x) = p(z) \left| \det \frac{\partial f^{-1}}{\partial x} \right|.
$$

Eles permitem:

- amostragem exata,  
- cálculo exato de densidade,  
- transformações complexas.

---

## 4. Modelos de Difusão

Modelos de difusão geram imagens revertendo um processo de ruído:

1. adicionar ruído progressivamente,  
2. aprender a remover ruído passo a passo.

A equação central é:

$$
x_{t-1} = x_t - \epsilon_\theta(x_t, t).
$$

Eles são a base de:

- Stable Diffusion,  
- DALL·E 3,  
- Midjourney.

---

## 5. Modelos Generativos Baseados em Transformers

Transformers podem ser usados como modelos generativos autoregressivos:

$$
p(x) = \prod_{t=1}^T p(x_t \mid x_{<t}).
$$

Eles são usados em:

- modelos de linguagem,  
- modelos multimodais,  
- geração de código,  
- síntese de texto e imagem.

---

### Conexão com EM e GMM

Apesar da complexidade dos modelos modernos, muitos conceitos da Unit 4 permanecem:

- variáveis latentes → VAEs  
- inferência → VI  
- mistura de distribuições → MoE modernos  
- densidades → flows  
- geração → difusão  
- clustering → embeddings e representações

A Unit 4 é a base teórica para entender modelos generativos modernos.

---

### Conclusão

Modelos generativos modernos são descendentes diretos dos modelos probabilísticos clássicos.  
Eles ampliam:

- a capacidade de modelar distribuições complexas,  
- a expressividade das variáveis latentes,  
- a eficiência da inferência,  
- a qualidade da geração.

A próxima seção é **Da GMM aos VAEs**.





<br><br>




<a id="gmm_vae"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit4](#unit4)

=============================  
Da GMM aos VAEs  
=============================

Os modelos de Mistura de Gaussianas (GMM) são um dos primeiros modelos generativos probabilísticos.  
Os Autoencoders Variacionais (VAEs) são modelos generativos modernos, capazes de gerar imagens, sons e dados complexos.

Apesar de parecerem muito diferentes, existe uma **linha conceitual contínua** que conecta GMM → EM → modelos latentes → VI → VAEs.

Esta seção explica essa transição histórica e conceitual.

---

### 1. GMM: O Primeiro Modelo Generativo Latente

O GMM assume:

- uma variável latente discreta $z$ (cluster),  
- uma distribuição Gaussiana para cada cluster,  
- mistura ponderada das Gaussianas.

Modelo:

$$
p(x) = \sum_{j=1}^K p_j \, \mathcal{N}(x; \mu_j, \Sigma_j).
$$

Inferência:

- $p(z \mid x)$ é analítica,  
- EM funciona perfeitamente.

O GMM é o primeiro exemplo de:

- variável latente,  
- inferência,  
- maximização de verossimilhança.

---

### 2. EM: O Primeiro Algoritmo de Inferência Latente

O EM resolve:

$$
\log p(x) = \log \sum_z p(x, z).
$$

Alternando entre:

- **Passo E:** inferir $p(z \mid x)$  
- **Passo M:** maximizar $\mathbb{E}[\log p(x, z)]$

O EM funciona quando o posterior é **exato**.

---

### 3. Modelos Latentes Contínuos

O PCA probabilístico introduz variáveis latentes contínuas:

$$
z \sim \mathcal{N}(0, I), \qquad x = Wz + \mu + \epsilon.
$$

O posterior ainda é analítico:

$$
p(z \mid x) = \mathcal{N}(\text{média}, \text{covariância}).
$$

Portanto, o EM ainda funciona.

---

### 4. O Problema: Modelos Latentes Profundos

Quando o modelo é profundo:

$$
z \to \text{rede neural} \to x
$$

o posterior:

$$
p(z \mid x)
$$

não é mais analítico.

O EM não consegue calcular o Passo E.

Precisamos de **inferência aproximada**.

---

### 5. Variational Inference (VI): O Novo Passo E

O VI substitui o Passo E por uma aproximação:

$$
q(z \mid x) \approx p(z \mid x)
$$

maximizando a ELBO:

$$
\text{ELBO}
=
\mathbb{E}_{q(z \mid x)}[\log p(x \mid z)]
-
\mathrm{KL}(q(z \mid x) \,\|\, p(z)).
$$

O VI é uma generalização do EM.

---

### 6. VAEs: EM + VI + Redes Neurais

O VAE combina:

- **variáveis latentes** (como GMM e PCA),  
- **inferência variacional** (como VI),  
- **modelos profundos** (como redes neurais).

O encoder é o Passo E aproximado:

$$
q_\phi(z \mid x)
$$

O decoder é o modelo generativo:

$$
p_\theta(x \mid z)
$$

O treinamento maximiza a ELBO:

$$
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
-
\mathrm{KL}(q_\phi(z \mid x) \,\|\, p(z)).
$$

---

### 7. Conexão Conceitual Completa

| Modelo | Variável Latente | Inferência | Treinamento |
|--------|-------------------|------------|-------------|
| GMM | discreta | exata | EM |
| PCA Probabilístico | contínua | exata | EM |
| VI | contínua | aproximada | ELBO |
| VAE | contínua + rede neural | aproximada (encoder) | ELBO + gradiente |

---

### 8. Intuição da Transição

1. **GMM**  
   Variáveis latentes simples → inferência exata.

2. **PCA Probabilístico**  
   Variáveis latentes contínuas → ainda exato.

3. **Modelos profundos**  
   Inferência exata impossível.

4. **VI**  
   Aproximação do posterior.

5. **VAE**  
   Inferência aproximada feita por uma rede neural.

---

### 9. Da GMM aos VAEs — A Linha Evolutiva

- GMM introduz variáveis latentes.  
- EM introduz inferência e maximização alternada.  
- PCA probabilístico introduz latentes contínuos.  
- VI introduz aproximação do posterior.  
- VAEs introduzem redes neurais para inferência e geração.

Os VAEs são descendentes diretos dos modelos probabilísticos clássicos.

---

### Conclusão

A transição GMM → EM → VI → VAE mostra como modelos generativos evoluíram:

- de Gaussianas simples,  
- para modelos latentes contínuos,  
- para inferência aproximada,  
- para redes neurais profundas.

Os VAEs são a culminação moderna da teoria iniciada com GMMs e EM.




<br><br>




<a id="lecture17_parte1"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 17 — Parte 1  
Overview — Introdução ao Aprendizado por Reforço  
=============================

Nesta lecture iniciamos a Unidade 5, dedicada ao **Reinforcement Learning (RL)**.  
Assim como na Lecture 16, esta parte apresenta:

- o objetivo geral da unidade,  
- a motivação por trás do RL,  
- o cenário que diferencia RL de aprendizado supervisionado,  
- o papel dos MDPs como primeiro passo,  
- e como isso se conecta ao projeto final da unidade.

---

## 1. O que é Reinforcement Learning?

Reinforcement Learning é um paradigma onde:

> O agente **toma ações**, **interage com o ambiente**, e recebe **recompensa apenas no final** da tarefa.

Isso contrasta com o aprendizado supervisionado, onde:

- cada ação possui um rótulo correto,  
- cada passo tem supervisão direta.

No RL:

- não há supervisão passo a passo,  
- o agente deve descobrir sozinho quais ações levam ao sucesso,  
- a recompensa é atrasada e deve ser propagada para trás.

---

## 2. Motivação: Por que RL é importante?

Um dos exemplos mais empolgantes é o **AlphaGo**, onde:

- a máquina não recebe recompensa por cada jogada,  
- apenas importa se o jogo foi vencido ou perdido,  
- o agente aprende estratégias complexas sem supervisão direta.

Outros exemplos incluem:

- robôs navegando terrenos complexos,  
- sistemas de marketing decidindo ações para conquistar clientes,  
- agentes que aprendem a jogar videogames apenas pela recompensa final.

---

## 3. O desafio central do RL

O agente deve:

- explorar ações,  
- experimentar caminhos,  
- descobrir o que funciona,  
- e aprender a partir de recompensas atrasadas.

O problema é:

> Como propagar a recompensa final para todas as ações que contribuíram para ela?

---

## 4. O primeiro passo: Processos de Decisão de Markov (MDPs)

Antes de estudar RL completo, começamos com um cenário simplificado:

> **MDPs — quando o agente conhece todas as recompensas e transições entre estados.**

Nos MDPs:

- sabemos exatamente como o ambiente funciona,  
- conhecemos as probabilidades de transição $T(s,a,s')$,  
- conhecemos as recompensas $R(s)$ ou $R(s,a,s')$,  
- podemos calcular a política ótima matematicamente.

Isso nos permite:

- formalizar o problema,  
- entender funções de valor,  
- estudar Equações de Bellman,  
- implementar Value Iteration.

---

## 5. O que vem depois dos MDPs?

Após dominar MDPs, removemos as suposições:

- o agente **não conhece** as transições,  
- o agente **não conhece** as recompensas,  
- o agente deve **explorar o mundo real** para aprender.

Esse é o RL completo, que será estudado nas Lectures 18 e 19.

---

## 6. Conexão com o Projeto da Unidade

Ao final da unidade, você implementará:

> Um agente capaz de jogar **jogos baseados em texto**, aprendendo apenas por reforço.

Esse projeto utiliza:

- MDPs,  
- Equações de Bellman,  
- Value Iteration,  
- Q-Learning.

---

## 7. Conclusão da Parte 1

Nesta parte entendemos:

- o que é RL,  
- por que ele é diferente do aprendizado supervisionado,  
- por que recompensas atrasadas tornam o problema difícil,  
- como MDPs servem como base matemática,  
- como isso se conecta ao projeto final da unidade.

Na próxima parte estudaremos:

- a definição formal de RL,  
- exemplos intuitivos,  
- e como RL se relaciona com controle de agentes.



<br><br>




<a id="lecture17_parte2"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 17 — Parte 2  
Learning to Control: Introdução ao Reinforcement Learning  
=============================

Nesta parte aprofundamos a motivação do RL e entendemos por que ele é fundamental para problemas de **controle**, **tomada de decisão sequencial** e **aprendizado baseado em interação com o ambiente**.

---

## 1. Reinforcement Learning: uma nova forma de aprender

A professora inicia explicando que muitos já ouviram falar de RL por causa de sistemas como **AlphaGo**, que derrotou campeões mundiais sem receber instruções passo a passo.

O RL é diferente do aprendizado supervisionado porque:

- não existe rótulo para cada ação,  
- não existe supervisão contínua,  
- o agente deve **descobrir sozinho** quais ações levam ao sucesso.

Em RL:

> O agente aprende **experimentando**, **explorando** e **observando consequências**.

---

## 2. Por que RL é diferente do aprendizado supervisionado?

No aprendizado supervisionado:

- cada exemplo possui uma resposta correta,  
- o modelo aprende diretamente a função alvo.

No RL:

- o agente toma ações sem saber se são boas,  
- só descobre depois, quando recebe a recompensa final,  
- deve aprender a **propagar a recompensa final para trás**.

Essa diferença é o coração do RL.

---

## 3. Exemplos intuitivos de RL

### 3.1 O rato no labirinto  
O rato:

- tenta caminhos diferentes,  
- não recebe recompensa a cada passo,  
- só recebe quando encontra comida.

### 3.2 Jogos de computador  
O agente:

- pode receber pequenas recompensas intermediárias,  
- mas o que realmente importa é **vencer ou perder**.

### 3.3 Robôs navegando ambientes  
O robô:

- toma ações com custos diferentes,  
- pode receber pequenas recompensas,  
- mas o objetivo é **chegar ao destino**.

### 3.4 Marketing e interação com clientes  
O agente (empresa):

- pode enviar e-mail, ligar, mandar presente, etc.,  
- cada ação tem custo,  
- o que importa é **fechar o contrato**.

---

## 4. O desafio central do RL

O problema é:

> Como aprender a partir de recompensas atrasadas?

O agente deve:

- explorar ações,  
- experimentar caminhos,  
- descobrir o que funciona,  
- aprender a partir de sucessos e fracassos.

Esse processo é o que torna RL mais complexo e mais poderoso que o aprendizado supervisionado.

---

## 5. A necessidade de formalização

Para estudar RL matematicamente, precisamos:

- definir estados $s$,  
- definir ações $a$,  
- definir transições $T(s,a,s')$,  
- definir recompensas $R(s)$ ou $R(s,a,s')$.

Isso nos leva ao conceito de:

> **Processos de Decisão de Markov (MDPs)**

que serão estudados formalmente na Parte 3.

---

## 6. Conexão com controle

A professora explica que RL é, essencialmente:

> **Aprender a controlar um agente em um ambiente incerto.**

O agente deve:

- decidir ações,  
- prever consequências,  
- maximizar recompensas futuras.

Isso conecta RL com:

- controle ótimo,  
- tomada de decisão sequencial,  
- planejamento probabilístico.

---

## 7. Conclusão da Parte 2

Nesta parte entendemos:

- por que RL é diferente do aprendizado supervisionado,  
- como RL se aplica a jogos, robótica e marketing,  
- por que recompensas atrasadas tornam o problema difícil,  
- como RL é essencialmente um problema de controle,  
- por que precisamos formalizar o problema usando MDPs.

Na próxima parte estudaremos:

- Estados,  
- Ações,  
- Transições,  
- Recompensas,  
- A definição formal de um MDP.




<br><br>




<a id="lecture17_parte3"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 17 — Parte 3  
Terminologia Fundamental do Reinforcement Learning  
=============================

Nesta parte formalizamos os conceitos essenciais que compõem um  
**Processo de Decisão de Markov (MDP)**.  
Assim como na Lecture 16, esta seção estabelece a linguagem matemática que será usada nas partes seguintes.

Os quatro elementos fundamentais são:

1. **Estados**  
2. **Ações**  
3. **Transições**  
4. **Recompensas**

---

## 1. Estados — $s$ e conjunto de estados — $S$

Um **estado** representa uma configuração do ambiente.

Usamos:

- $s$ para um estado individual,  
- $S$ para o conjunto de todos os estados.

Nesta lecture assumimos:

> **Todos os estados são observáveis.**

Ou seja, o agente sabe exatamente onde está no ambiente.

Exemplo:

- Em um grid com 8 posições, cada posição é um estado.  
- Estados especiais podem representar perigo, objetivo ou paredes.

---

## 2. Ações — $a$ e conjunto de ações — $A$

Uma **ação** é algo que o agente pode fazer em um estado.

Usamos:

- $a$ para uma ação individual,  
- $A$ para o conjunto de ações possíveis.

Exemplo típico em um grid:

$$
A = \{\text{cima},\ \text{baixo},\ \text{esquerda},\ \text{direita}\}
$$

Se o agente tenta se mover contra uma parede:

> Ele permanece no mesmo estado.

---

## 3. Transições — $T(s,a,s')$

O ambiente é **não determinístico**.

Isso significa que:

> Mesmo que o agente escolha uma ação, ele pode acabar em estados diferentes.

A função de transição é definida como:

$$
T(s,a,s') = P(s' \mid s,a)
$$

Exemplo clássico:

- 80% de chance de ir na direção desejada,  
- 10% de chance de desviar para a esquerda,  
- 10% de chance de desviar para a direita.

Assim, se o agente está em $s$ e escolhe ação $a$:

- ele pode terminar em $s'$, $s''$, ou até permanecer em $s$.

---

## 4. Recompensas — $R(s)$ ou $R(s,a,s')$

A recompensa define o que é “bom” ou “ruim” para o agente.

Pode ser definida de duas formas:

### 4.1 Recompensa por estado  
$$
R(s)
$$

Exemplo:

- estado objetivo: $R = +1$  
- estado perigoso: $R = -1$

### 4.2 Recompensa por transição  
$$
R(s,a,s')
$$

Exemplo:

- custo por movimento: $-0.01$  
- custo por ação de marketing: depende da ação tomada  
- recompensa final: depende do estado alcançado

Ambas são válidas e usadas em diferentes problemas.

---

## 5. O MDP completo

Um MDP é definido pelo quádruplo:

$$
(S,\ A,\ T,\ R)
$$

onde:

- $S$ — conjunto de estados  
- $A$ — conjunto de ações  
- $T$ — função de transição  
- $R$ — função de recompensa  

Podemos adicionar:

- estado inicial $s_0$,  
- estados terminais (absorventes),  
- restrições de movimento.

Mas a estrutura fundamental permanece a mesma.

---

## 6. Por que essa terminologia é essencial?

Porque ela permite:

- formalizar o problema de RL,  
- definir funções de valor,  
- escrever Equações de Bellman,  
- implementar Value Iteration,  
- conectar MDPs ao RL completo.

Sem essa base, não é possível avançar para as partes seguintes.

---

## 7. Conclusão da Parte 3

Nesta parte aprendemos:

- o que são estados, ações, transições e recompensas,  
- como o ambiente é modelado probabilisticamente,  
- como definir formalmente um MDP,  
- por que essa estrutura é essencial para RL.

Na próxima parte estudaremos:

- Funções de utilidade,  
- Recompensas descontadas,  
- Por que precisamos do fator de desconto $\gamma$,  
- Como garantir que a utilidade seja finita.




<Br><br>




<a id="lecture17_parte4"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 17 — Parte 4  
Função de Utilidade e Recompensas Descontadas  
=============================

Nesta parte formalizamos como o agente deve **agregar recompensas ao longo do tempo**.  
O objetivo é responder:

> Como medir a qualidade de uma trajetória de estados em um MDP?

Para isso, introduzimos a **função de utilidade**, que define o valor total acumulado pelo agente.

---

## 1. O problema: como somar recompensas?

Uma trajetória é uma sequência de estados:

$$
s_0,\ s_1,\ s_2,\ \dots
$$

Uma primeira ideia seria somar todas as recompensas:

$$
U = \sum_{t=0}^{\infty} R(s_t)
$$

Mas isso apresenta dois problemas sérios:

### ❌ 1. A soma pode ser infinita  
Se o agente recebe pequenas recompensas continuamente, a soma diverge.

### ❌ 2. O comportamento deixa de ser estacionário  
O valor de um estado passa a depender do **tempo**, não apenas do estado.

Isso viola a propriedade de Markov que queremos manter.

---

## 2. Tentativa alternativa: horizonte finito

Outra ideia seria limitar a soma:

$$
U = \sum_{t=0}^{n} R(s_t)
$$

Mas isso também é problemático:

- o comportamento passa a depender de **quantos passos restam**,  
- o agente pode agir de forma **arriscada** quando está perto do fim,  
- a política deixa de ser estacionária.

Por isso, essa abordagem não é usada em MDPs.

---

## 3. Solução correta: Recompensas Descontadas

Para garantir que a utilidade seja finita e bem comportada, usamos **desconto exponencial**:

$$
U = \sum_{t=0}^{\infty} \gamma^t R(s_t)
$$

onde:

- $0 < \gamma < 1$ é o **fator de desconto**,  
- recompensas futuras valem menos que recompensas imediatas.

---

## 4. Intuição do fator de desconto

O fator $\gamma$ modela um comportamento natural:

- valorizamos mais o que recebemos **agora**,  
- valorizamos menos o que recebemos **no futuro**.

Exemplo intuitivo:

- assistir um filme agora vs. estudar para uma prova distante,  
- comer um doce agora vs. manter dieta para um benefício futuro.

O agente se torna **ganancioso**, mas de forma controlada.

---

## 5. Prova de que a utilidade é limitada

Se $R(s_t) \le R_{\max}$, então:

$$
U \le \sum_{t=0}^{\infty} \gamma^t R_{\max}
$$

Fatorando:

$$
U \le R_{\max} \sum_{t=0}^{\infty} \gamma^t
$$

A soma é uma série geométrica:

$$
\sum_{t=0}^{\infty} \gamma^t = \frac{1}{1 - \gamma}
$$

Portanto:

$$
U \le \frac{R_{\max}}{1 - \gamma}
$$

### ✔ A utilidade é finita  
### ✔ O problema é bem definido  
### ✔ Podemos aplicar Bellman Equations  

---

## 6. Consequências importantes

### 6.1 O comportamento do agente se torna estacionário  
A melhor ação depende **apenas do estado atual**, não do tempo.

### 6.2 O problema se torna matematicamente tratável  
Podemos definir funções de valor e políticas ótimas.

### 6.3 O algoritmo de Value Iteration passa a convergir  
Sem desconto, a convergência não é garantida.

---

## 7. Conclusão da Parte 4

Nesta parte aprendemos:

- por que somar recompensas diretamente não funciona,  
- como recompensas descontadas resolvem o problema,  
- como o fator $\gamma$ garante finitude,  
- como isso prepara o terreno para políticas ótimas e Equações de Bellman.

Na próxima parte estudaremos:

- Políticas ótimas,  
- Funções de valor,  
- Relação entre política e utilidade.




<br><br>



<a id="lecture17_parte5"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 17 — Parte 5  
Políticas e Funções de Valor  
=============================

Nesta parte concluímos a Lecture 17 introduzindo dois conceitos fundamentais para resolver MDPs:

1. **Política** — como o agente decide suas ações.  
2. **Função de valor** — como medir a qualidade de um estado.

Esses conceitos são essenciais para formular as **Equações de Bellman** e, posteriormente, o algoritmo de **Value Iteration**.

---

## 1. O que é uma política?

Uma **política** define o comportamento do agente.

### Definição:
Uma política é uma função:

$$
\pi(s) = a
$$

que indica qual ação $a$ deve ser tomada quando o agente está no estado $s$.

### Política ótima:
Chamamos de política ótima:

$$
\pi^*
$$

a política que maximiza a utilidade esperada ao longo do tempo.

---

## 2. Por que políticas são necessárias?

Porque o ambiente é **não determinístico**:

- o agente escolhe uma ação,  
- mas pode acabar em estados diferentes,  
- com probabilidades diferentes.

Assim, a política deve considerar:

- recompensas imediatas,  
- recompensas futuras,  
- transições probabilísticas.

---

## 3. Função de valor

A função de valor mede **o valor esperado** de estar em um estado, seguindo uma política.

### Definição geral:

$$
V^\pi(s) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t R(s_t) \mid s_0 = s,\ \pi \right]
$$

Ou seja:

- começamos no estado $s$,  
- seguimos a política $\pi$,  
- acumulamos recompensas descontadas,  
- e calculamos o valor esperado.

---

## 4. Função de valor ótima

A função de valor ótima é:

$$
V^*(s) = \max_{\pi} V^\pi(s)
$$

Ela representa:

> O valor máximo possível que podemos obter ao começar no estado $s$ e agir de forma ótima.

---

## 5. Relação entre política e valor

A política ótima é aquela que escolhe ações que maximizam o valor esperado:

$$
\pi^*(s) = \arg\max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s') \right]
$$

Essa expressão diz:

- para cada estado $s$,  
- escolha a ação $a$ que leva aos estados futuros mais valiosos,  
- considerando transições probabilísticas,  
- recompensas imediatas,  
- e o fator de desconto $\gamma$.

---

## 6. Exemplo intuitivo

Considere o grid da Parte 1:

- estado com recompensa $+1$ → valor alto  
- estado com punição $-1$ → valor baixo  
- estados próximos ao $+1$ → valor intermediário  
- estados próximos ao $-1$ → valor negativo  

A política ótima:

- move o agente para estados com maior valor esperado,  
- evitando estados com valor negativo.

---

## 7. Por que precisamos da função de valor?

Porque ela permite:

- comparar estados,  
- avaliar políticas,  
- formular Equações de Bellman,  
- executar Value Iteration,  
- resolver MDPs de forma eficiente.

Sem a função de valor, não há como medir “quão bom” é um estado.

---

## 8. Conclusão da Parte 5

Nesta parte aprendemos:

- o que é uma política,  
- o que é uma política ótima,  
- como definir a função de valor,  
- como definir a função de valor ótima,  
- como políticas e valores se relacionam,  
- como isso prepara o terreno para as Equações de Bellman.

Com isso, concluímos a Parte 5.

Na Parte 6 estudaremos:

- a intuição das Equações de Bellman,  
- como valores são propagados pelo ambiente.




<br><br>



<a id="lecture17_parte6"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 17 — Parte 6  
Equações de Bellman — Intuição  
=============================

Nesta parte iniciamos a transição entre a definição de políticas e funções de valor (Parte 5) e o algoritmo de **Value Iteration** (Lecture 18).  
O objetivo é entender **a intuição por trás das Equações de Bellman**, antes de formalizá-las matematicamente.

---

## 1. O problema: como propagar recompensas?

Até agora sabemos:

- o agente recebe recompensas em estados específicos,  
- recompensas podem ser positivas ou negativas,  
- recompensas são descontadas por $\gamma$,  
- queremos calcular o valor de cada estado.

Mas surge a pergunta central:

> **Como transmitir a qualidade de um estado terminal para todos os estados que levam até ele?**

Essa é a função das Equações de Bellman.

---

## 2. Intuição básica

Considere o grid da Lecture 17:

- O estado com recompensa $+1$ é excelente.  
- O estado com recompensa $-1$ é ruim.  
- Estados próximos ao $+1$ devem ter valor alto.  
- Estados próximos ao $-1$ devem ter valor baixo.

A pergunta é:

> **Como calcular esses valores intermediários?**

A resposta é:

> **Propagando valores dos estados futuros para os estados atuais.**

---

## 3. A ideia central de Bellman

A Equação de Bellman captura a seguinte ideia:

> **O valor de um estado é igual à recompensa imediata  
> mais o valor esperado dos estados futuros.**

Ou seja:

- se o estado é bom, seu valor é alto;  
- se o estado leva a estados bons, seu valor também é alto;  
- se o estado leva a estados ruins, seu valor é baixo.

Bellman transforma essa intuição em uma equação recursiva.

---

## 4. Exemplo intuitivo no grid

Imagine:

- $s_G$ é o estado com recompensa $+1$,  
- $s_B$ é o estado com recompensa $-1$,  
- $s$ é um estado vizinho de $s_G$.

O valor de $s$ deve ser aproximadamente:

$$
V(s) \approx \gamma \cdot V(s_G)
$$

porque:

- a recompensa imediata é zero,  
- mas o estado leva rapidamente ao estado bom,  
- e o valor é descontado por $\gamma$.

Se $\gamma = 0.9$, então:

$$
V(s) \approx 0.9 \cdot 1 = 0.9
$$

Esse valor será propagado para estados mais distantes:

$$
V(s') \approx \gamma \cdot V(s) = 0.9 \cdot 0.9 = 0.81
$$

E assim por diante.

---

## 5. Propagação como “ondas de valor”

A professora descreve Bellman como:

> **Uma onda de valores que se espalha pelo ambiente.**

Essa onda:

- começa nos estados terminais,  
- se espalha para estados vizinhos,  
- continua se espalhando até preencher todo o grid.

Essa propagação é exatamente o que **Value Iteration** implementa.

---

## 6. Por que Bellman é essencial?

Porque ele permite:

- calcular valores ótimos,  
- comparar estados,  
- derivar políticas ótimas,  
- resolver MDPs de forma exata.

Bellman é a base matemática de todo RL moderno.

---

## 7. Conclusão da Parte 6

Nesta parte entendemos:

- a intuição das Equações de Bellman,  
- como valores são propagados pelo ambiente,  
- como estados bons influenciam estados vizinhos,  
- como isso prepara o terreno para a formulação matemática completa.

Na próxima parte estudaremos:

- **Bellman Optimality**,  
- a forma matemática completa da equação,  
- como ela define $V^*(s)$.




<br><br>



<a id="lecture17_parte7"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 17 — Parte 7  
Bellman Optimality — Propagação de Valores  
=============================

Na Parte 6 entendemos a **intuição** das Equações de Bellman:  
o valor de um estado depende da recompensa imediata e dos valores dos estados futuros.

Agora formalizamos essa ideia matematicamente e introduzimos a **Equação de Bellman Ótima**, que define o valor ótimo de cada estado.

---

## 1. Relembrando a intuição

A professora explicou que:

- estados com recompensa positiva irradiam valores altos,  
- estados com punição irradiam valores baixos,  
- estados intermediários recebem valores proporcionais à proximidade dos estados bons ou ruins.

Bellman captura essa propagação de forma exata.

---

## 2. Bellman para uma política fixa

Se seguimos uma política $\pi$, o valor de um estado é:

$$
V^\pi(s) = R(s) + \gamma \sum_{s'} T(s,\pi(s),s')\, V^\pi(s')
$$

Essa equação diz:

- receba a recompensa imediata $R(s)$,  
- depois considere o valor dos estados futuros,  
- ponderados pelas probabilidades de transição.

---

## 3. Bellman Optimality — Forma Completa

Para encontrar a política ótima, precisamos da função de valor ótima:

$$
V^*(s)
$$

Ela é definida pela **Equação de Bellman Ótima**:

$$
V^*(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s') \right]
$$

Essa equação expressa:

- para cada estado $s$,  
- considere todas as ações possíveis $a$,  
- calcule o valor esperado de cada ação,  
- escolha a ação que maximiza esse valor.

---

## 4. Interpretação da Equação Ótima

A Equação de Bellman Ótima é um **sistema de equações acopladas**:

- cada estado depende dos valores dos estados futuros,  
- que dependem dos valores dos estados seguintes,  
- e assim por diante.

O valor ótimo é o **ponto fixo** dessa relação recursiva.

---

## 5. Relação entre Bellman e política ótima

A política ótima é:

$$
\pi^*(s) = \arg\max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s') \right]
$$

Ou seja:

> **A melhor ação é aquela que leva ao maior valor futuro esperado.**

---

## 6. Exemplo intuitivo no grid

Considere o grid da Lecture 17:

- O estado com recompensa $+1$ tem valor alto.  
- O estado com recompensa $-1$ tem valor baixo.  
- Estados próximos ao $+1$ têm valores intermediários.  
- Estados próximos ao $-1$ têm valores negativos.

Bellman:

- pega o valor dos estados terminais,  
- propaga para os vizinhos,  
- propaga para os vizinhos dos vizinhos,  
- até preencher todo o grid.

Essa propagação é exatamente o que **Value Iteration** implementa.

---

## 7. Por que Bellman Optimality é tão importante?

Porque ele permite:

- resolver MDPs de forma exata,  
- calcular valores ótimos,  
- derivar políticas ótimas,  
- fundamentar algoritmos como Value Iteration e Policy Iteration,  
- preparar o terreno para Q-Learning (Lecture 19).

Bellman é o coração matemático do RL.

---

## 8. Conclusão da Parte 7

Nesta parte aprendemos:

- a forma completa da Equação de Bellman Ótima,  
- como ela define $V^*(s)$,  
- como derivar a política ótima $\pi^*(s)$,  
- como valores são propagados pelo ambiente,  
- como Bellman prepara o terreno para **Value Iteration**.

Na próxima parte estudaremos:

- como Bellman leva diretamente ao algoritmo de Value Iteration,  
- e como isso fecha a Lecture 17 e abre a Lecture 18.




<br><br>



<a id="lecture17_parte8"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 17 — Parte 8  
Preparação para Value Iteration  
=============================

Nesta parte concluímos a Lecture 17 conectando as **Equações de Bellman** ao algoritmo que será estudado na Lecture 18: **Value Iteration**.

O objetivo aqui é entender:

1. Por que Bellman define um sistema de equações acopladas.  
2. Por que esse sistema pode ser resolvido iterativamente.  
3. Como Bellman leva naturalmente ao algoritmo de Value Iteration.  
4. Como isso fecha a Lecture 17 e abre a Lecture 18.

---

## 1. Bellman como um sistema de equações

A Equação de Bellman Ótima:

$$
V^*(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s') \right]
$$

é uma **equação recursiva**.

Isso significa que:

- o valor de cada estado depende dos valores dos estados futuros,  
- que dependem dos valores dos estados seguintes,  
- e assim por diante.

Portanto:

> **Resolver Bellman significa resolver um sistema de equações acopladas.**

---

## 2. Por que não resolvemos Bellman diretamente?

Resolver Bellman diretamente exigiria:

- montar um sistema linear (ou não linear),  
- resolver simultaneamente todas as equações,  
- lidar com dependências complexas entre estados.

Isso é inviável para:

- ambientes grandes,  
- ambientes contínuos,  
- ambientes com milhares ou milhões de estados.

Por isso, usamos uma abordagem **iterativa**.

---

## 3. A ideia da solução iterativa

A professora explica que:

> **Podemos começar com uma estimativa inicial para $V(s)$  
> e aplicar Bellman repetidamente até que os valores convirjam.**

Essa ideia é a base do algoritmo de **Value Iteration**.

Cada aplicação de Bellman:

- melhora a estimativa de $V(s)$,  
- propaga valores dos estados futuros para os estados atuais,  
- aproxima o valor ótimo.

---

## 4. Convergência garantida

A razão pela qual essa abordagem funciona é:

- Bellman é uma **contração** matemática,  
- o fator de desconto $\gamma$ garante que a influência dos estados futuros diminui,  
- a atualização converge para um **ponto fixo**,  
- esse ponto fixo é exatamente $V^*(s)$.

Portanto:

> **Aplicar Bellman repetidamente sempre converge para o valor ótimo.**

---

## 5. Intuição final antes do algoritmo

A professora usa o grid como exemplo:

- Começamos com todos os valores iguais (ex.: zero).  
- Aplicamos Bellman.  
- Os estados terminais propagam seus valores.  
- Estados vizinhos recebem valores intermediários.  
- A “onda de valor” se espalha pelo ambiente.  
- Eventualmente, todos os estados estabilizam.

Essa estabilização é:

$$
V^*(s)
$$

E, uma vez que temos $V^*(s)$, podemos extrair:

$$
\pi^*(s)
$$

a política ótima.

---

## 6. Conexão direta com a Lecture 18

A professora encerra a Lecture 17 dizendo:

> “Com base nessa discussão, seremos capazes de formular o algoritmo de iteração de valores que pode resolver esses processos de decisão de Markov.”

Ou seja:

- Lecture 17 introduz Bellman,  
- Lecture 18 implementa Bellman iterativamente,  
- **Value Iteration é a aplicação prática de Bellman Optimality**.

---

## 7. Conclusão da Parte 8

Nesta parte entendemos:

- por que Bellman define um sistema de equações acopladas,  
- por que esse sistema pode ser resolvido iterativamente,  
- como Bellman leva diretamente ao algoritmo de Value Iteration,  
- como isso fecha a Lecture 17 e abre a Lecture 18.

Com isso, concluímos a Lecture 17.




<br><br>



<a id="lecture18_parte1"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 18 — Parte 1  
Bellman Optimality — Revisão Formal  
=============================

Nesta lecture começamos a aplicar, de forma operacional, tudo o que foi desenvolvido na Lecture 17.  
O objetivo desta primeira parte é **formalizar completamente** a Equação de Bellman Ótima e preparar o terreno para o algoritmo de **Value Iteration**.

---

## 1. Relembrando o objetivo

Queremos resolver um MDP:

$$
(S,\ A,\ T,\ R)
$$

Encontrando:

- a **função de valor ótima** $V^*(s)$  
- a **política ótima** $\pi^*(s)$  

que maximizam a utilidade esperada ao longo do tempo.

---

## 2. A função de valor ótima

A função de valor ótima é definida como:

$$
V^*(s) = \max_{\pi} V^\pi(s)
$$

onde:

$$
V^\pi(s) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t R(s_t) \mid s_0 = s,\ \pi \right]
$$

Ou seja:

> $V^*(s)$ é o valor máximo possível que podemos obter ao começar no estado $s$ e agir de forma ótima.

---

## 3. A Equação de Bellman Ótima

A Equação de Bellman Ótima formaliza a relação recursiva entre estados:

$$
V^*(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s') \right]
$$

Essa equação expressa:

- a recompensa imediata $R(s,a)$,  
- mais o valor esperado dos estados futuros,  
- ponderado pelas probabilidades de transição,  
- escolhendo a ação que maximiza esse valor.

---

## 4. Interpretação geométrica e probabilística

A Equação de Bellman Ótima combina:

### **Geometria**
- valores altos irradiam dos estados bons,  
- valores baixos irradiam dos estados ruins,  
- estados intermediários recebem valores proporcionais à proximidade dos estados terminais.

### **Probabilidade**
- cada ação leva a múltiplos estados possíveis,  
- cada estado futuro contribui proporcionalmente à sua probabilidade $T(s,a,s')$.

Bellman captura essa propagação de forma exata.

---

## 5. Bellman como operador

Podemos definir o **operador de Bellman**:

$$
(BV)(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V(s') \right]
$$

A função de valor ótima é o **ponto fixo** desse operador:

$$
V^* = BV^*
$$

Essa visão é essencial para entender a convergência do Value Iteration.

---

## 6. Por que revisar Bellman antes do algoritmo?

Porque Value Iteration nada mais é do que:

> Aplicar o operador de Bellman repetidamente  
> até que a função de valor convirja para o ponto fixo $V^*$.

Ou seja:

$$
V_{k+1}(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V_k(s') \right]
$$

Essa é a forma iterativa da Equação de Bellman Ótima.

---

## 7. Conclusão da Parte 1

Nesta parte revisamos:

- a definição formal de $V^*(s)$,  
- a Equação de Bellman Ótima,  
- a interpretação geométrica e probabilística,  
- o operador de Bellman e seu ponto fixo,  
- a conexão direta entre Bellman e Value Iteration.

Na próxima parte estudaremos:

- a forma iterativa de Bellman,  
- o algoritmo completo de **Value Iteration**,  
- e por que ele sempre converge para $V^*(s)$.




<br><br>



<a id="lecture18_parte2"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 18 — Parte 2  
Value Iteration — Atualização Recursiva  
=============================

Nesta parte introduzimos o algoritmo central da Lecture 18: **Value Iteration**.  
Ele é a implementação prática da Equação de Bellman Ótima e permite resolver MDPs de forma eficiente, mesmo quando o número de estados é grande.

---

## 1. Ideia central do Value Iteration

A Equação de Bellman Ótima define $V^*(s)$ como:

$$
V^*(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s') \right]
$$

Value Iteration aplica essa equação **iterativamente**, aproximando $V^*(s)$ a cada passo.

A atualização é:

$$
V_{k+1}(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V_k(s') \right]
$$

Ou seja:

> Começamos com uma estimativa inicial $V_0(s)$  
> e aplicamos Bellman repetidamente até que os valores convirjam.

---

## 2. Por que isso funciona?

Porque o operador de Bellman:

$$
(BV)(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V(s') \right]
$$

é uma **contração** quando $0 < \gamma < 1$.

Isso significa:

- cada aplicação de Bellman aproxima $V(s)$ do ponto fixo,  
- o ponto fixo é exatamente $V^*(s)$,  
- portanto, a sequência $V_0, V_1, V_2, \dots$ converge para $V^*$.

---

## 3. Escolha da inicialização

A inicialização mais comum é:

$$
V_0(s) = 0
$$

Mas outras escolhas são possíveis:

- valores aleatórios,  
- aproximações heurísticas,  
- estimativas baseadas na estrutura do ambiente.

A inicialização não afeta o valor final — apenas a velocidade de convergência.

---

## 4. Atualização estado por estado

Para cada estado $s$:

1. Consideramos todas as ações $a$.  
2. Para cada ação, calculamos:

$$
Q(s,a) = R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V_k(s')
$$

3. Escolhemos a melhor ação:

$$
V_{k+1}(s) = \max_a Q(s,a)
$$

Essa é a essência do Value Iteration.

---

## 5. Critério de parada

Value Iteration para quando:

$$
\max_s |V_{k+1}(s) - V_k(s)| < \varepsilon
$$

onde $\varepsilon$ é um limiar pequeno (ex.: $10^{-3}$).

Isso significa:

> Os valores mudam tão pouco que já estão próximos de $V^*(s)$.

---

## 6. Extração da política ótima

Depois que $V^*(s)$ converge, extraímos a política ótima:

$$
\pi^*(s) = \arg\max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s') \right]
$$

Ou seja:

> A melhor ação é aquela que leva aos estados futuros mais valiosos.

---

## 7. Intuição geométrica

Value Iteration funciona como:

> **Uma onda de valores que se propaga pelo ambiente.**

- Estados terminais definem valores iniciais.  
- Estados vizinhos recebem valores intermediários.  
- A onda se espalha até preencher todo o grid.  
- Quando a onda estabiliza, temos $V^*(s)$.

---

## 8. Conclusão da Parte 2

Nesta parte aprendemos:

- a forma iterativa da Equação de Bellman Ótima,  
- como Value Iteration atualiza $V(s)$ recursivamente,  
- por que o operador de Bellman garante convergência,  
- como extrair a política ótima após a convergência.

Na próxima parte estudaremos:

- a prova de convergência,  
- a propriedade de contração,  
- e por que Value Iteration sempre encontra $V^*(s)$.




<br><br>



<a id="lecture18_parte3"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 18 — Parte 3  
Convergência — Contração e Ponto Fixo  
=============================

Nesta parte entendemos **por que** o algoritmo de Value Iteration sempre converge para a função de valor ótima $V^*(s)$.  
A chave para essa garantia é a propriedade de **contração** do operador de Bellman.

---

## 1. O operador de Bellman

Definimos o operador de Bellman como:

$$
(BV)(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V(s') \right]
$$

Esse operador transforma uma função de valor $V$ em uma nova função de valor $BV$.

A função de valor ótima é o **ponto fixo** desse operador:

$$
V^* = BV^*
$$

Ou seja:

> Aplicar Bellman sobre $V^*$ devolve o próprio $V^*$.

---

## 2. O que significa ser uma contração?

Um operador $B$ é uma **contração** se:

$$
\| BV - BU \|_\infty \le \gamma \| V - U \|_\infty
$$

para qualquer par de funções de valor $V$ e $U$.

Aqui:

- $\| \cdot \|_\infty$ é a norma máximo,  
- $0 < \gamma < 1$ é o fator de desconto.

Essa desigualdade significa:

> Bellman aproxima funções de valor umas das outras, reduzindo a distância entre elas por um fator $\gamma$.

---

## 3. Consequência da contração: convergência garantida

Se $B$ é uma contração, então:

- existe um único ponto fixo $V^*$,  
- qualquer sequência gerada por $V_{k+1} = BV_k$ converge para esse ponto fixo,  
- a convergência é geométrica (rápida).

Portanto:

$$
V_k \longrightarrow V^*
$$

independentemente da inicialização $V_0$.

---

## 4. Prova intuitiva da contração

Considere dois valores $V$ e $U$.

Para qualquer estado $s$:

$$
(BV)(s) - (BU)(s)
$$

envolve:

- a mesma recompensa imediata $R(s,a)$,  
- a mesma soma ponderada de transições,  
- apenas os valores futuros mudam.

Como o fator de desconto $\gamma$ multiplica todos os valores futuros:

$$
| (BV)(s) - (BU)(s) | \le \gamma \max_{s'} | V(s') - U(s') |
$$

Portanto:

$$
\| BV - BU \|_\infty \le \gamma \| V - U \|_\infty
$$

Bellman é uma contração.

---

## 5. Convergência do Value Iteration

Value Iteration aplica Bellman repetidamente:

$$
V_{k+1}(s) = (BV_k)(s)
$$

Como $B$ é uma contração:

$$
\| V_{k+1} - V^* \|_\infty \le \gamma \| V_k - V^* \|_\infty
$$

Isso significa:

- cada iteração aproxima $V_k$ de $V^*$,  
- a distância diminui por um fator $\gamma$,  
- a convergência é garantida e rápida.

---

## 6. Critério de parada revisitado

O critério:

$$
\max_s |V_{k+1}(s) - V_k(s)| < \varepsilon
$$

é uma forma prática de detectar que:

> A sequência já está suficientemente próxima do ponto fixo $V^*$.

---

## 7. Intuição geométrica

A contração garante que:

- a “onda de valor” não oscila,  
- não diverge,  
- não explode,  
- sempre se estabiliza.

O valor ótimo emerge como o único estado estável da dinâmica de Bellman.

---

## 8. Conclusão da Parte 3

Nesta parte entendemos:

- o operador de Bellman,  
- a propriedade de contração,  
- o ponto fixo $V^*$,  
- por que Value Iteration sempre converge,  
- por que a convergência é rápida e independente da inicialização.

Na próxima parte estudaremos:

- como extrair a política ótima $\pi^*(s)$  
- a partir da função de valor ótima $V^*(s)$.




<br><br>



<a id="lecture18_parte4"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 18 — Parte 4  
Extração da Política Ótima  
=============================

Nesta parte aprendemos como **extrair a política ótima** depois que o algoritmo de **Value Iteration** converge para a função de valor ótima $V^*(s)$.

Até agora, Value Iteration nos deu:

$$
V^*(s)
$$

Mas nosso objetivo final ao resolver um MDP é:

> **Encontrar a ação ótima em cada estado.**

Ou seja, construir a política ótima $\pi^*(s)$.

---

## 1. A política ótima depende de $V^*(s)$

Depois que temos $V^*(s)$, a política ótima é definida como:

$$
\pi^*(s) = \arg\max_{a} \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s') \right]
$$

Essa expressão diz:

- para cada estado $s$,  
- considere todas as ações possíveis $a$,  
- calcule o valor esperado de cada ação,  
- escolha a ação que maximiza esse valor.

---

## 2. Intuição: escolher a melhor ação olhando para o futuro

A política ótima não olha apenas para:

- a recompensa imediata $R(s,a)$.

Ela também considera:

- os estados futuros,  
- seus valores $V^*(s')$,  
- e as probabilidades de transição $T(s,a,s')$.

Ou seja:

> **A melhor ação é aquela que leva aos estados futuros mais valiosos.**

---

## 3. Relação entre $V^*(s)$ e $Q^*(s,a)$

Podemos definir o valor ótimo de uma ação:

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s')
$$

Então:

$$
V^*(s) = \max_a Q^*(s,a)
$$

E a política ótima é simplesmente:

$$
\pi^*(s) = \arg\max_a Q^*(s,a)
$$

Essa relação será essencial na Lecture 19, quando introduzirmos **Q-Learning**.

---

## 4. Processo de extração da política

Para cada estado $s$:

1. Para cada ação $a$, compute:

$$
Q(s,a) = R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s')
$$

2. Escolha a ação com maior valor:

$$
\pi^*(s) = \arg\max_a Q(s,a)
$$

Esse processo é simples, direto e eficiente.

---

## 5. Exemplo intuitivo no grid

Considere o grid da Lecture 17:

- estados com $+1$ irradiam valores altos,  
- estados com $-1$ irradiam valores baixos.

Depois que $V^*(s)$ converge:

- estados próximos ao $+1$ terão ações que os aproximam do objetivo,  
- estados próximos ao $-1$ terão ações que os afastam da punição.

A política ótima emerge naturalmente da estrutura dos valores.

---

## 6. Por que a extração da política é separada da iteração de valores?

Porque:

- Value Iteration calcula **valores**,  
- e só depois extraímos **ações**.

Separar essas duas etapas torna o algoritmo:

- mais simples,  
- mais modular,  
- mais eficiente,  
- mais fácil de implementar.

---

## 7. Conclusão da Parte 4

Nesta parte aprendemos:

- como extrair a política ótima a partir de $V^*(s)$,  
- a relação entre $V^*(s)$ e $Q^*(s,a)$,  
- por que a política ótima escolhe ações que maximizam valores futuros,  
- como essa etapa completa o processo de Value Iteration.

Na próxima parte estudaremos:

- exemplos concretos em **Gridworld**,  
- visualização da propagação de valores,  
- e como a política ótima emerge no ambiente.



<br><br>



<a id="lecture18_parte5"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 18 — Parte 5  
Exemplos em Gridworld — Propagação de Valores  
=============================

Nesta parte visualizamos como o algoritmo de **Value Iteration** funciona na prática, usando o ambiente clássico **Gridworld**.  
O objetivo é entender como:

- valores são propagados pelo grid,  
- estados bons irradiam valores positivos,  
- estados ruins irradiam valores negativos,  
- a política ótima emerge naturalmente após a convergência.

---

## 1. O ambiente Gridworld

Considere um grid simples:

- células vazias → recompensa $0$  
- objetivo → recompensa $+1$  
- perigo → recompensa $-1$  
- paredes → estados inacessíveis  

O agente pode se mover:

$$
A = \{\text{cima},\ \text{baixo},\ \text{esquerda},\ \text{direita}\}
$$

E o movimento é **estocástico**:

- 80% de chance de ir na direção desejada,  
- 10% de chance de desviar para a esquerda,  
- 10% de chance de desviar para a direita.

---

## 2. Inicialização dos valores

Começamos com:

$$
V_0(s) = 0
$$

para todos os estados não terminais.

Os estados terminais já possuem:

- $V(s_G) = +1$  
- $V(s_B) = -1$

---

## 3. Primeira iteração de Bellman

Para cada estado $s$:

$$
V_1(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V_0(s') \right]
$$

Como $V_0(s') = 0$ para todos os estados não terminais:

- estados vizinhos do objetivo recebem valores positivos,  
- estados vizinhos do perigo recebem valores negativos.

A “onda de valor” começa a se formar.

---

## 4. Propagação ao longo das iterações

A cada iteração:

$$
V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V_k(s') \right]
$$

Os valores:

- aumentam conforme se aproximam do objetivo,  
- diminuem conforme se aproximam do perigo,  
- estabilizam quando a onda de valor atinge todo o grid.

---

## 5. Visualização da propagação

A professora mostra que:

- estados próximos ao $+1$ recebem valores como $0.9$, $0.81$, $0.72$, …  
- estados próximos ao $-1$ recebem valores como $-0.9$, $-0.81$, $-0.72$, …  

Com $\gamma = 0.9$:

$$
V(s_{\text{vizinho do objetivo}}) \approx 0.9 \cdot 1 = 0.9
$$

$$
V(s_{\text{vizinho do perigo}}) \approx 0.9 \cdot (-1) = -0.9
$$

Esses valores continuam se propagando até estabilizar.

---

## 6. Emergência da política ótima

Depois que $V^*(s)$ converge, extraímos a política:

$$
\pi^*(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s') \right]
$$

No grid:

- estados com valor alto apontam para o objetivo,  
- estados com valor baixo apontam para longe do perigo,  
- estados neutros escolhem caminhos que maximizam o valor futuro.

A política ótima emerge naturalmente da estrutura dos valores.

---

## 7. Intuição final

Gridworld mostra claramente que:

> **Value Iteration é um processo de difusão de valores.**

- estados terminais são fontes de valor,  
- o valor se espalha pelo ambiente,  
- a política ótima é simplesmente seguir o gradiente de valor.

---

## 8. Conclusão da Parte 5

Nesta parte entendemos:

- como valores são propagados no Gridworld,  
- como a estrutura do ambiente influencia $V^*(s)$,  
- como a política ótima emerge após a convergência,  
- como Value Iteration funciona visualmente e intuitivamente.

Na próxima parte estudaremos:

- a conexão entre Value Iteration, planejamento e controle,  
- e como esses conceitos se generalizam para ambientes maiores.



<br><br>



<a id="lecture18_parte6"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 18 — Parte 6  
Conexão com Planejamento e Controle  
=============================

Nesta parte concluímos a Lecture 18 conectando o algoritmo de **Value Iteration** com conceitos mais amplos de **planejamento**, **controle**, e **tomada de decisão sequencial**.  
O objetivo é entender como MDPs e Value Iteration se encaixam em uma visão mais geral de agentes inteligentes.

---

## 1. Value Iteration como planejamento

A professora explica que:

> **Value Iteration é um algoritmo de planejamento.**

Ou seja:

- ele calcula antecipadamente o valor de cada estado,  
- antes de o agente começar a agir no ambiente,  
- usando conhecimento completo das transições $T(s,a,s')$ e recompensas $R(s,a,s')$.

Isso é chamado de **planejamento offline**.

---

## 2. Planejamento vs. Aprendizado

Value Iteration assume:

- o ambiente é totalmente conhecido,  
- todas as transições são conhecidas,  
- todas as recompensas são conhecidas.

Mas em muitos problemas reais:

- o agente não conhece o ambiente,  
- não sabe as probabilidades de transição,  
- não sabe as recompensas.

Nesses casos, o agente deve **aprender** interagindo com o ambiente.

Essa transição do planejamento para o aprendizado é o tema da Lecture 19.

---

## 3. Conexão com controle

A professora destaca que:

> Resolver um MDP é resolver um problema de **controle ótimo**.

O agente deve:

- escolher ações,  
- prever consequências,  
- maximizar recompensas futuras.

Isso é exatamente o que sistemas de controle fazem:

- robôs,  
- drones,  
- veículos autônomos,  
- sistemas industriais.

MDPs fornecem uma base matemática para esses sistemas.

---

## 4. Value Iteration como solução de controle ótimo

Value Iteration encontra:

- a função de valor ótima $V^*(s)$,  
- a política ótima $\pi^*(s)$.

Essa política é:

> **A regra de controle ótima para o agente.**

Ela diz:

- qual ação tomar em cada estado,  
- para maximizar a utilidade esperada,  
- levando em conta incerteza e estocasticidade.

---

## 5. Planejamento em ambientes estocásticos

MDPs são especialmente importantes porque:

- o ambiente é incerto,  
- ações têm resultados probabilísticos,  
- o agente deve considerar múltiplos futuros possíveis.

Value Iteration incorpora essa incerteza diretamente:

$$
\gamma \sum_{s'} T(s,a,s')\, V^*(s')
$$

Essa soma ponderada é o coração do planejamento estocástico.

---

## 6. Conexão com agentes reais

A professora mostra que MDPs e Value Iteration aparecem em:

- robótica (navegação, manipulação),  
- economia (decisões sequenciais),  
- marketing (interação com clientes),  
- jogos (IA de adversários),  
- logística (planejamento de rotas),  
- sistemas de recomendação (decisões sequenciais).

Em todos esses casos:

> O agente deve planejar ações levando em conta incerteza e recompensas futuras.

---

## 7. Value Iteration como base para RL

A professora encerra explicando:

- Value Iteration é a solução exata quando o ambiente é conhecido.  
- Reinforcement Learning é a solução aproximada quando o ambiente é desconhecido.  
- RL aprende uma aproximação de $V^*(s)$ ou $Q^*(s,a)$ interagindo com o ambiente.  
- Q-Learning (Lecture 19) é a versão “aprendida” de Value Iteration.

Ou seja:

> **Value Iteration é o modelo teórico que inspira os algoritmos de RL.**

---

## 8. Conclusão da Parte 6

Nesta parte entendemos:

- como Value Iteration é um algoritmo de planejamento,  
- como MDPs se conectam ao controle ótimo,  
- como planejamento lida com incerteza,  
- como Value Iteration é a base conceitual do RL,  
- como isso prepara o terreno para Q-Learning.

Com isso, concluímos a Lecture 18.

Na Lecture 19 estudaremos:

- Q-Learning,  
- aprendizado de valores sem conhecer o ambiente,  
- exploração vs. exploração,  
- e como RL generaliza Value Iteration para ambientes reais.



<br><br>



<a id="lecture19_parte1"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 19 — Parte 1  
Do Planejamento ao Aprendizado — Motivação para Q-Learning  
=============================

Nesta lecture iniciamos a transição entre:

- **MDPs com modelo conhecido** (Lecture 17–18),  
- **Reinforcement Learning sem modelo** (Lecture 19).

O objetivo desta primeira parte é entender **por que precisamos de RL** e **por que Value Iteration não é suficiente** em ambientes reais.

---

## 1. O problema com Value Iteration

Value Iteration resolve MDPs quando conhecemos:

- todas as transições $T(s,a,s')$,  
- todas as recompensas $R(s,a,s')$,  
- todos os estados e ações.

Mas na prática:

> **O agente não conhece o ambiente.**

Ele deve aprender:

- quais ações são boas,  
- quais ações são ruins,  
- quais transições são prováveis,  
- quais recompensas existem.

Isso exige **exploração**.

---

## 2. Exemplos reais onde o modelo é desconhecido

### 2.1 Robótica  
O robô não sabe:

- como o atrito afeta o movimento,  
- como objetos respondem ao toque,  
- como o ambiente muda com o tempo.

### 2.2 Marketing  
A empresa não sabe:

- como o cliente reagirá a cada ação,  
- qual sequência de ações maximiza conversão.

### 2.3 Jogos  
O agente não sabe:

- como o adversário se comporta,  
- quais ações levam à vitória.

### 2.4 Navegação  
O agente não sabe:

- a topologia completa do ambiente,  
- onde estão obstáculos ou perigos.

Em todos esses casos:

> **O agente deve aprender interagindo com o ambiente.**

---

## 3. A diferença fundamental: modelo conhecido vs. desconhecido

### Planejamento (Value Iteration)
- usa $T(s,a,s')$ e $R(s,a,s')$ diretamente,  
- calcula valores sem experimentar,  
- funciona apenas quando o modelo é dado.

### Aprendizado (Q-Learning)
- não precisa de $T(s,a,s')$,  
- não precisa de $R(s,a,s')$,  
- aprende valores **experimentando**.

Essa é a grande mudança da Lecture 19.

---

## 4. O que o agente observa no RL?

O agente observa apenas:

- o estado atual $s$,  
- a ação tomada $a$,  
- o próximo estado $s'$,  
- a recompensa recebida $r$.

Ou seja:

> O agente vê **amostras** do ambiente, não o modelo completo.

---

## 5. O desafio: aprender valores sem saber transições

Value Iteration usa:

$$
\sum_{s'} T(s,a,s')\, V(s')
$$

Mas em RL:

- não sabemos $T(s,a,s')$,  
- não sabemos a distribuição de transições,  
- só vemos **um** próximo estado por vez.

Portanto:

> Precisamos de uma forma de aprender valores usando apenas amostras.

Essa forma é **Q-Learning**.

---

## 6. A ideia central que leva ao Q-Learning

A professora explica:

> “Se não sabemos o modelo, não podemos calcular o valor esperado diretamente.  
> Mas podemos **estimar** esse valor usando amostras.”

Ou seja:

- cada transição observada fornece uma estimativa parcial,  
- essas estimativas são acumuladas ao longo do tempo,  
- o valor converge para o valor ótimo.

Essa é a essência do aprendizado por reforço.

---

## 7. Conexão com Value Iteration

Value Iteration atualiza:

$$
V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V_k(s') \right]
$$

Q-Learning atualiza:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

A estrutura é a mesma, mas:

- Value Iteration usa o modelo,  
- Q-Learning usa amostras.

---

## 8. Conclusão da Parte 1

Nesta parte entendemos:

- por que Value Iteration não funciona em ambientes desconhecidos,  
- por que RL é necessário,  
- como agentes reais aprendem por interação,  
- como amostras substituem o modelo completo,  
- como isso motiva o surgimento do Q-Learning.

Na próxima parte estudaremos:

- a definição formal de Q-Learning,  
- a atualização temporal (TD update),  
- e como Q-Learning converge para $Q^*(s,a)$.



<br><br>



<a id="lecture19_parte2"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 19 — Parte 2  
Q-Learning — Atualização Temporal (TD Update)  
=============================

Nesta parte introduzimos o algoritmo central da Lecture 19: **Q-Learning**.  
Ele é a versão “aprendida” de Value Iteration — isto é, uma forma de aproximar $Q^*(s,a)$ **sem conhecer o modelo do ambiente**.

O objetivo é entender:

- o que é a função $Q(s,a)$,  
- como ela substitui $V(s)$,  
- como atualizá-la usando apenas amostras,  
- como isso leva ao aprendizado da política ótima.

---

## 1. Por que aprender $Q(s,a)$?

Em Value Iteration, precisamos de:

$$
T(s,a,s') \quad \text{e} \quad R(s,a,s')
$$

Mas em RL:

- não sabemos $T(s,a,s')$,  
- não sabemos $R(s,a,s')$,  
- só observamos **um** próximo estado por vez.

Portanto, não podemos calcular:

$$
\sum_{s'} T(s,a,s')\, V(s')
$$

A solução é aprender diretamente:

$$
Q(s,a)
$$

que representa:

> **O valor esperado de tomar ação $a$ no estado $s$ e seguir a política ótima depois disso.**

---

## 2. Definição da função $Q^*(s,a)$

Formalmente:

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} T(s,a,s')\, V^*(s')
$$

E:

$$
V^*(s) = \max_a Q^*(s,a)
$$

Ou seja:

- $Q^*(s,a)$ é mais informativo que $V^*(s)$,  
- $Q^*(s,a)$ permite escolher ações diretamente,  
- Q-Learning aprende $Q^*(s,a)$ sem conhecer o modelo.

---

## 3. A atualização temporal (TD Update)

Quando o agente observa uma transição:

- estado atual: $s$  
- ação tomada: $a$  
- recompensa recebida: $r$  
- próximo estado: $s'$  

ele atualiza:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

Essa é a **regra de atualização do Q-Learning**.

---

## 4. Intuição da atualização

A atualização compara:

- o valor atual $Q(s,a)$  
- com uma **estimativa melhor** baseada na experiência:

$$
r + \gamma \max_{a'} Q(s',a')
$$

O termo:

$$
\delta = r + \gamma \max_{a'} Q(s',a') - Q(s,a)
$$

é chamado de **erro temporal (TD error)**.

A atualização é:

$$
Q \leftarrow Q + \alpha \delta
$$

onde:

- $\alpha$ é a taxa de aprendizado (learning rate),  
- $0 < \alpha \le 1$.

---

## 5. Por que isso funciona?

Porque:

- cada transição observada fornece uma estimativa parcial de $Q^*(s,a)$,  
- essas estimativas são acumuladas ao longo do tempo,  
- o TD error corrige o valor atual,  
- o processo converge para $Q^*(s,a)$.

Q-Learning é, essencialmente:

> **Value Iteration aplicado sobre amostras.**

---

## 6. Q-Learning não precisa do modelo

A grande vantagem:

- não precisa de $T(s,a,s')$,  
- não precisa de $R(s,a,s')$,  
- aprende apenas com experiência.

O agente aprende enquanto:

- joga,  
- explora,  
- tenta ações,  
- observa recompensas.

---

## 7. Extração da política ótima

Depois que $Q(s,a)$ converge:

$$
\pi^*(s) = \arg\max_a Q(s,a)
$$

Ou seja:

> Escolha a ação com maior valor aprendido.

---

## 8. Conclusão da Parte 2

Nesta parte aprendemos:

- o que é a função $Q(s,a)$,  
- como ela substitui $V(s)$ em ambientes desconhecidos,  
- a regra de atualização temporal (TD update),  
- o papel do TD error,  
- como Q-Learning aprende $Q^*(s,a)$ sem conhecer o modelo.

Na próxima parte estudaremos:

- o papel da **exploração**,  
- o dilema exploração vs. exploração,  
- e a estratégia $\varepsilon$-greedy.



<Br><Br>



<a id="lecture19_parte3"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 19 — Parte 3  
Exploração vs. Exploração — Estratégia ε-greedy  
=============================

Nesta parte estudamos um dos dilemas fundamentais do aprendizado por reforço:

> **Como equilibrar explorar ações novas e explorar ações já conhecidas como boas?**

Esse dilema é chamado de **exploration vs. exploitation**.

Q-Learning só funciona bem se o agente explorar o ambiente de forma adequada.

---

## 1. O dilema central

O agente deve escolher entre:

### **Exploração**
- tentar ações novas,  
- descobrir recompensas desconhecidas,  
- aprender transições que ainda não foram observadas.

### **Exploração**
- escolher a melhor ação conhecida,  
- maximizar recompensa imediata,  
- seguir a política atual.

O problema é:

> Se o agente explorar demais, ele não aproveita o que já aprendeu.  
> Se explorar de menos, ele nunca aprende o suficiente.

---

## 2. Por que exploração é necessária?

Sem exploração:

- o agente pode ficar preso em políticas ruins,  
- nunca descobre ações melhores,  
- nunca aprende transições importantes.

Exemplo clássico:

- o agente tenta uma ação uma vez, recebe uma recompensa baixa,  
- mas essa ação poderia levar a uma recompensa alta em outro estado,  
- sem exploração, ele nunca descobrirá isso.

---

## 3. Estratégia ε-greedy

A solução mais usada é a estratégia **ε-greedy**.

Ela funciona assim:

- com probabilidade **ε**, o agente **explora** (escolhe uma ação aleatória),  
- com probabilidade **1 − ε**, o agente **explora** (escolhe a melhor ação segundo $Q$).

Formalmente:

$$
\pi(s) =
\begin{cases}
\text{ação aleatória}, & \text{com probabilidade } \varepsilon \\
\arg\max_a Q(s,a), & \text{com probabilidade } 1 - \varepsilon
\end{cases}
$$

---

## 4. Intuição da estratégia

A professora explica que:

> “ε-greedy mantém o agente curioso, mas não ingênuo.”

Ou seja:

- ele explora o suficiente para aprender,  
- mas explora o suficiente para aproveitar o que já sabe.

---

## 5. Escolha do valor de ε

Valores típicos:

- $\varepsilon = 0.1$ → 10% de exploração  
- $\varepsilon = 0.01$ → 1% de exploração  
- $\varepsilon = 0.3$ → exploração agressiva

A escolha depende:

- da complexidade do ambiente,  
- da quantidade de estados,  
- da necessidade de descobrir transições raras.

---

## 6. Decaimento de ε (ε-decay)

Uma estratégia comum é reduzir ε ao longo do tempo:

$$
\varepsilon_t = \varepsilon_0 \cdot \text{decay}^t
$$

Por exemplo:

- começa com $\varepsilon_0 = 0.3$,  
- reduz gradualmente até $\varepsilon = 0.01$.

Intuição:

> No início, o agente deve explorar bastante.  
> Depois, deve explorar menos e aproveitar o que aprendeu.

---

## 7. Conexão com Q-Learning

Q-Learning atualiza:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

Mas essa atualização só funciona bem se:

- o agente visitar todos os estados suficientes vezes,  
- o agente tentar todas as ações suficientes vezes.

ε-greedy garante isso.

---

## 8. Conclusão da Parte 3

Nesta parte entendemos:

- o dilema exploração vs. exploração,  
- por que exploração é essencial para RL,  
- como funciona a estratégia ε-greedy,  
- como escolher ε,  
- como usar ε-decay,  
- como isso garante que Q-Learning converge.

Na próxima parte estudaremos:

- a convergência formal do Q-Learning,  
- condições necessárias,  
- e por que Q-Learning encontra $Q^*(s,a)$.



<br><br>



<a id="lecture19_parte4"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 19 — Parte 4  
Convergência do Q-Learning  
=============================

Nesta parte estudamos **por que** o algoritmo de Q-Learning converge para a função de valor ótima de ações:

$$
Q^*(s,a)
$$

Assim como Value Iteration converge para $V^*(s)$, Q-Learning converge para $Q^*(s,a)$ — mas agora usando **apenas amostras** do ambiente, sem conhecer o modelo.

---

## 1. Relembrando a atualização do Q-Learning

Quando o agente observa uma transição:

- estado atual: $s$  
- ação tomada: $a$  
- recompensa recebida: $r$  
- próximo estado: $s'$  

ele atualiza:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

Esse é o **TD update** (Temporal Difference update).

---

## 2. O operador de Bellman para Q

O operador ótimo para Q é:

$$
(BQ)(s,a) = R(s,a) + \gamma \sum_{s'} T(s,a,s')\, \max_{a'} Q(s',a')
$$

O valor ótimo é o ponto fixo:

$$
Q^* = BQ^*
$$

Q-Learning tenta aproximar esse ponto fixo usando amostras.

---

## 3. Q-Learning como aproximação estocástica de Bellman

A atualização:

$$
r + \gamma \max_{a'} Q(s',a')
$$

é uma **amostra** da expectativa:

$$
R(s,a) + \gamma \sum_{s'} T(s,a,s')\, \max_{a'} Q(s',a')
$$

Portanto:

> Q-Learning é uma aproximação estocástica do operador de Bellman.

Cada transição observada fornece uma estimativa parcial do valor verdadeiro.

---

## 4. Condições para convergência

A professora destaca três condições essenciais:

### **1. Taxa de aprendizado adequada**

A sequência $\alpha_t$ deve satisfazer:

- $\sum_t \alpha_t = \infty$  
- $\sum_t \alpha_t^2 < \infty$

Na prática:

- $\alpha_t = \frac{1}{t}$ funciona,  
- ou $\alpha$ constante pequena (ex.: 0.1) funciona bem empiricamente.

### **2. Exploração suficiente**

O agente deve:

- visitar todos os estados suficientes vezes,  
- tentar todas as ações suficientes vezes.

Estratégias como **ε-greedy** garantem isso.

### **3. Fator de desconto**

$$
0 < \gamma < 1
$$

Isso garante que o operador é uma contração.

---

## 5. Teorema de Convergência

Sob as condições acima:

> **Q-Learning converge para $Q^*(s,a)$ com probabilidade 1.**

Ou seja:

- o aprendizado é garantido,  
- o valor ótimo é alcançado,  
- a política ótima pode ser extraída.

---

## 6. Intuição da convergência

A professora explica que:

> “Q-Learning é Value Iteration com ruído.”

- Cada atualização é uma versão ruidosa da atualização de Bellman.  
- O ruído vem da amostra única do próximo estado.  
- Com muitas amostras, o ruído se cancela.  
- A média converge para o valor verdadeiro.

É exatamente como estimar uma média com amostras aleatórias.

---

## 7. Convergência da política

Depois que $Q(s,a)$ converge:

$$
\pi^*(s) = \arg\max_a Q(s,a)
$$

Essa política é:

- ótima,  
- determinística,  
- equivalente à política de Value Iteration.

---

## 8. Conclusão da Parte 4

Nesta parte entendemos:

- como Q-Learning aproxima o operador de Bellman,  
- por que o TD update converge,  
- quais condições são necessárias para convergência,  
- o papel da exploração e da taxa de aprendizado,  
- como Q-Learning encontra $Q^*(s,a)$.

Na próxima parte estudaremos:

- exemplos concretos de Q-Learning em Gridworld,  
- como o agente aprende valores ao longo do tempo,  
- como a política ótima emerge da experiência.



<br><br>



<a id="lecture19_parte5"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 19 — Parte 5  
Q-Learning em Gridworld — Aprendizado por Experiência  
=============================

Nesta parte visualizamos como o algoritmo de **Q-Learning** funciona na prática, usando o ambiente clássico **Gridworld**.  
O objetivo é entender como:

- o agente aprende valores por tentativa e erro,  
- o TD update ajusta $Q(s,a)$ ao longo do tempo,  
- a política ótima emerge gradualmente,  
- o aprendizado difere de Value Iteration.

---

## 1. O ambiente Gridworld revisitado

O ambiente é o mesmo usado na Lecture 18:

- estados com recompensa $+1$ (objetivo),  
- estados com recompensa $-1$ (perigo),  
- estados com recompensa $0$ (vazios),  
- paredes (inacessíveis),  
- ações estocásticas (80% direção desejada, 10% desvios).

Mas agora:

> **O agente não conhece o modelo.**

Ele deve aprender tudo por experiência.

---

## 2. Inicialização de Q(s,a)

No início:

$$
Q(s,a) = 0
$$

para todos os estados e ações.

Isso significa:

- o agente não sabe nada,  
- todas as ações parecem igualmente boas,  
- exploração é essencial.

---

## 3. Episódios de aprendizado

O agente executa vários episódios:

1. começa em um estado inicial,  
2. escolhe ações usando ε-greedy,  
3. observa recompensas,  
4. atualiza $Q(s,a)$,  
5. termina quando chega em um estado terminal.

Cada episódio melhora a estimativa de $Q(s,a)$.

---

## 4. Atualização durante o episódio

Quando o agente observa:

- estado atual: $s$  
- ação tomada: $a$  
- recompensa: $r$  
- próximo estado: $s'$  

ele aplica:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

Essa atualização:

- corrige o valor da ação,  
- incorpora informação nova,  
- aproxima $Q(s,a)$ de $Q^*(s,a)$.

---

## 5. Propagação de valores por experiência

Diferente de Value Iteration:

- a propagação não ocorre globalmente,  
- ocorre apenas nos estados visitados,  
- depende da trajetória do agente.

Se o agente nunca visita um estado:

- ele nunca aprende seu valor,  
- por isso a exploração é crucial.

---

## 6. Como os valores emergem

Com o tempo:

- ações que levam ao objetivo recebem valores altos,  
- ações que levam ao perigo recebem valores negativos,  
- ações neutras recebem valores intermediários.

Exemplo:

Se o agente está perto do objetivo:

$$
Q(s,\text{ir para o objetivo}) \approx 0.9
$$

Se está perto do perigo:

$$
Q(s,\text{ir para o perigo}) \approx -0.9
$$

Esses valores são aprendidos **somente** pelas transições observadas.

---

## 7. Emergência da política ótima

Depois que $Q(s,a)$ converge:

$$
\pi^*(s) = \arg\max_a Q(s,a)
$$

A política ótima emerge naturalmente:

- estados próximos ao objetivo apontam para ele,  
- estados próximos ao perigo apontam para longe,  
- estados neutros escolhem caminhos que maximizam valor futuro.

A política final é idêntica à obtida por Value Iteration — mas aprendida por experiência.

---

## 8. Diferença fundamental entre Value Iteration e Q-Learning

### **Value Iteration**
- usa o modelo completo,  
- atualiza todos os estados simultaneamente,  
- é um algoritmo de planejamento.

### **Q-Learning**
- usa apenas amostras,  
- atualiza apenas estados visitados,  
- é um algoritmo de aprendizado.

A professora enfatiza:

> “Q-Learning é Value Iteration aplicado ao mundo real.”

---

## 9. Conclusão da Parte 5

Nesta parte entendemos:

- como Q-Learning funciona em Gridworld,  
- como valores são aprendidos por tentativa e erro,  
- como o TD update ajusta $Q(s,a)$,  
- como a política ótima emerge gradualmente,  
- como Q-Learning difere de Value Iteration.

Na próxima parte estudaremos:

- **SARSA**,  
- a diferença entre aprendizado **on-policy** e **off-policy**,  
- e como isso afeta o comportamento do agente.



<br><br>
<a id="lecture19_parte6"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 19 — Parte 6  
SARSA — Aprendizado On-Policy  
=============================

Nesta parte estudamos o algoritmo **SARSA**, uma alternativa ao Q-Learning.  
Enquanto Q-Learning é **off-policy**, SARSA é **on-policy** — e essa diferença muda profundamente o comportamento do agente.

O objetivo é entender:

- o que significa aprendizado on-policy,  
- como SARSA atualiza valores,  
- como ele difere de Q-Learning,  
- quando SARSA é preferível.

---

## 1. On-policy vs. Off-policy

### **Off-policy (Q-Learning)**  
O agente aprende a **política ótima**, mesmo que esteja seguindo outra política (ex.: ε-greedy).

Ele atualiza usando:

$$
\max_{a'} Q(s',a')
$$

Ou seja:

> Aprende como **agir de forma ótima**, mesmo que não esteja agindo assim agora.

### **On-policy (SARSA)**  
O agente aprende a **política que ele realmente executa**.

Ele atualiza usando a **ação que realmente tomou** no próximo estado.

Ou seja:

> Aprende a política **que está sendo seguida**, não a política ótima.

---

## 2. O nome SARSA

SARSA vem da sequência de elementos usados na atualização:

- **S**: estado atual  
- **A**: ação atual  
- **R**: recompensa  
- **S'**: próximo estado  
- **A'**: próxima ação  

A atualização depende de **A'**, a ação realmente escolhida.

---

## 3. Atualização do SARSA

Quando o agente observa:

- estado atual: $s$  
- ação tomada: $a$  
- recompensa: $r$  
- próximo estado: $s'$  
- próxima ação escolhida: $a'$  

ele atualiza:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s',a') - Q(s,a) \right]
$$

Compare com Q-Learning:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

A diferença está no termo:

- SARSA usa **$Q(s',a')$**  
- Q-Learning usa **$\max_{a'} Q(s',a')$**

---

## 4. Intuição da diferença

### **Q-Learning (off-policy)**  
Assume que o agente **sempre escolherá a melhor ação** no futuro.

### **SARSA (on-policy)**  
Assume que o agente **continuará seguindo sua política atual**, que pode incluir exploração.

Isso significa:

> SARSA leva em conta o comportamento exploratório do agente.

---

## 5. Consequência prática

SARSA tende a aprender políticas **mais seguras** em ambientes estocásticos.

Exemplo clássico:

- há um caminho curto até o objetivo, mas perigoso,  
- há um caminho longo, porém seguro.

### Q-Learning  
Assume que o agente sempre escolherá a melhor ação → tende a preferir o caminho curto.

### SARSA  
Considera que o agente pode explorar → tende a evitar o caminho perigoso.

---

## 6. SARSA com ε-greedy

Como SARSA é on-policy, a política usada para escolher ações deve ser a mesma usada na atualização.

A política típica é:

- ε-greedy para escolher ações,  
- e a mesma ε-greedy para atualizar $Q(s,a)$.

---

## 7. Convergência do SARSA

SARSA converge para a política ótima **da política exploratória**.

Se ε decai para zero:

- a política exploratória converge para a política ótima,  
- SARSA converge para $Q^*(s,a)$.

Se ε não decai:

- SARSA converge para a política ótima **com exploração**,  
- que pode ser mais conservadora.

---

## 8. Quando usar SARSA?

SARSA é preferível quando:

- o ambiente é perigoso,  
- ações ruins podem causar grandes perdas,  
- exploração pode levar a estados indesejados.

SARSA aprende políticas mais prudentes.

---

## 9. Conclusão da Parte 6

Nesta parte entendemos:

- a diferença entre aprendizado on-policy e off-policy,  
- como SARSA atualiza valores usando a ação realmente tomada,  
- como isso muda o comportamento do agente,  
- por que SARSA é mais seguro em ambientes estocásticos,  
- quando SARSA é preferível ao Q-Learning.

Na próxima parte estudaremos:

- **comparação direta entre Q-Learning e SARSA**,  
- exemplos concretos,  
- e implicações práticas para agentes reais.



<br><br>



<a id="lecture19_parte7"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 19 — Parte 7  
Comparação entre Q-Learning e SARSA  
=============================

Nesta parte comparamos diretamente os dois algoritmos de aprendizado por reforço vistos na Lecture 19:

- **Q-Learning** (off-policy)  
- **SARSA** (on-policy)

A professora explica que ambos aprendem funções de valor de ações, mas o comportamento e o tipo de política aprendida são diferentes.

---

## 1. Diferença fundamental

### **Q-Learning — Off-policy**

Atualiza usando:

$$
r + \gamma \max_{a'} Q(s',a')
$$

Ou seja:

> Assume que o agente **sempre escolherá a melhor ação** no futuro.

Ele aprende **a política ótima**, independentemente da política usada para explorar.

---

### **SARSA — On-policy**

Atualiza usando:

$$
r + \gamma Q(s',a')
$$

onde $a'$ é a **ação realmente tomada** no próximo estado.

Ou seja:

> Aprende **a política que está sendo executada**, incluindo exploração.

---

## 2. Consequência prática

### **Q-Learning**
- ignora o comportamento exploratório,  
- assume que o agente será ótimo no futuro,  
- tende a aprender políticas **mais agressivas**,  
- pode escolher caminhos arriscados.

### **SARSA**
- leva em conta a ação exploratória,  
- assume que o agente pode tomar ações subótimas,  
- tende a aprender políticas **mais conservadoras**,  
- evita caminhos perigosos quando há estocasticidade.

---

## 3. Exemplo clássico: o Cliff Walking

A professora usa o ambiente “Cliff Walking” para ilustrar:

- há um caminho curto até o objetivo, mas ao lado de um precipício,  
- há um caminho longo, porém seguro.

### **Q-Learning**
- aprende a política ótima teórica,  
- escolhe o caminho curto,  
- mas durante exploração pode cair no precipício.

### **SARSA**
- considera que o agente pode explorar,  
- evita o precipício,  
- escolhe o caminho longo e seguro.

Esse exemplo mostra claramente a diferença entre on-policy e off-policy.

---

## 4. Convergência

### **Q-Learning**
Converge para:

$$
Q^*(s,a)
$$

desde que:

- todas as ações sejam exploradas,  
- $\alpha$ decaia adequadamente.

### **SARSA**
Converge para:

$$
Q^{\pi}(s,a)
$$

onde $\pi$ é a política **ε-greedy** usada durante o aprendizado.

Se $\varepsilon \to 0$ com o tempo:

- SARSA também converge para $Q^*(s,a)$.

Se $\varepsilon$ é constante:

- SARSA converge para a política ótima **com exploração**.

---

## 5. Quando usar cada algoritmo?

### Use **Q-Learning** quando:
- o ambiente é seguro,  
- exploração não causa grandes perdas,  
- você quer a política ótima teórica,  
- o ambiente é determinístico ou pouco estocástico.

### Use **SARSA** quando:
- o ambiente é perigoso,  
- ações exploratórias podem causar danos,  
- você quer políticas mais prudentes,  
- o ambiente é altamente estocástico.

---

## 6. Resumo da comparação

| Aspecto | Q-Learning | SARSA |
|--------|------------|--------|
| Tipo | Off-policy | On-policy |
| Atualização | $\max_{a'} Q(s',a')$ | $Q(s',a')$ |
| Política aprendida | Ótima | Política executada |
| Comportamento | Agressivo | Conservador |
| Risco | Alto | Baixo |
| Convergência | $Q^*$ | $Q^\pi$ |

---

## 7. Intuição final

A professora resume:

> “Q-Learning aprende como agir **se fosse perfeito**.  
> SARSA aprende como agir **sabendo que pode cometer erros**.”

Essa diferença é essencial para escolher o algoritmo certo em aplicações reais.

---

## 8. Conclusão da Parte 7

Nesta parte entendemos:

- a diferença entre aprendizado on-policy e off-policy,  
- como Q-Learning e SARSA atualizam valores,  
- como isso afeta o comportamento do agente,  
- quando cada algoritmo é mais adequado,  
- e por que SARSA é mais seguro em ambientes estocásticos.

Na próxima parte estudaremos:

- **Expected SARSA**,  
- uma versão intermediária entre Q-Learning e SARSA,  
- que suaviza a atualização usando expectativas.



<br><br>



<a id="lecture19_parte8"></a>
[$\Uparrow$ Índice](#indice)  
[$\uparrow$ Índice unit5](#unit5)

=============================  
Lecture 19 — Parte 8  
Expected SARSA — Atualização Esperada  
=============================

Nesta parte estudamos **Expected SARSA**, um algoritmo que fica exatamente entre:

- **Q-Learning** (off-policy, agressivo)  
- **SARSA** (on-policy, conservador)

Expected SARSA suaviza a atualização usando **expectativas** sobre as ações futuras, em vez de usar:

- a melhor ação (como Q-Learning),  
- ou a ação realmente tomada (como SARSA).

---

## 1. Motivação para Expected SARSA

A professora explica que:

> “Q-Learning pode ser agressivo demais.  
> SARSA pode ser conservador demais.  
> Expected SARSA é o meio-termo.”

A ideia é:

- reduzir a variância da atualização,  
- considerar todas as ações possíveis no próximo estado,  
- ponderar essas ações pela política de exploração.

---

## 2. Atualização do Expected SARSA

Quando o agente observa:

- estado atual: $s$  
- ação tomada: $a$  
- recompensa: $r$  
- próximo estado: $s'$  

ele atualiza:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \, \mathbb{E}_{a' \sim \pi} \left[ Q(s',a') \right] - Q(s,a) \right]
$$

O termo novo é:

$$
\mathbb{E}_{a' \sim \pi} [Q(s',a')]
$$

que significa:

> A média ponderada dos valores das ações futuras, segundo a política atual.

---

## 3. Como calcular a expectativa

Se a política é **ε-greedy**, então:

- com probabilidade $1 - \varepsilon$, escolhemos a ação ótima,  
- com probabilidade $\varepsilon$, escolhemos uma ação aleatória.

Portanto:

$$
\mathbb{E}_{a' \sim \pi} [Q(s',a')] = (1 - \varepsilon) \max_{a'} Q(s',a') + \frac{\varepsilon}{|A|} \sum_{a'} Q(s',a')
$$

Essa expressão suaviza a atualização.

---

## 4. Comparação com Q-Learning e SARSA

### **Q-Learning**
Usa:

$$
\max_{a'} Q(s',a')
$$

→ agressivo, ignora exploração.

### **SARSA**
Usa:

$$
Q(s',a')
$$

→ conservador, depende da ação tomada.

### **Expected SARSA**
Usa:

$$
\mathbb{E}_{a' \sim \pi} [Q(s',a')]
$$

→ intermediário, pondera todas as ações.

---

## 5. Consequências práticas

Expected SARSA:

- reduz variância,  
- é mais estável que Q-Learning,  
- é menos conservador que SARSA,  
- converge mais suavemente,  
- funciona bem em ambientes estocásticos.

A professora destaca:

> “Expected SARSA é frequentemente mais estável que Q-Learning.”

---

## 6. Intuição da atualização

Expected SARSA considera:

- que o agente pode explorar,  
- mas também pode escolher ações boas,  
- e pondera essas possibilidades.

Isso evita:

- atualizações muito otimistas (Q-Learning),  
- atualizações muito pessimistas (SARSA).

---

## 7. Convergência

Expected SARSA converge para:

- a política ótima se $\varepsilon \to 0$,  
- a política ε-greedy se $\varepsilon$ é constante.

Assim como SARSA, ele é **on-policy**, mas com menor variância.

---

## 8. Quando usar Expected SARSA?

Use Expected SARSA quando:

- o ambiente é estocástico,  
- você quer estabilidade,  
- Q-Learning oscila demais,  
- SARSA é conservador demais.

É muito usado em:

- robótica,  
- navegação,  
- ambientes com ruído,  
- simulações físicas.

---

## 9. Conclusão da Parte 8

Nesta parte entendemos:

- o que é Expected SARSA,  
- como ele suaviza a atualização usando expectativas,  
- como ele se posiciona entre Q-Learning e SARSA,  
- por que ele reduz variância,  
- quando ele é preferível,  
- como ele converge.

Com isso, concluímos a Lecture 19.

A próxima lecture (Lecture 20) introduzirá:

- **Model-based RL**,  
- **Dyna-Q**,  
- e como combinar planejamento com aprendizado.



<br><br>


[$\Uparrow$ Índice](#indice)  
