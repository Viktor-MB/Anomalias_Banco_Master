# 🔍 Análise Forense com IA no Caso Banco Master

## 📰 Resumo do Projeto
Este repositório contém uma investigação de **Análise Forense Quantitativa** aplicada ao mercado de FIDCs (Fundos de Investimento em Direitos Creditórios). O objetivo central foi validar se um pipeline de **Machine Learning** poderia ter detectado anomalias sistemáticas nos grupos **Master, Trustee, Reag e Letsbank** meses antes das intervenções do Banco Central do Brasil.

O diferencial deste projeto é a entrega: os resultados são apresentados através de um Web App em **Streamlit** com estética de **Jornal Investigativo Moderno**, convertendo métricas estatísticas em uma narrativa acessível para auditoria e tomada de decisão.

---

## 🎯 O Problema Investigado
Entre novembro de 2025 e fevereiro de 2026, oito instituições ligadas ao Grupo Master sofreram liquidação extrajudicial. O foco da investigação foi analisar:
1. **Conflitos de Interesse:** Cedentes e administradores pertencentes ao mesmo grupo econômico.
2. **Ocultação de Risco:** Uso de altas taxas de giro de carteira (aquisição de novos títulos) para mascarar ativos inadimplentes.
3. **Pontos Cegos Regulatórios:** Ausência sistemática de declaração de CNPJs de cedentes à CVM em fundos críticos.

---

## 🛠️ Como foi feito (Metodologia)

### 1. Extração e Tratamento (Pipeline de Dados)
Foram utilizados dados públicos do **Portal de Dados Abertos da CVM**. O pipeline processou os informes mensais de março a novembro de 2025, focando em **8 indicadores forenses**:
* **Inadimplência Declarada:** Relação entre atrasos e Patrimônio Líquido (PL).
* **Concentração de Cedente:** Exposição ao maior devedor da carteira.
* **Giro de Carteira (Taxa de Aquisição):** Volume de novos direitos creditórios comprados.
* **Taxa de Devolução:** Recompras feitas pelo cedente (sinal de substituição de "ativos podres").
* **Ratio Inadimplência/Giro:** Indicador de "maquiagem" de balanço.
* **Volatilidade da Inadimplência:** Desvio-padrão histórico da inadimplência por fundo — fundos fraudulentos apresentam variação artificialmente baixa.
* **Volatilidade da Taxa de Aquisição:** Desvio-padrão histórico do giro de carteira por fundo.
* **Inadimplência Congelada:** Flag que identifica quando a inadimplência declarada é exatamente igual à do mês anterior por múltiplos períodos — comportamento estatisticamente improvável em condições normais.

### 2. O Algoritmo: CatBoost Supervisionado
Para detectar as fraudes, utilizei o **CatBoost Classifier**, um algoritmo de **aprendizado supervisionado** baseado em Gradient Boosting.

* **Por que esta técnica?** Ao contrário de abordagens não supervisionadas, o CatBoost aprende diretamente a partir de casos confirmados: os grupos investigados (Master, Trustee, Reag, Letsbank) são usados como **classe positiva (anomalia)** e o mercado geral ativo como **classe negativa (normal)**. Isso torna o modelo significativamente mais preciso e interpretável.
* **Treinamento Supervisionado:** O modelo recebe rótulos binários — `1` para registros dos grupos investigados e `0` para o mercado de referência — aprendendo os padrões combinados que caracterizam a fraude.
* **Balanceamento de Classes:** Como o mercado é muito maior que os grupos investigados, é aplicado `scale_pos_weight` proporcional ao desequilíbrio, evitando viés para a classe majoritária.
* **Normalização:** `RobustScaler` calibrado na mediana do mercado ativo, mais resistente a outliers extremos do que o `StandardScaler`.

#### Configuração do Modelo:
```python
CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    scale_pos_weight=scale_pos_weight,  # balanceamento automático
    eval_metric='AUC',
    random_seed=42,
)
```

O **score final** de cada fundo é a probabilidade estimada de pertencer aos grupos investigados — de `0` (comportamento normal) a `1` (máxima semelhança com os suspeitos).

---

## 💡 Principais Insights Forenses
* **Antecipação do Sinal:** O algoritmo detectou scores críticos no **Banco Master S.A.** meses antes da liquidação pelo Banco Central, operando sistematicamente **98% acima da média de suspeição do mercado** ao longo de 2025.
* **Casos Extremos (Outliers):** O "NF Fundo" (Trustee) e o "Engelberg 41" (Reag) atingiram scores de **0.834** e **0.736**, respectivamente, figurando como os pontos mais anômalos de toda a base de dados nacional.
* **Paradoxo da Inadimplência Zero:** Detectamos fundos com giro de carteira superior a 20% ao mês e inadimplência declarada de 0%. A combinação com **volatilidade zero** — inadimplência congelada por meses consecutivos — provou ser estatisticamente impossível em condições normais, com probabilidade de ocorrência casual estimada em menos de **0,3%**.
* **Impacto Estimado:** R$ 51,8 bilhões em exposição ao FGC ao longo das 8 instituições liquidadas.

---

## 💻 Stack Tecnológica
* **Linguagem:** Python
* **Machine Learning:** `CatBoost` (CatBoostClassifier — Gradient Boosting Supervisionado), `Scikit-learn` (RobustScaler)
* **Processamento:** `Pandas`, `NumPy`
* **Visualização:** `Plotly` (Gráficos interativos e Violin Plots para análise de densidade)
* **Interface:** `Streamlit` (Layout Jornalístico com CSS Customizado)

--- 

> **Disclaimer:** Este projeto tem finalidade estritamente acadêmica e investigativa. Todas as análises baseiam-se em dados públicos da CVM e metodologias estatísticas reprodutíveis. Não constitui recomendação de investimento
